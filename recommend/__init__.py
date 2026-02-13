import azure.functions as func
import json
import logging
import os
import io
import joblib
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient

#import implicit

import scipy.sparse
import gzip
from sklearn.metrics.pairwise import cosine_similarity

# Configuration initiale
BLOB_CONNECTION_STRING = os.environ.get("BLOB_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "processed"

# Fichiers à charger depuis Azure Blob
FILES_TO_LOAD = {
    "trending_items": "trending_top_items.json",
    "interactions_df": "interactions.csv",
    "articles_df": "articles_metadata_min.csv",
    "embeddings_cb": "articles_embeddings_cb.npy",
    "item_similarity_df": "item_similarity_df.parquet",
    # "model_als": "model_als.pkl",
    # "user_item_sparse": "user_item_sparse.npz",
    "user_item_matrix": "user_item_matrix.csv.gz",
    # "user_id_map": "user_id_map.pkl",
    # "item_id_map": "item_id_map.pkl"
}

# Variables globales
trending_items = []
interactions_df = None
articles_df = None
embeddings_cb = None
item_similarity_df = None
model_als = None
user_item_sparse = None
user_id_map = None
item_id_map = None
user_item_matrix = None

# Chargement des artefacts

artifacts_loaded = False

def load_artifacts():
    global artifacts_loaded
    global trending_items, interactions_df, articles_df, embeddings_cb, item_similarity_df
    global user_item_matrix
    # global model_als, user_item_sparse, user_id_map, item_id_map, 

    if artifacts_loaded:
        return

    try:
        logging.info("Chargement des artefacts depuis Azure Blob Storage...")
        blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client = blob_service.get_container_client(CONTAINER_NAME)

        for key, blob_name in FILES_TO_LOAD.items():
            blob_client = container_client.get_blob_client(blob_name)
            blob_data = blob_client.download_blob().readall()

            # JSON
            if key == "trending_items":
                trending_items = json.loads(blob_data).get("items", [])

            # CSV
            elif key in ["interactions_df", "articles_df"]:
                globals()[key] = pd.read_csv(io.BytesIO(blob_data))
        
            # CSV .gz
            elif key == "user_item_matrix":
                globals()[key] = pd.read_csv(io.BytesIO(blob_data), compression='gzip')

            # Parquet
            elif key == "item_similarity_df":
                globals()[key] = pd.read_parquet(io.BytesIO(blob_data))

            # Numpy .npy
            elif key == "embeddings_cb":
                globals()[key] = np.load(io.BytesIO(blob_data), allow_pickle=True)

            # Numpy .npz
            #elif key == "user_item_sparse":
                #with open("/tmp/user_item_sparse.npz", "wb") as f:
                    #f.write(blob_data)
                #globals()[key] = scipy.sparse.load_npz("/tmp/user_item_sparse.npz")
                
            # Pickle
            #elif key in ["model_als", "user_id_map"]:
                #globals()[key] = joblib.load(io.BytesIO(blob_data))

            # Pickle
            #elif key == "item_id_map":
                #item_id_map = joblib.load(io.BytesIO(blob_data))
                #globals()[key] = item_id_map
        
        artifacts_loaded = True

    except Exception as e:
        logging.error(f":-( Erreur lors du chargement des artefacts : {e}")


# Format de sortie standardisé

def format_x_results(user_id, article_ids, scores, engine):
    return [
        {
            "user_id": int(user_id) if user_id is not None else None,
            "article_id": int(article_id),
            "rank": int(rank + 1),
            "score": float(score),
            "engine": engine
        }
        for rank, (article_id, score) in enumerate(zip(article_ids, scores))
    ]

# CF global
def format_results(user_id, article_ids):
    # Score artificiel mais honnête : décroissant avec le rang afin de l'adapter à cf_global
    scores = [1 / (rank + 1) for rank in range(len(article_ids))]
    return format_x_results(
        user_id=user_id,
        article_ids=article_ids,
        scores=scores,
        engine="cf_global"
    )


# Ajout des wrappers

# Moteur Content-Based Filtering
def recommend_cb_wrapper(user_id, top_k=5):
    if user_id not in interactions_df["user_id"].values:
        return [], []

    last_click = (
        interactions_df[interactions_df["user_id"] == user_id]
        .sort_values("click_timestamp", ascending=False)
        .iloc[0]["article_id"]
    )

    if last_click >= embeddings_cb.shape[0]:
        return [], []

    # Similarité cosinus
    similarities = cosine_similarity(
        [embeddings_cb[last_click]],
        embeddings_cb
    ).flatten()

    similarities[last_click] = -1

    # Normalisation min-max
    min_s = similarities.min()
    max_s = similarities.max()

    if max_s > min_s:
        similarities = (similarities - min_s) / (max_s - min_s)
    else:
        similarities = np.zeros_like(similarities)

    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_articles = top_indices.tolist()
    top_scores = similarities[top_indices]

    return top_articles, top_scores

# Moteur Collaborative Filtering Item-Based
def recommend_cf_wrapper(user_id, top_k=5):
    if user_id not in interactions_df["user_id"].values:
        return [], []

    clicked_articles = interactions_df[interactions_df["user_id"] == user_id]["article_id"].tolist()

    if not clicked_articles:
        return [], []

    # Somme des similarités de tous les articles cliqués
    scores = item_similarity_df.loc[clicked_articles].sum()

    scores = scores.drop(labels=clicked_articles, errors='ignore')  # retirer les articles déjà vus

    # Normalisation min-max
    min_s = scores.min()
    max_s = scores.max()
    if max_s > min_s:
        scores = (scores - min_s) / (max_s - min_s)
    else:
        scores = scores * 0

    # Tri
    scores = scores.sort_values(ascending=False)

    top_articles = scores.head(top_k).index.tolist()
    top_scores = scores.head(top_k).values.tolist()

    return top_articles, top_scores

# Collaborative Filtering global
def recommend_cf_global(user_id, top_k=5):
    article_ids = trending_items[:top_k]
    return format_results(user_id, article_ids)

# Moteur ALS implicit
def recommend_als_wrapper(user_id, top_k=5):
    try:
        if user_id_map is None or item_id_map is None or model_als is None:
            return []
        
        internal_user_id = user_id_map.get(user_id, None)
        if internal_user_id is None:
            return []

        internal_user_id = int(internal_user_id.item() if isinstance(internal_user_id, np.generic) else internal_user_id)

        user_items_row = user_item_sparse[internal_user_id]
        if not isinstance(user_items_row, scipy.sparse.csr_matrix):
            user_items_row = user_items_row.tocsr()

        item_ids, scores = model_als.recommend(
            userid=internal_user_id,
            user_items=user_items_row,
            N=top_k,
            filter_already_liked_items=True
        )

        reverse_map = {v: k for k, v in item_id_map.items()}
        articles = []
        scores_clean = []

        for rank, (i, score) in enumerate(zip(item_ids, scores)):
            if int(i) in reverse_map:
                articles.append(int(reverse_map[int(i)]))
                scores_clean.append(float(score))

        # Normalisation min-max ALS
        if scores_clean:
            min_s = min(scores_clean)
            max_s = max(scores_clean)

            if max_s > min_s:
                scores_clean = [
                    (s - min_s) / (max_s - min_s)
                    for s in scores_clean
                ]
            else:
                scores_clean = [0 for _ in scores_clean]

        return format_x_results(
            user_id=user_id,
            article_ids=articles,
            scores=scores_clean,
            engine="als"
        )
    
    except Exception as e:
        logging.error(f"ALS error for user {user_id} → {e}")
        return []

# Recommend

def get_user_session_count(user_id, interactions_df):
    if user_id is None:
        return 0
    user_sessions = interactions_df.loc[
        interactions_df["user_id"] == user_id, "session_id"
    ]
    return user_sessions.nunique()

def get_user_click_count(user_id, interactions_df):
    if user_id is None:
        return 0
    return len(
        interactions_df[interactions_df["user_id"] == user_id]
    )

def merge_hybrid(cb_articles, cb_scores, cf_articles, cf_scores, top_k, click_count):

    # Pondérations (modifiables plus tard dynamiquement)
    if click_count < 3:
        w_cb = 0.7
        w_cf = 0.3
    else:
        w_cb = 0.5
        w_cf = 0.5

    scores_dict = {}

    # Ajouter scores CB
    for a, s in zip(cb_articles, cb_scores):
        scores_dict[a] = w_cb * s
    #Ajouter scores CF
    for a, s in zip(cf_articles, cf_scores):
        if a in scores_dict:
            scores_dict[a] += w_cf * s
        else:
            scores_dict[a] = w_cf * s

    # Tri final
    sorted_items = sorted(
        scores_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    articles = [int(a) for a, _ in sorted_items[:top_k]]
    scores = [float(s) for _, s in sorted_items[:top_k]]

    return articles, scores

def recommend(
    user_id,
    strategy="auto",
    top_k=5
):

    # Normalisation du user_id
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        user_id = None

    # Nombre de sessions utilisateur
    click_count = get_user_click_count(user_id, interactions_df)

    # 1er cas : STRATÉGIES FORCÉES

    if strategy != "auto":

        if strategy == "content_based":
            articles, scores = recommend_cb_wrapper(user_id, top_k)
            return format_x_results(
                user_id=user_id,
                article_ids=articles,
                scores=scores,
                engine="content_based"
            )

        if strategy == "cf_item":
            articles, scores = recommend_cf_wrapper(user_id, top_k)
            return format_x_results(
                user_id=user_id,
                article_ids=articles,
                scores=scores,
                engine="cf_item"
            )

        if strategy == "cf_global":
            return recommend_cf_global(user_id, top_k)

        if strategy == "hybrid":
            cb_articles, cb_scores = recommend_cb_wrapper(user_id, top_k=top_k)
            cf_articles, cf_scores = recommend_cf_wrapper(user_id, top_k=top_k)

            articles, scores = merge_hybrid(
                cb_articles, cb_scores, cf_articles, cf_scores, top_k, click_count
            )

            return format_x_results(
                user_id=user_id,
                article_ids=articles,
                scores=scores,
                engine="hybrid"
            )

        #if strategy == "als":
            #return recommend_als_wrapper(user_id, top_k)

        raise ValueError(f"Unknown strategy: {strategy}")


    # 2e cas : STRATÉGIE AUTO

    ALS_CLICK_THRESHOLD = 10

    # O clic = Cold start : user inconnu ou sans clic
    if click_count == 0:
        return recommend_cf_global(user_id, top_k)

    # 1 clic = Content Based pur
    if click_count == 1:
        articles, scores = recommend_cb_wrapper(user_id, top_k)
        return format_x_results(
            user_id=user_id,
            article_ids=articles,
            scores=scores,
            engine="content_based"
        )

    # ALS désactivé en production (incompatibilité Azure implicit)
    # Architecture prête à réactivation en environnement compatible
    # Historique dense → ALS auto
    #if click_count >= ALS_CLICK_THRESHOLD:
        #results = recommend_als_wrapper(user_id, top_k)
        #if results:
            #return results
        # si ALS échoue, on continue vers hybrid

    # Cas intermédiaire = Hybrid
    cb_articles, cb_scores = recommend_cb_wrapper(user_id, top_k=top_k)
    cf_articles, cf_scores = recommend_cf_wrapper(user_id, top_k=top_k)

    articles, scores = merge_hybrid(
        cb_articles, cb_scores, cf_articles, cf_scores, top_k, click_count
    )
        
    return format_x_results(
        user_id=user_id,
        article_ids=articles,
        scores=scores,
        engine="hybrid"
    )


# Point d'entrée API, endpoint HTTP

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.warning("=== FUNCTION STARTED ===")
    try:
        load_artifacts()
        logging.warning(f"DEBUG interactions_df is None? {interactions_df is None}")
        logging.warning(f"DEBUG embeddings_cb is None? {embeddings_cb is None}")
        user_id = req.params.get("user_id")
        strategy = req.params.get("strategy", "auto")
        top_k = int(req.params.get("top_k", 5))

        # Normalisation
        try:
            user_id = int(user_id)
        except (TypeError, ValueError):
            user_id = None

        if not trending_items:
            return func.HttpResponse(
                json.dumps({
                    "user_id": user_id,
                    "strategy": strategy,
                    "reason": "trending_items_not_loaded",
                    "recommendations": []
                }),
                mimetype="application/json",
                status_code=200
            )

        # MVP : reco simple CF global
        results = recommend(user_id=user_id, strategy=strategy, top_k=top_k)

        return func.HttpResponse(
            json.dumps(results),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f":-( Erreur dans recommend_endpoint : {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )




