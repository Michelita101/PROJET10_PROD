import os
import json
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
USERS_PATH = os.path.join(BASE_DIR, "data", "streamlit_users_demo.parquet")

if not os.path.exists(USERS_PATH):
    st.error("Fichier des utilisateurs de démo introuvable.")
    st.stop()

df_users = pd.read_parquet(USERS_PATH)
user_ids = df_users["user_id"].dropna().astype(int).sort_values().tolist()



# Config & style

st.set_page_config(
    page_title="News Reco — Projet 10",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { border-radius: 10px; padding: 0.6rem 1rem; }
    .card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: rgba(255,255,255,0.02);
    }
    .muted { opacity: 0.75; font-size: 0.9rem; }
    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        border: 1px solid rgba(49, 51, 63, 0.25);
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Helpers

def normalize_user_id(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False, ttl=60)
def call_reco_api(base_url, user_id, strategy, top_k, http_method="GET", timeout=15):
    """
    Appelle l'API Azure Functions / local.
    Retourne: (status_code, payload_json_or_text, elapsed_ms)
    """
    t0 = time.time()

    params = {
        "user_id": "" if user_id is None else str(user_id),
        "strategy": strategy,
        "top_k": str(top_k),
    }

    try:
        if http_method.upper() == "POST":
            # On envoie en JSON (tout en gardant params possible si tu veux)
            resp = requests.post(
                base_url,
                json={"user_id": user_id, "strategy": strategy, "top_k": top_k},
                timeout=timeout,
            )
        else:
            resp = requests.get(base_url, params=params, timeout=timeout)

        elapsed_ms = int((time.time() - t0) * 1000)

        # Essai JSON
        try:
            return resp.status_code, resp.json(), elapsed_ms
        except Exception:
            return resp.status_code, resp.text, elapsed_ms

    except requests.exceptions.RequestException as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        return 0, {"error": str(e)}, elapsed_ms


def coerce_results_to_table(payload):
    """
    Ton API renvoie actuellement une liste de dicts:
    [{"user_id":..., "article_id":..., "rank":..., "score":..., "engine":...}, ...]
    On convertit ça en DataFrame.
    """
    if isinstance(payload, list):
        if len(payload) == 0:
            return None, None, pd.DataFrame(columns=["article_id", "rank", "score"])
        engine = payload[0].get("engine")
        user_id = payload[0].get("user_id")
        rows = []
        for r in payload:
            rows.append({
                "article_id": r.get("article_id"),
                "rank": r.get("rank"),
                "score": r.get("score"),
            })
        df = pd.DataFrame(rows).sort_values("rank")
        return user_id, engine, df

    # Si un jour tu changes le format (dict wrapper), on le supporte aussi.
    if isinstance(payload, dict) and "recommendations" in payload:
        user_id = payload.get("user_id")
        engine = payload.get("engine")
        df = pd.DataFrame(payload.get("recommendations", []))
        if not df.empty and "rank" in df.columns:
            df = df.sort_values("rank")
        return user_id, engine, df

    return None, None, None


def render_recos(user_id, engine, df, show_scores=True):
    if df is None or df.empty:
        st.warning("Aucune recommandation à afficher.")
        return

    # Bandeau “résumé”
    cols = st.columns([2, 2, 2, 4])
    cols[0].metric("User ID", "—" if user_id is None else str(user_id))
    cols[1].metric("Engine", "—" if engine is None else str(engine))
    cols[2].metric("Top-K", str(len(df)))

    st.markdown("")

    # Cartes
    for _, row in df.iterrows():
        article_id = row.get("article_id")
        rank = row.get("rank")
        score = row.get("score")

        st.markdown(
            f"""
            <div class="card">
              <div>
                <span class="pill">Rank #{int(rank)}</span>
                <span class="pill">Article ID: <b>{int(article_id)}</b></span>
                {"<span class='pill'>Score: <b>"+str(round(float(score), 6))+"</b></span>" if show_scores and score is not None else ""}
              </div>
              <div class="muted">Astuce: tu pourras remplacer l’ID par un titre dès que tu ajoutes un mapping article_id → title.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def load_user_ids_from_upload(uploaded_file):
    """
    Supporte CSV et Parquet uploadés.
    On attend une colonne 'user_id'.
    """
    if uploaded_file is None:
        return []

    name = uploaded_file.name.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        # CSV par défaut
        df = pd.read_csv(uploaded_file)

    if "user_id" not in df.columns:
        return []

    user_ids = (
        df["user_id"]
        .dropna()
        .astype(int, errors="ignore")
        .tolist()
    )

    # dédoublonnage en gardant l'ordre
    seen = set()
    uniq = []
    for u in user_ids:
        try:
            u2 = int(u)
        except Exception:
            continue
        if u2 not in seen:
            uniq.append(u2)
            seen.add(u2)

    return uniq


# UI

st.title("📰 Recommandation d’articles — Demo (Projet 10)")
st.caption("Interface Streamlit pour appeler ton Azure Function et afficher les 5 recommandations.")

with st.sidebar:
    st.header("⚙️ Configuration")

    base_url = st.text_input(
        "URL de l’API (Azure Function / local)",
        value="https://p10-reco-api-michele.azurewebsites.net/api/recommend",
        help="En local: http://localhost:7071/api/recommend. En prod: https://p10-reco-api-michele.azurewebsites.net/api/recommend.",
    )

    http_method = st.selectbox("Méthode HTTP", ["GET", "POST"], index=0)

    st.divider()
    st.subheader("👤 Sélection utilisateur")

    # 1-Upload de fichier (CSV/Parquet <200MB)
    uploaded = st.file_uploader(
        "Optionnel: uploader un fichier avec une colonne user_id (CSV ou Parquet)",
        type=["csv", "parquet"],
    )

    uploaded_user_ids = load_user_ids_from_upload(uploaded) if uploaded else []

    # 2-Selectbox des users embarqués (parquet de démo)
    selected_demo_user = st.selectbox(
        "Ou choisir un utilisateur existant (échantillon de démo)",
        options=["(aucun)"] + user_ids,
        index=0,
    )

    # 3-Saisie manuelle
    manual_user = st.text_input(
        "Ou saisir un user_id",
        value="",
        help="Ex: 10, 5890, 999999. Laisser vide pour simuler la page d’accueil (user_id=None).",
    )

    # 4-Règle de priorité
    user_id = None
    user_id_list = None

    if uploaded_user_ids:
        user_id_list = uploaded_user_ids
        st.caption(f"{len(uploaded_user_ids)} user_id chargés depuis le fichier.")

    elif manual_user.strip() != "":
        try:
            user_id = int(manual_user)
        except ValueError:
            st.warning("Le user_id saisi doit être un entier.")

    elif selected_demo_user != "(aucun)":
        user_id = int(selected_demo_user)
        

    st.divider()
    st.subheader("🧠 Stratégie")
    strategy = st.selectbox(
        "strategy",
        options=["auto", "content_based", "cf_item", "cf_global", "hybrid", "als"],
        index=0,
        help="auto = routing MVP. Les autres = debug / démonstration.",
    )

    top_k = st.slider("top_k", min_value=1, max_value=20, value=5, step=1)

    st.divider()
    st.subheader("🧾 Affichage")
    show_scores = st.checkbox("Afficher les scores", value=True)
    show_raw = st.checkbox("Afficher la réponse brute JSON", value=False)

    st.divider()
    timeout = st.slider("Timeout (secondes)", 3, 60, 15, 1)


# Action

colA, colB = st.columns([1, 2])

with colA:
    run = st.button("🚀 Recommander", use_container_width=True)

with colB:
    st.markdown(
        f"<div class='muted'>Dernier test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )

if run:
    with st.spinner("Appel de l’API en cours…"):
        status, payload, elapsed_ms = call_reco_api(
            base_url=base_url,
            user_id=user_id,
            strategy=strategy,
            top_k=top_k,
            http_method=http_method,
            timeout=timeout,
        )

    st.markdown("")
    left, right = st.columns([2, 1])
    left.markdown(f"### Résultat")
    right.metric("Latence", f"{elapsed_ms} ms")

    if status == 0:
        st.error(f"Impossible de joindre l’API. Détails: {payload}")
        st.stop()

    if status >= 400:
        st.error(f"Erreur API (HTTP {status})")
        if isinstance(payload, dict):
            st.json(payload)
        else:
            st.code(str(payload))
        st.stop()

    # Normalisation affichage
    u, eng, df = coerce_results_to_table(payload)

    # Affichage principal
    render_recos(u, eng, df, show_scores=show_scores)

    # Réponse brute
    if show_raw:
        st.markdown("### Réponse brute")
        if isinstance(payload, (dict, list)):
            st.json(payload)
        else:
            st.code(str(payload))

else:
    st.info(
        "Choisis un `user_id` et clique sur **Recommander**. "
        "Astuce: laisse `user_id` vide pour simuler la page d’accueil (cold start)."
    )
