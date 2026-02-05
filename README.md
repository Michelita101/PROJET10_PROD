---
# 📰 Projet 10 – Application de recommandation de contenu


Ce projet consiste à concevoir et déployer une application complète de recommandation d’articles, intégrant plusieurs moteurs de recommandation et une interface de démonstration interactive.

---

## 🎯 Objectifs
- Construire plusieurs moteurs de recommandation :
  - Content-Based Filtering
  - Collaborative Filtering (item-based)
  - Moteur hybride (CB + CF)
  - ALS (Collaborative Filtering matriciel)
- Exposer l’inférence via une API (Azure Functions)
- Proposer une interface utilisateur pour tester les recommandations
- Déployer une application fonctionnelle en ligne

---

## 🧠 Architecture globale

- **API d’inférence** : Azure Functions  
- **Stockage des artefacts** : Azure Blob Storage  
- **Interface utilisateur** : Streamlit  
- **Déploiement front** : Streamlit Cloud  

Utilisateur  
⬇️  
Application Streamlit Cloud  
⬇️  
API Azure Functions   
⬇️  
Moteurs de recommandation  
⬇️  
Azure Blob Storage

---

## ⚙️ Fonctionnalités principales

- Sélection utilisateur via :
  - liste d’utilisateurs embarquée (échantillon de démonstration)
  - saisie manuelle d’un `user_id`
  - upload d’un fichier CSV ou Parquet (< 200 MB)
- Choix de la stratégie :
  - auto : routing MVP basé sur l’historique utilisateur
  - content_based
  - cf_item
  - cf_global
  - hybrid
- Paramétrage de l’inférence :
  - nombre de recommandations (`top_k`)
  - timeout de l’appel API (gestion du temps de réponse et des appels longs)
- Affichage des résultats :
  - recommandations classées par rang
  - affichage optionnel des scores
  - affichage de la réponse brute JSON (mode debug)


>⚠️ **Remarque sur ALS**  
>Le moteur ALS est implémenté.
>Il n’est pas activé dans la version déployée sur Azure en raison des contraintes de compatibilité de la librairie `implicit` avec une architecture serverless.
>En environnement industriel, ce moteur serait déployé via un service dédié (VM, batch ou microservice spécialisé).

---

## 📂 Structure du projet

```
PROJET10_PROD/
├── app.py                  # Application Streamlit
├── recommend/
│   ├── __init__.py         # API Azure Functions (endpoint recommend)
│   └── function.json
├── data/
│   └── streamlit_users_demo.parquet
├── requirements.txt
├── README.md

```

---

## 🚀 Lancer le projet en local

### 1. API Azure Functions
```bash
func start
```
Endpoint disponible :   
http://localhost:7071/api/recommend

### 2. Application Streamlit

```bash
streamlit run app.py
```

---

## 🌐 Déploiement

- L’API d'inférence est déployée via **Azure Functions**.
- L’interface utilisateur est déployée via **Streamlit Cloud**.
- Le stockage des artefacts (modèles, matrices utilisateurs-items, embeddings, fichiers de similarité et liste d'utilisateurs connus) est assuré par **Azure Blob Storage**.

---

## 🔗 Points d’entrée de l’application

- **API Azure Functions (inférence)**  
https://p10-reco-api-michele.azurewebsites.net/api/recommend

Paramètres optionnels supportés :
- user_id
- strategy (auto, content_based, cf_item, cf_global, hybrid)
- top_k (nombre de recommandations)  

Exemple :  
https://p10-reco-api-michele.azurewebsites.net/api/recommend?user_id=13&strategy=auto&top_k=5

- **Application Streamlit (interface de démonstration)**  
https://projet10app-ghju8p6mp4zrnhgoyibnz2.streamlit.app

---

## ℹ️ Remarque sur le mode démo

L’application Streamlit agit comme client de l'API et permet :
- de tester les recommandations sans user_id (cold start)
- de simuler différents profils utilisateurs
- d’évaluer le comportement des moteurs selon la stratégie choisie  

**Cette architecture permet une évolution indépendante des moteurs de recommandation, de l’API et de l’interface utilisateur.**

---

## 👩‍💻 Auteur

Projet réalisé par **Michèle Dewerpe**  
Dans le cadre du parcours *Ingénieur IA – OpenClassrooms*
