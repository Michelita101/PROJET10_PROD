# 📰 Projet 10 – Application de recommandation de contenu

Ce projet consiste à concevoir et déployer une application complète de recommandation d’articles, intégrant plusieurs moteurs de recommandation et une interface de démonstration interactive.

## 🎯 Objectifs
- Construire plusieurs moteurs de recommandation :
  - Content-Based Filtering
  - Collaborative Filtering (item-based)
  - ALS (Collaborative Filtering matriciel)
  - Moteur hybride
- Exposer l’inférence via une API (Azure Functions)
- Proposer une interface utilisateur pour tester les recommandations
- Déployer l’application en ligne

## 🧠 Architecture globale

- **API d’inférence** : Azure Functions  
- **Stockage des artefacts** : Azure Blob Storage  
- **Interface utilisateur** : Streamlit  
- **Déploiement front** : Streamlit Cloud  

Utilisateur  
⬇️  
Streamlit App  
⬇️  
Azure Function API  
⬇️  
Moteurs de recommandation  
⬇️  
Azure Blob Storage

## ⚙️ Fonctionnalités principales

- Sélection utilisateur via :
  - liste d’utilisateurs embarquée (démo)
  - saisie manuelle d’un `user_id`
  - upload d’un fichier (< 200 MB)
- Choix de la stratégie :
  - auto (routing MVP basé sur l’historique utilisateur)
  - content_based
  - cf_item
  - cf_global
  - hybrid
  - als
- Paramétrage du nombre de recommandations (`top_k`)
- Affichage des scores et de la réponse brute

## 📂 Structure du projet

```
PROJET10_PROD/
├── app.py                  # Application Streamlit
├── function_app.py         # Azure Functions (API)
├── data/
│   └── streamlit_users_demo.parquet
├── requirements.txt
├── README.md

```

## 🚀 Lancer le projet en local

### 1. API Azure Functions
```bash
func start
```

### 2. Application Streamlit

```
streamlit run app.py
```

## 🌐 Déploiement

- L’API est déployée via **Azure Functions**
- L’interface est déployée via **Streamlit Cloud**
- Les artefacts lourds (modèles, matrices, embeddings) sont stockés dans **Azure Blob Storage**

## 👩‍💻 Auteur

Projet réalisé par **Michèle Dewerpe**
Dans le cadre du parcours *Ingénieur IA – OpenClassrooms*
