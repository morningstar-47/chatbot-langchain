# Assistant Virtuel Polyvalent avec LangChain

Un assistant virtuel intelligent développé avec LangChain, FastAPI et OpenAI, capable de maintenir le contexte conversationnel et de fournir des réponses pertinentes grâce à une base de connaissances vectorielle.

## Fonctionnalités

- 💬 Chat conversationnel avec mémoire contextuelle
- 🧠 Récupération de contexte (RAG) depuis une base de connaissances
- 📚 Gestion de base de données vectorielle (ChromaDB)
- 🔄 Gestion de sessions utilisateur multiples
- 📄 Support pour l'ajout de documents à la base de connaissances
- 🌐 API REST complète avec documentation automatique

## Prérequis

- Python 3.9 ou supérieur
- Clé API OpenAI

## Installation

1. Clonez le dépôt :
```bash
git clone <repository-url>
cd chatbot-langchain
```

2. Créez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurez les variables d'environnement :
```bash
cp .env.example .env
# Éditez .env et ajoutez vos clés API :
# - OPENAI_API_KEY (requis)
# - RAPIDAPI_KEY (optionnel, pour la recherche d'emploi)
```

## Utilisation

1. Démarrez le serveur :
```bash
uvicorn app.main:app --reload
```

2. Accédez à la documentation interactive :
- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc

## Endpoints API

### Chat

- `POST /chat` - Envoyer un message et recevoir une réponse
- `POST /chat/session/{session_id}` - Messages pour une session spécifique
- `GET /chat/session/{session_id}/history` - Récupérer l'historique d'une session
- `DELETE /chat/session/{session_id}` - Réinitialiser une session

### Base de connaissances

- `POST /knowledge/upload` - Ajouter des documents à la base de connaissances

### Recherche d'emploi

- `GET /jobs/search` - Rechercher des emplois avec critères détaillés
- `GET /jobs/search/summary` - Rechercher des emplois avec résumé formaté
- `GET /jobs/{job_id}` - Récupérer les détails d'un emploi spécifique

## Exemples d'utilisation

### Chat

```python
import requests

# Envoyer un message
response = requests.post("http://localhost:8000/chat", json={
    "message": "Bonjour, pouvez-vous m'aider ?",
    "session_id": "user-123"
})

print(response.json())
```

### Recherche d'emploi

```python
import requests

# Rechercher des emplois
response = requests.get("http://localhost:8000/jobs/search", params={
    "query": "développeur Python",
    "location": "Paris, France"
})

print(response.json())

# Recherche avec résumé formaté
response = requests.get("http://localhost:8000/jobs/search/summary", params={
    "query": "data scientist",
    "limit": 5
})

print(response.json())
```

## Structure du projet

```
chatbot-langchain/
├── app/
│   ├── main.py              # Point d'entrée FastAPI
│   ├── config.py            # Configuration
│   ├── models/              # Modèles Pydantic
│   ├── services/            # Services métier
│   └── routers/             # Routes API
├── data/
│   └── knowledge_base/      # Documents de connaissances
└── requirements.txt
```

## Documentation Frontend

Pour intégrer l'API dans votre application frontend, consultez la **[documentation frontend complète](FRONTEND_API_DOCS.md)** qui inclut :

- Exemples JavaScript/TypeScript
- Exemples React et Vue.js
- Gestion des erreurs
- Exemple HTML fonctionnel (`examples/frontend_example.html`)

## Technologies utilisées

- **LangChain** : Framework pour applications LLM
- **FastAPI** : Framework web moderne et rapide
- **OpenAI** : Modèles de langage GPT
- **ChromaDB** : Base de données vectorielle
- **Pydantic** : Validation de données
- **RapidAPI JSearch** : API de recherche d'emploi

## Licence

MIT

