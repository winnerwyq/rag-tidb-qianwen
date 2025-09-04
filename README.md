Q&A PDF Chatbot (Streamlit + ali qianwen + TiDB)
This app lets users upload a PDF, generate text embeddings using ali qianwen, store those embeddings in TiDB Cloud Serverless, and chat with the document via a simple Streamlit UI.

Features
Upload and parse PDF files

Chunk and embed documents with qianwen Embeddings (e.g., text-embedding-004)

Store embeddings in TiDB Cloud Serverless

Retrieve similar chunks via vector search (cosine/vector distance)

Answer user queries with qianwen (e.g., qianwen-1.5-flash / qianwen-1.5-pro)

Deployment
Clone this repo.

Deploy on Streamlit Cloud: https://streamlit.io/cloud

Click New app → select your GitHub repo.

Choose main branch and app.py as the main file.

In Streamlit Cloud, open your app → Settings → Secrets and add:

# --- TiDB Cloud (Serverless) ---
TIDB_HOST = "your_host"
TIDB_USER = "your_user"
TIDB_PASSWORD = "your_password"
TIDB_DATABASE = "your_database"
TIDB_PORT = "your_database_port"

# --- ali qianwen (AI Studio) ---
qianwen_API_KEY = "your_qianwen_api_key"


