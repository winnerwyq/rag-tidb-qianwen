import streamlit as st
import google.generativeai as genai
import pymupdf
import hashlib
import numpy as np
from typing import List, Dict, Any
import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="TiDB + Gemini RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'documents_count' not in st.session_state:
    st.session_state.documents_count = 0
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = None
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

class RAGSystem:
    def __init__(self):
        self.db_connection = None
        # ✅ Gemini models
        self.embedding_model = "text-embedding-004"   # returns 768-d vectors
        self.chat_model = "gemini-2.0-flash"          # or "gemini-2.0-pro"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self._genai_inited = False

    def initialize_gemini(self, api_key: str):
        """Initialize Gemini client"""
        try:
            if not api_key or not api_key.strip():
                st.error("Gemini API key is empty")
                return False
            genai.configure(api_key=api_key)
            # quick sanity call: fetch model list (cheap)
            _ = genai.list_models()
            self._genai_inited = True
            return True
        except Exception as e:
            st.error(f"Gemini initialization failed: {str(e)}")
            return False

    def initialize_database(self, host: str, port: int, user: str, password: str, database: str):
        """Initialize TiDB connection and create table if needed"""
        try:
            self.db_connection = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                ssl_disabled=False,
                autocommit=True
            )
            cursor = self.db_connection.cursor()
            # ⚠️ VECTOR(768) matches text-embedding-004; change if you switch models
            create_table_query = """
            CREATE TABLE IF NOT EXISTS gemini_documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(768) NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_filename (filename),
                INDEX idx_file_hash (file_hash),
                VECTOR INDEX idx_embedding ((VEC_COSINE_DISTANCE(embedding)))
            )
            """
            cursor.execute(create_table_query)
            cursor.close()
            return True
        except Error as e:
            st.error(f"Database connection failed: {str(e)}")
            return False

    def get_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"PDF extraction failed: {str(e)}")
            return ""

    def extract_text_from_txt(self, txt_file) -> str:
        try:
            txt_file.seek(0)
            return txt_file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Text extraction failed: {str(e)}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        words = text.split()
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini embeddings"""
        try:
            if not self._genai_inited:
                st.error("Gemini not initialized")
                return []
            # Gemini embedding call
            resp = genai.embed_content(
                model=self.embedding_model,
                content=text,
            )
            vec = resp["embedding"]
            # Ensure Python list[float]
            return [float(x) for x in vec]
        except Exception as e:
            st.error(f"Embedding generation failed: {str(e)}")
            return []

    def check_document_exists(self, filename: str, file_hash: str) -> bool:
        try:
            cursor = self.db_connection.cursor()
            query = "SELECT COUNT(*) FROM gemini_documents WHERE filename = %s AND file_hash = %s"
            cursor.execute(query, (filename, file_hash))
            count = cursor.fetchone()[0]
            cursor.close()
            return count > 0
        except Error as e:
            st.error(f"Database query failed: {str(e)}")
            return False

    def store_document_chunks(self, filename: str, chunks: List[str], file_hash: str):
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("DELETE FROM gemini_documents WHERE filename = %s", (filename,))
            insert_query = """
            INSERT INTO gemini_documents (filename, chunk_index, content, embedding, file_hash)
            VALUES (%s, %s, %s, %s, %s)
            """
            for i, chunk in enumerate(chunks):
                embedding = self.generate_embedding(chunk)
                if embedding:
                    # TiDB VECTOR literal as JSON-ish string: [0.1,0.2,...]
                    embedding_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
                    cursor.execute(insert_query, (filename, i, chunk, embedding_str, file_hash))
            cursor.close()
            return True
        except Error as e:
            st.error(f"Database insertion failed: {str(e)}")
            return False

    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            embedding_str = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
            cursor = self.db_connection.cursor()
            sql = (
                "SELECT id, filename, content, "
                "VEC_COSINE_DISTANCE(embedding, %s) AS similarity "
                "FROM gemini_documents "
                "ORDER BY similarity ASC "
                "LIMIT %s"
            )
            cursor.execute(sql, (embedding_str, top_k))
            rows = cursor.fetchall()
            cursor.close()
            return [
                {"id": r[0], "filename": r[1], "content": r[2], "similarity": float(r[3])}
                for r in rows
            ]
        except Error as e:
            st.error(f"Search failed: {str(e)}")
            return []

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer with Gemini chat model (grounded in retrieved chunks)"""
        try:
            if not self._genai_inited:
                return "Gemini not initialized"
            context = "\n\n".join([c["content"] for c in context_chunks])
            prompt = f"""You are a helpful assistant. Answer strictly based on the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""
            model = genai.GenerativeModel(self.chat_model)
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=800
                )
            )
            return resp.text or "(No output)"
        except Exception as e:
            st.error(f"Answer generation failed: {str(e)}")
            return "Sorry, I couldn't generate an answer at this time."

    def get_document_count(self) -> int:
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM gemini_documents")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Error:
            return 0

def main():
    st.title("TiDB + Gemini RAG Document Q&A")
    st.markdown("Upload documents and ask questions to get AI-powered answers using **Google Gemini** with **TiDB Vector** features.")

    # Auto-initialize
    if not st.session_state.db_initialized:
        with st.spinner("🚀 Initializing system..."):
            try:
                gemini_api_key = st.secrets["GOOGLE_API_KEY"]
                db_host = st.secrets["TIDB_HOST"]
                db_port = int(st.secrets["TIDB_PORT"])
                db_user = st.secrets["TIDB_USER"]
                db_password = st.secrets["TIDB_PASSWORD"]
                db_name = st.secrets["TIDB_DATABASE"]

                rag = RAGSystem()

                if rag.initialize_gemini(gemini_api_key):
                    if rag.initialize_database(db_host, db_port, db_user, db_password, db_name):
                        st.session_state.db_initialized = True
                        st.session_state.rag_system = rag
                        st.session_state.documents_count = rag.get_document_count()
                        st.success("✅ System initialized successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Database connection failed")
                        return
                else:
                    st.error("❌ Gemini initialization failed")
                    return

            except KeyError as e:
                st.error(f"❌ Missing secret: {e}")
                st.info("Please ensure all required secrets are configured in your Streamlit app:")
                st.code("""
GOOGLE_API_KEY = "your-gemini-api-key"
TIDB_HOST = "your-tidb-host"
TIDB_PORT = "4000"
TIDB_USER = "your-username"
TIDB_PASSWORD = "your-password"
TIDB_DATABASE = "your-database-name"
                """)
                return
            except Exception as e:
                st.error(f"❌ System initialization failed: {str(e)}")
                return

    # Main interface
    rag = st.session_state.rag_system

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", st.session_state.documents_count)
    with col2:
        st.metric("🔧 Status", "Ready" if st.session_state.db_initialized else "Not Ready")
    with col3:
        st.metric("⏰ Last Query",
                  st.session_state.last_query_time.strftime("%H:%M:%S") if st.session_state.last_query_time else "Never")

    with st.expander("🔧 System Controls", expanded=False):
        if st.button("🔄 Reset System"):
            st.session_state.db_initialized = False
            st.session_state.rag_system = None
            st.session_state.documents_count = 0
            st.session_state.last_query_time = None
            st.rerun()

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'],
                                         help="Upload PDF or text files to add to the knowledge base")

        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                if uploaded_file.type == "application/pdf":
                    text = rag.extract_text_from_pdf(uploaded_file)
                else:
                    text = rag.extract_text_from_txt(uploaded_file)

                if text:
                    file_hash = rag.get_text_hash(text)
                    if rag.check_document_exists(uploaded_file.name, file_hash):
                        st.warning("⚠️ Document already exists in the database!")
                    else:
                        chunks = rag.chunk_text(text)
                        if rag.store_document_chunks(uploaded_file.name, chunks, file_hash):
                            st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                            st.info(f"📊 Created {len(chunks)} chunks")
                            st.session_state.documents_count = rag.get_document_count()
                        else:
                            st.error("❌ Failed to store document")
                else:
                    st.error("❌ Failed to extract text from document")

    with col2:
        st.header("Ask a Question")
        query = st.text_area("Enter your question:",
                             placeholder="What would you like to know about your documents?",
                             height=100)

        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if query:
                with st.spinner("🔍 Searching for relevant information..."):
                    similar_chunks = rag.search_similar_chunks(query, top_k=5)
                    if similar_chunks:
                        with st.spinner("🤖 Generating answer..."):
                            answer = rag.generate_answer(query, similar_chunks)
                            st.subheader("💡 Answer:")
                            st.markdown(answer)

                            st.subheader("📚 Sources:")
                            for i, chunk in enumerate(similar_chunks):
                                # smaller distance = more similar
                                dist = chunk['similarity']
                                color = "🟢" if dist < 0.2 else "🟡" if dist < 0.4 else "🔴"
                                with st.expander(f"{color} Source {i+1}: {chunk['filename']} (Distance: {dist:.4f})"):
                                    preview = chunk['content']
                                    st.write(preview[:500] + "..." if len(preview) > 500 else preview)

                            st.session_state.last_query_time = datetime.now()
                    else:
                        st.warning("⚠️ No relevant information found in the uploaded documents.")
            else:
                st.error("❌ Please enter a question.")

if __name__ == "__main__":
    main()
