# -------------------- 完整代码（已合并修复） --------------------
import streamlit as st
import pymupdf
import pymysql
import numpy as np
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any
from dashscope import Generation, TextEmbedding


# ---------------- 页面配置 ----------------
st.set_page_config(
    page_title="TiDB + 千问 RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# ---------------- session state ----------------
for k in ['documents_count', 'last_query_time', 'db_initialized', 'rag_system']:
    if k not in st.session_state:
        st.session_state[k] = 0 if k == 'documents_count' else None if k == 'last_query_time' else False

# ---------------- RAG 系统 ----------------
class RAGSystem:
    def __init__(self):
        self.db_connection = None
        self.embedding_model = "text-embedding-v1"
        self.chat_model    = "qwen-max"
        self.chunk_size    = 1000
        self.chunk_overlap = 200
        self._inited       = False

    # ---------- 初始化 ----------
    def initialize(self, api_key: str,
                   host: str, port: int, user: str, password: str, database: str) -> bool:
        import dashscope
        dashscope.api_key = api_key.strip()

        # 千问探活
        try:
            Generation.call(model="qwen-turbo",
                            messages=[{"role": "user", "content": "ping"}],
                            max_tokens=1)
            self._inited = True
        except Exception as e:
            st.error(f"千问初始化失败：{e}")
            return False

        # 连接 TiDB
        try:
            self.db_connection = pymysql.connect(
                host=host, port=port, user=user, password=password,
                database=database, autocommit=True, ssl={"ssl": True}
            )
            self._init_db()
            return True
        except Exception as e:
            st.error(f"TiDB 连接失败：{e}")
            return False

    # ---------- 建表 ----------
    def _init_db(self):
        with self.db_connection.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS qwen_documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(1536) NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_filename (filename),
                INDEX idx_file_hash (file_hash),
                VECTOR INDEX idx_embedding ((VEC_COSINE_DISTANCE(embedding)))
            )
            """)

    # ---------- 文本提取 ----------
    def extract_pdf(self, file) -> str:
        doc = pymupdf.open(stream=file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text

    def extract_txt(self, file) -> str:
        file.seek(0)
        return file.read().decode("utf-8")

    # ---------- 分块 ----------
    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        step  = max(1, self.chunk_size - self.chunk_overlap)
        return [" ".join(words[i:i+self.chunk_size]) for i in range(0, len(words), step) if words[i:i+self.chunk_size]]

    # ---------- 向量化 ----------
    def embed(self, text: str) -> List[float]:
        resp = TextEmbedding.call(model=self.embedding_model, input=text)
        return [float(x) for x in resp.output["embeddings"][0]["embedding"]]

    # ---------- 存储 ----------
    def store(self, filename: str, chunks: List[str], file_hash: str):
        with self.db_connection.cursor() as cur:
            cur.execute("DELETE FROM qwen_documents WHERE filename=%s", (filename,))
            sql = "INSERT INTO qwen_documents (filename,chunk_index,content,embedding,file_hash) VALUES (%s,%s,%s,%s,%s)"
            for idx, chunk in enumerate(chunks):
                vec = self.embed(chunk)
                vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
                cur.execute(sql, (filename, idx, chunk, vec_str, file_hash))

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        vec = self.embed(query)
        vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        with self.db_connection.cursor() as cur:
            cur.execute("""
            SELECT id, filename, content,
                   VEC_COSINE_DISTANCE(embedding, %s) AS similarity
            FROM qwen_documents
            ORDER BY similarity ASC
            LIMIT %s
            """, (vec_str, top_k))
            return [{"id": r[0], "filename": r[1], "content": r[2], "similarity": float(r[3])} for r in cur.fetchall()]

    # ---------- 生成回答 ----------
    def answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(c["content"] for c in chunks)
        prompt = f"""Answer the question using only the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""
        resp = Generation.call(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
            result_format="message"
        )
        return resp.output.choices[0]["message"]["content"].strip()

    # ---------- 统计 ----------
    def doc_count(self) -> int:
        with self.db_connection.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT filename) FROM qwen_documents")
            return cur.fetchone()[0]

# ---------------- Streamlit 主界面 ----------------
def main():
    st.title("TiDB + 千问 RAG Document Q&A")
    st.markdown("上传文档，基于 **阿里通义千问** 与 **TiDB Vector** 进行问答。")

    # 初始化
    if not st.session_state.db_initialized:
        with st.spinner("🚀 正在初始化……"):
            sec = st.secrets
            rag = RAGSystem()
            ok = rag.initialize(sec["DASHSCOPE"]["DASHSCOPE_API_KEY"],
                                sec["TIDB"]["TIDB_HOST"],
                                int(sec["TIDB"]["TIDB_PORT"]),
                                sec["TIDB"]["TIDB_USER"],
                                sec["TIDB"]["TIDB_PASSWORD"],
                                sec["TIDB"]["TIDB_DATABASE"])
            if ok:
                st.session_state.rag_system = rag
                st.session_state.db_initialized = True
                st.session_state.documents_count = rag.doc_count()
                st.success("✅ 系统初始化完成！")
            else:
                st.error("❌ 系统初始化失败，请检查 secrets.toml")
                st.stop()

    rag = st.session_state.rag_system

    # 统计卡片
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📄 文档数", st.session_state.documents_count)
    with col2: st.metric("🔧 状态", "就绪")
    with col3: st.metric("⏰ 上次查询", st.session_state.last_query_time.strftime("%H:%M:%S") if st.session_state.last_query_time else "无")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    # 左侧：上传
    with col_left:
        st.header("上传文档")
        # ① 加 key，方便后面清掉
        file = st.file_uploader("选择 PDF / TXT 文件", type=["pdf", "txt"], key="uploader")
    
        if file:
            with st.spinner("处理中……"):
                text = rag.extract_pdf(file) if file.type == "application/pdf" else rag.extract_txt(file)
                if not text.strip():
                    st.error("❌ 提取文本失败")
                    st.stop()
    
                h = hashlib.sha256(text.encode()).hexdigest()
    
                with rag.db_connection.cursor() as cur:
                    cur.execute("SELECT 1 FROM qwen_documents WHERE file_hash=%s LIMIT 1", (h,))
                    if cur.fetchone():
                        st.warning("⚠️ 该文件已存在，跳过")
                    else:
                        chunks = rag.chunk_text(text)
                        rag.store(file.name, chunks, h)
                        st.success(f"✅ 已存储 {file.name}")
                        st.session_state.documents_count = rag.doc_count()
    
                    # ② 关键：立即清空上传控件
                    st.session_state.pop("uploader", None)


    # 右侧：提问
    with col_right:
        st.header("提问")
        q = st.text_area("输入问题：", height=100)
        if st.button("🔍 提问", type="primary", use_container_width=True):
            if q:
                with st.spinner("检索中……"):
                    chunks = rag.search(q, top_k=5)
                    if chunks:
                        with st.spinner("生成回答……"):
                            ans = rag.answer(q, chunks)
                            st.markdown("**💡 回答：**")
                            st.write(ans)

                            st.write("**📚 来源：**")
                            for i, c in enumerate(chunks):
                                color = "🟢" if c["similarity"] < 0.2 else "🟡"
                                with st.expander(f"{color} 来源 {i+1}: {c['filename']} (距离: {c['similarity']:.4f})"):
                                    st.write(c["content"][:500] + "…")
                            st.session_state.last_query_time = datetime.now()
                    else:
                        st.warning("⚠️ 未找到相关内容")
            else:
                st.error("❌ 请输入问题")

if __name__ == "__main__":
    main()
