
-- Table for chunks + embeddings (Gemini)
CREATE TABLE IF NOT EXISTS gemini_documents (
  id INT AUTO_INCREMENT PRIMARY KEY,
  filename VARCHAR(255) NOT NULL,
  chunk_index INT NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(768) NOT NULL,       -- 768 dims for text-embedding-004
  file_hash VARCHAR(64) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_filename (filename),
  INDEX idx_file_hash (file_hash),
  -- Vector index for cosine similarity
  VECTOR INDEX idx_embedding ((VEC_COSINE_DISTANCE(embedding)))
);
