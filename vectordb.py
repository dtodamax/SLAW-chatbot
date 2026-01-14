import json
import os
import sys
import time
from typing import List, Dict, Any
from tqdm import tqdm
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data_with_vectors.json"
TABLE_NAME = "law_chunks"
MODEL_NAME = "intfloat/multilingual-e5-large"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Phần 1: DATABASE IMPORTER (UPLOAD DỮ LIỆU)

class DatabaseImporter:
    def __init__(self, url: str, key: str):
        if not url or not key:
            raise ValueError("[ERROR] Thiếu Supabase URL hoặc Key.")
        self.client: Client = create_client(url, key)
        print(f"[CONNECTION] Đã kết nối tới Supabase: {url[:20]}...")

    def validate_input(self, data: List[Dict]) -> bool:
        """Kiểm tra xem dữ liệu có khớp với Schema mới không"""
        if not data: return False
        sample = data[0]
        meta = sample.get('metadata', {})

        # Kiểm tra các trường mới
        required_fields = ['full_context', 'id_path']
        missing = [f for f in required_fields if f not in meta]

        if missing:
            print(f"[WARNING] Dữ liệu thiếu các trường metadata mới: {missing}")
            return False
        return True

    def run_import(self, file_path: str, batch_size: int = 50):
        print(f"\n=== BẮT ĐẦU UPLOAD DỮ LIỆU TỪ {file_path} ===")

        if not os.path.exists(file_path):
            print(f"[ERROR] Không tìm thấy file: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not self.validate_input(data):
            user_input = input("Dữ liệu có vẻ không khớp chuẩn mới. Vẫn tiếp tục? (y/n): ")
            if user_input.lower() != 'y': return

        # 1. Xóa dữ liệu cũ (Optional - Vì ta đã chạy SQL drop table rồi, nhưng cứ để cho chắc)
        print(f"[ACTION] Đang làm sạch bảng '{TABLE_NAME}'...")
        try:
            self.client.table(TABLE_NAME).delete().neq("id", 0).execute()
        except Exception as e:
            print(f"[INFO] Bảng có thể đang trống hoặc vừa tạo mới. ({e})")

        # 2. Upload Batch
        total = len(data)
        print(f"[ACTION] Đang upload {total} records...")

        success_count = 0
        with tqdm(total=total, unit="chunk") as pbar:
            for i in range(0, total, batch_size):
                batch = data[i : i + batch_size]
                payload = []

                for item in batch:
                    # Chuẩn hóa payload cho Supabase
                    payload.append({
                        "content": item["content"],
                        "metadata": item["metadata"], # JSONB sẽ tự xử lý dict này
                        "embedding": item["embedding"]
                    })

                try:
                    self.client.table(TABLE_NAME).insert(payload).execute()
                    success_count += len(batch)
                    pbar.update(len(batch))
                except Exception as e:
                    print(f"\n[ERROR] Lỗi tại batch {i}: {e}")
                    # Retry logic đơn giản (nếu cần)
                    time.sleep(1)

        print(f"\n[SUCCESS] Đã upload thành công {success_count}/{total} dòng.")

# PHẦN 2: HYBRID SEARCH ENGINE (TÌM KIẾM NÂNG CAO)

class VectorSearchEngine:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)
        print(f"[MODEL] Đang tải model {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)

    def query_raw(self, query_text: str, top_k: int = 30) -> List[Dict]:
        """
        Tìm kiếm LAI (Hybrid Search): Kết hợp Vector + Keyword
        """
        # 1. Tạo Vector (E5 bắt buộc prefix 'query: ')
        query_vec = self.model.encode(
            f"query: {query_text}",
            normalize_embeddings=True,
            convert_to_numpy=True
        ).tolist()

        # 2. Chuẩn bị tham số cho hàm 'hybrid_search' trong SQL
        # Lưu ý: Tên tham số (key) phải khớp 100% với tên tham số trong hàm SQL
        params = {
            "query_text": query_text,       # Dùng để tìm từ khóa (FTS)
            "query_embedding": query_vec,   # Dùng để tìm ngữ nghĩa
            "match_count": top_k,           # Số lượng lấy về (ví dụ 30)
            "full_text_weight": 1.0,        # (Tùy chọn) Trọng số từ khóa
            "semantic_weight": 1.0          # (Tùy chọn) Trọng số vector
        }

        try:
            # 3. Gọi hàm RPC 'hybrid_search' thay vì 'match_law_chunks'
            response = self.client.rpc("hybrid_search", params).execute()
            return response.data
        except Exception as e:
            print(f"[ERROR] Lỗi RPC Supabase: {e}")
            # Fallback: Nếu hàm hybrid lỗi (do chưa setup SQL), thử quay về vector thường
            print("[INFO] Đang thử fallback về Vector Search thường...")
            try:
                fallback_params = {
                    "query_embedding": query_vec,
                    "match_threshold": 0.0,
                    "match_count": top_k
                }
                return self.client.rpc("match_law_chunks", fallback_params).execute().data
            except:
                return []

# Phần 3: CONTEXT AWARE RETRIEVER (LOGIC SMALL-TO-BIG)

class ContextAwareRetriever:
    def __init__(self, engine: VectorSearchEngine):
        self.engine = engine

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        print(f"\n[SEARCH] Query: '{query}'")

        # 1. Lấy nhiều kết quả thô hơn cần thiết (Recall) để lọc trùng
        raw_results = self.engine.query_raw(query, top_k=top_k * 3)

        if not raw_results:
            print("[INFO] Không tìm thấy kết quả.")
            return []

        # 2. Xử lý Small-to-Big & Deduplication
        unique_contexts = {}
        final_results = []

        print(f"[PROCESS] Tìm thấy {len(raw_results)} chunks thô. Đang mở rộng ngữ cảnh...")

        for item in raw_results:
            meta = item.get('metadata', {})

            # Dùng id_path hoặc tên Điều luật để định danh duy nhất
            # Ưu tiên dùng id_path (ví dụ: ChI|D5) nếu có, vì nó chính xác hơn
            doc_id = meta.get('article') # Hoặc meta.get('id_path')

            full_context = meta.get('full_context')
            if not full_context: continue # Bỏ qua nếu dữ liệu lỗi

            # Logic: Chỉ lấy bản ghi có điểm cao nhất cho mỗi Điều luật
            if doc_id not in unique_contexts:
                unique_contexts[doc_id] = item['similarity']

                final_results.append({
                    "score": item['similarity'],
                    "source": doc_id,
                    "id_path": meta.get('id_path', 'N/A'),
                    "small_chunk": item['content'],
                    "full_context": full_context
                })

        # Sắp xếp và cắt Top K
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]

# PHẦN 4: MAIN EXECUTION

if __name__ == "__main__":
    # Lấy credentials từ Colab Userdata hoặc Environment
    try:
        from google.colab import userdata
        url = userdata.get('SUPABASE_URL')
        key = userdata.get('SUPABASE_KEY')
    except ImportError:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print(" Vui lòng cấu hình SUPABASE_URL và SUPABASE_KEY.")
        sys.exit(1)

    # --- MODE 1: UPLOAD (Chạy 1 lần khi có dữ liệu mới) ---
    # Uncomment dòng dưới để chạy upload
    importer = DatabaseImporter(url, key)
    importer.run_import(INPUT_FILE)
