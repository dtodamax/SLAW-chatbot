import os
import sys
import time
import torch
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

# --- CẤU HÌNH HỆ THỐNG ---
class Config:
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # Tên hàm RPC mới (Hybrid Search)
    RPC_FUNCTION = "hybrid_search"

    # Models
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

    # Parameters
    # Hybrid Search: Lấy 30 Vector + 30 Keyword = ~50 ứng viên
    TOP_K_RETRIEVAL = 50  
    TOP_K_FINAL = 5       
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Xử lý Colab Secrets
try:
    from google.colab import userdata
    if not Config.SUPABASE_URL: Config.SUPABASE_URL = userdata.get('SUPABASE_URL')
    if not Config.SUPABASE_KEY: Config.SUPABASE_KEY = userdata.get('SUPABASE_KEY')
except ImportError:
    pass

# --- TIỆN ÍCH LOGGING & TIMING ---
class Logger:
    HEADER = '\033[95m'; BLUE = '\033[94m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; ENDC = '\033[0m'

    @staticmethod
    def info(msg): print(f"{Logger.BLUE}[INFO]{Logger.ENDC} {msg}")
    @staticmethod
    def success(msg): print(f"{Logger.GREEN}[SUCCESS]{Logger.ENDC} {msg}")
    @staticmethod
    def step(msg): print(f"\n{Logger.HEADER}=== {msg} ==={Logger.ENDC}")
    @staticmethod
    def metric(name, value): print(f"{Logger.YELLOW}   ↳ {name}: {value}{Logger.ENDC}")

class Timer:
    def __init__(self):
        self.start = time.time()
    def stop(self):
        return (time.time() - self.start) * 1000 # ms

# Phần 1: QUẢN LÝ MODEL (MODEL MANAGERS)

class ModelManager:
    def __init__(self):
        Logger.info(f"Khởi tạo trên thiết bị: {Config.DEVICE.upper()}")

        # 1. Load Embedding Model (E5)
        Logger.info(f"Đang tải Embedding Model: {Config.EMBEDDING_MODEL}...")
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)

        # 2. Load Reranker Model (BGE)
        Logger.info(f"Đang tải Reranker Model: {Config.RERANKER_MODEL}...")
        # use_fp16=True giúp tăng tốc độ trên GPU
        self.reranker = FlagReranker(Config.RERANKER_MODEL, use_fp16=(Config.DEVICE == "cuda"))

        # 3. Connect Supabase
        if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
            raise ValueError("Thiếu Supabase Credentials!")
        self.db_client: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

        Logger.success("Hệ thống đã sẵn sàng!")

    def embed_query(self, text: str) -> List[float]:
        # E5 bắt buộc prefix 'query: '
        return self.embedder.encode(f"query: {text}", normalize_embeddings=True).tolist()

    def rerank_pairs(self, pairs: List[List[str]]) -> List[float]:
        if not pairs: return []
        scores = self.reranker.compute_score(pairs)
        return scores if isinstance(scores, list) else [scores]

# Phần 2: PIPELINE XỬ LÝ CHÍNH (CORE LOGIC)

@dataclass
class SearchResult:
    source_id: str      # Định danh (VD: Điều 15)
    score: float        # Điểm Rerank
    small_chunk: str    # Đoạn máy tìm thấy
    full_context: str   # Đoạn người đọc
    id_path: str        # Đường dẫn cây luật

class AdvancedRetriever:
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager

    def run(self, query: str) -> List[SearchResult]:
        total_timer = Timer()
        Logger.step(f"BẮT ĐẦU TRUY VẤN (HYBRID): '{query}'")

        # --- BƯỚC 1: HYBRID RETRIEVAL (Vector + Keyword) ---
        # Mục tiêu: RECALL (Lấy rộng bằng cả ngữ nghĩa và từ khóa)
        t1 = Timer()
        query_vec = self.mm.embed_query(query)

        # Tham số cho hàm hybrid_search trong SQL
        rpc_params = {
            "query_text": query,          # Dùng cho Keyword Search (bắt chữ chính xác)
            "query_embedding": query_vec, # Dùng cho Vector Search (bắt ngữ nghĩa)
            "match_count": Config.TOP_K_RETRIEVAL # Lấy 30 mỗi loại -> Gộp lại
        }

        try:
            response = self.mm.db_client.rpc(Config.RPC_FUNCTION, rpc_params).execute()
            candidates = response.data
        except Exception as e:
            Logger.info(f"Lỗi Supabase: {e}")
            return []

        Logger.metric("Stage 1 Latency", f"{t1.stop():.2f} ms")
        Logger.metric("Candidates Found", len(candidates)) 
        
        if not candidates: return []

        # --- BƯỚC 2: RERANKING (BGE-M3) ---
        # Mục tiêu: PRECISION (Lọc tinh, chọn ra cái đúng nhất)
        t2 = Timer()

        # Tạo cặp [Query, Small Chunk Content] để chấm điểm
        pairs = [[query, item['content']] for item in candidates]
        scores = self.mm.rerank_pairs(pairs)

        # Gán điểm và sort
        for i, item in enumerate(candidates):
            item['rerank_score'] = scores[i]

        # Sắp xếp giảm dần theo điểm Rerank
        ranked_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        Logger.metric("Stage 2 Latency", f"{t2.stop():.2f} ms")
        # In ra điểm của Top 1 để kiểm tra độ tự tin của model
        Logger.metric("Top 1 Score", f"{ranked_candidates[0]['rerank_score']:.4f}")

        # --- BƯỚC 3: EXPANSION & DEDUPLICATION (Small-to-Big) ---
        t3 = Timer()
        final_results = []
        seen_parent_ids: Set[str] = set()

        for item in ranked_candidates:
            # Chỉ lấy Top 5 kết quả cuối cùng
            if len(final_results) >= Config.TOP_K_FINAL:
                break

            meta = item.get('metadata', {})
            parent_id = meta.get('article')

            # Khử trùng lặp
            if parent_id in seen_parent_ids:
                continue

            seen_parent_ids.add(parent_id)

            final_results.append(SearchResult(
                source_id=parent_id,
                score=item['rerank_score'],
                small_chunk=item['content'],
                full_context=meta.get('full_context', item['content']),
                id_path=meta.get('id_path', 'N/A')
            ))

        Logger.metric("Stage 3 Latency", f"{t3.stop():.2f} ms")
        Logger.metric("Final Unique Docs", len(final_results))
        Logger.metric("Total Latency", f"{total_timer.stop():.2f} ms")

        return final_results


# Phần 3: CHẠY THỰC NGHIỆM & HIỂN THỊ (EXECUTION)

def print_report(results: List[SearchResult]):
    print("\n" + "="*80)
    print(f"{'RANK':<5} | {'SCORE':<8} | {'SOURCE (ĐIỀU LUẬT)':<30} | {'ID PATH':<20}")
    print("-" * 80)

    for i, res in enumerate(results):
        print(f"#{i+1:<4} | {res.score:.4f}   | {res.source_id[:28]:<30} | {res.id_path}")
    print("="*80 + "\n")

    # Hiển thị chi tiết Top 1 để kiểm tra ngữ cảnh
    if results:
        top1 = results[0]
        print(f"{Logger.GREEN}>>> CHI TIẾT NGỮ CẢNH TOP 1 (Gửi cho LLM):{Logger.ENDC}")
        print(f"Nguồn: {top1.source_id}")
        print("-" * 40)
        print(top1.full_context) # In toàn bộ ngữ cảnh
        print("-" * 40)

if __name__ == "__main__":
    # 1. Init
    try:
        manager = ModelManager()
        retriever = AdvancedRetriever(manager)

        # 2. Test Query
        # Câu hỏi này cần tổng hợp thông tin từ Điều kiện chào bán
        query = "Điều kiện về vốn khi chào bán cổ phiếu ra công chúng là bao nhiêu?"

        # 3. Run Pipeline
        results = retriever.run(query)

        # 4. Report
        print_report(results)

    except Exception as e:
        print(f"\n LỖI CHƯƠNG TRÌNH: {e}")

# TEST 
queries = [
    "Hồ sơ đăng ký chào bán cổ phiếu ra công chúng gồm những giấy tờ gì?",
    "Vốn điều lệ tối thiểu để trở thành công ty đại chúng là bao nhiêu?",
    "Hành vi thao túng thị trường chứng khoán bị cấm như thế nào?",
    "Điều kiện chào bán cổ phiếu riêng lẻ của công ty đại chúng?",
]

for q in queries:
    results = retriever.run(q)
    print_report(results)



test_query = "Nhà đầu tư chứng khoán chuyên nghiệp là ai?"
results = retriever.run(test_query)
print_report(results)


queries = [
    "Ủy ban Chứng khoán Nhà nước có những nhiệm vụ và quyền hạn gì trong quản lý nhà nước về chứng khoán và thị trường chứng khoán"
]

for q in queries:
    results = retriever.run(q)
    print_report(results)

