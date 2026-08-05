import json
import os
import torch
import gc
import sys
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CẤU HÌNH ---
# Input là file output của bước Chunking Hybrid trước đó
INPUT_FILE = "final_rag_data.json"
OUTPUT_FILE = "data_with_vectors.json"
MODEL_NAME = "intfloat/multilingual-e5-large"

class EmbeddingPipeline:
    def __init__(self, input_path: str, output_path: str, model_name: str):
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.device = self._get_device()

    def _get_device(self) -> str:
        if torch.cuda.is_available(): return "cuda"
        elif torch.backends.mps.is_available(): return "mps" # Cho Mac M1/M2
        return "cpu"

    def load_data(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.input_path):
            print(f"[ERROR] Không tìm thấy file input: {self.input_path}")
            sys.exit(1)
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] File JSON lỗi: {e}")
            sys.exit(1)

    def _validate_data_readiness(self, data: List[Dict]):
        """Kiểm tra xem dữ liệu có đủ chuẩn Small-to-Big không"""
        print("1. [Check] Kiểm tra cấu trúc dữ liệu...")
        if not data:
            sys.exit("[FAIL] File dữ liệu rỗng!")

        sample = data[0]
        # Kiểm tra trường quan trọng nhất: full_context
        if 'full_context' not in sample.get('metadata', {}):
            print("[FAIL] Thiếu trường 'full_context' trong metadata.")
            print("Hãy chạy lại bước Chunking Hybrid để đảm bảo có ngữ cảnh mở rộng.")
            sys.exit(1)

        print(f"[PASS] Dữ liệu hợp lệ. Tổng số chunk: {len(data)}")

    def run(self):
        print(f"\n=== BẮT ĐẦU EMBEDDING VỚI {self.device.upper()} ===")

        # 1. Load & Validate
        data = self.load_data()
        self._validate_data_readiness(data)

        # 2. Load Model
        print(f"2. [Model] Đang tải {self.model_name}...")
        model = SentenceTransformer(self.model_name, device=self.device)

        # 3. Chuẩn bị text (Thêm prefix 'passage: ' cho E5)
        # Chỉ embed phần 'content' (Small Chunk), không embed 'full_context'
        texts_to_embed = [f"passage: {item['content']}" for item in data]

        # 4. Encoding (Batch processing)
        batch_size = 32 if self.device == "cuda" else 4
        print(f"3. [Encoding] Đang tạo vector cho {len(texts_to_embed)} đoạn văn...")

        embeddings = model.encode(
            texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True # Quan trọng cho Cosine Similarity
        )

        # 5. Gán Vector vào Data
        print("4. [Saving] Đang lưu kết quả...")
        for i, item in enumerate(data):
            item['embedding'] = embeddings[i].tolist()

        # 6. Lưu file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] Hoàn tất! File lưu tại: {self.output_path}")

        # Dọn dẹp bộ nhớ
        del model, embeddings
        if self.device == "cuda": torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    pipeline = EmbeddingPipeline(INPUT_FILE, OUTPUT_FILE, MODEL_NAME)
    pipeline.run()



# Kiểm tra số lượng vector để upload lên Supabase
import json
import os
import numpy as np

INPUT_FILE = "data_with_vectors.json"
EXPECTED_DIMENSION = 1024 # E5-large luôn là 1024

def validate_vectors():
    print(f"\n=== KIỂM TRA CHẤT LƯỢNG VECTOR ===")

    if not os.path.exists(INPUT_FILE):
        print("[ERROR] Không tìm thấy file data_with_vectors.json")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    error_count = 0
    warning_count = 0

    print(f"Đang kiểm tra {len(data)} bản ghi...")

    for i, item in enumerate(data):
        vec = item.get('embedding')
        meta = item.get('metadata', {})

        # 1. Kiểm tra Vector tồn tại
        if not vec:
            print(f"[ERROR] Chunk #{i}: Thiếu vector.")
            error_count += 1; continue

        # 2. Kiểm tra số chiều (Dimension)
        if len(vec) != EXPECTED_DIMENSION:
            print(f"[ERROR] Chunk #{i}: Sai số chiều ({len(vec)} != {EXPECTED_DIMENSION}).")
            error_count += 1; continue

        # 3. Kiểm tra giá trị rác (NaN/Inf)
        if not np.isfinite(vec).all():
            print(f"[ERROR] Chunk #{i}: Vector chứa giá trị lỗi (NaN/Inf).")
            error_count += 1; continue

        # 4. Kiểm tra Small-to-Big (Full Context)
        if not meta.get('full_context'):
            print(f"[ERROR] Chunk #{i}: Mất dữ liệu 'full_context'.")
            error_count += 1

    if error_count == 0:
        print(f"\n [PASS] Dữ liệu hoàn hảo! Sẵn sàng upload lên Supabase.")
    else:
        print(f"\n [FAIL] Phát hiện {error_count} lỗi nghiêm trọng.")

if __name__ == "__main__":
    validate_vectors()

# Testing thử chunking từ Vector
import json
import os

INPUT_FILE = "data_with_vectors.json"

class Color:
    BLUE = '\033[94m'; GREEN = '\033[92m'; ENDC = '\033[0m'; BOLD = '\033[1m'

def inspect_record(article_number: str):
    """Tìm và hiển thị chi tiết chunk của một Điều luật cụ thể"""
    if not os.path.exists(INPUT_FILE): return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Tìm chunk thuộc Điều X
    target = f"Điều {article_number}"
    found = [d for d in data if target in d['metadata'].get('article', '')]

    if not found:
        print(f"Không tìm thấy dữ liệu cho {target}")
        return

    # Lấy mẫu đầu tiên tìm được
    sample = found[0]
    meta = sample['metadata']

    print(f"\n{Color.BOLD}KIỂM CHỨNG DỮ LIỆU: {meta['article']}{Color.ENDC}")
    print(f"ID Path: {meta.get('id_path', 'N/A')}")

    print(f"\n{Color.BLUE}[1. SMALL CHUNK]):{Color.ENDC}")
    print(f"Content: {sample['content'][:200]}...")
    print(f"(Độ dài: {len(sample['content'])} ký tự)")

    print(f"\n{Color.GREEN}[2. BIG CONTEXT]:{Color.ENDC}")
    full_ctx = meta.get('full_context', '')
    print(f"Content: {full_ctx[:200]}...")
    print(f"(Độ dài: {len(full_ctx)} ký tự)")

    print(f"\n{Color.BOLD}[3. VECTOR]{Color.ENDC}")
    print(f"Dimension: {len(sample['embedding'])}")
    print(f"Values: {sample['embedding'][:3]}... (đã ẩn 1021 số còn lại)")

if __name__ == "__main__":
    # Bạn có thể thay đổi số Điều muốn kiểm tra (ví dụ: "15", "9", "4")
    inspect_record("11")
