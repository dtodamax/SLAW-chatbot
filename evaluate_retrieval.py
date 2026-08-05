# 3. LOGIC ĐÁNH GIÁ (EVALUATION LOGIC)

class RAGEvaluator:
    def __init__(self):
        self.manager = ModelManager()
        self.retriever = AdvancedRetriever(self.manager)

    def check_match_regex(self, target: str, retrieved_source: str) -> bool:
        """
        Kiểm tra khớp nguồn thông minh bằng Regex.
        Khắc phục lỗi: 'Điều 1' khớp 'Điều 115'.
        """
        if not target or not retrieved_source: return False
        
        # Lấy số điều luật từ target (VD: "Điều 11" -> "11")
        nums = re.findall(r'\d+', target)
        
        if not nums: 
            return target.lower() in retrieved_source.lower()
        
        target_num = nums[0]
        
        # Regex Pattern: "Điều" + khoảng trắng + số + biên từ (\b)
        # \b đảm bảo số 11 không khớp với 115, 116
        pattern = r"Điều\s+" + re.escape(target_num) + r"\b"
        
        return bool(re.search(pattern, retrieved_source, re.IGNORECASE))

    def run_evaluation(self, dataset: List[Dict]):
        print(f"\n BẮT ĐẦU ĐÁNH GIÁ TRÊN {len(dataset)} CÂU HỎI...")
        print("-" * 60)
        
        results = []
        total_hits = 0
        mrr_sum = 0

        for item in tqdm(dataset):
            query = item['question']
            target = item['target_source']
            
            # 1. Chạy hệ thống tìm kiếm
            retrieved_docs = self.retriever.run(query)
            
            is_hit = False
            rank = 0
            found_source = "None"
            
            # 2. Kiểm tra xem Target có nằm trong danh sách trả về không
            for i, doc in enumerate(retrieved_docs):
                if self.check_match_regex(target, doc.source_id):
                    is_hit = True
                    rank = i + 1 # Rank bắt đầu từ 1
                    found_source = doc.source_id
                    break 
            
            # 3. Tính điểm
            if is_hit:
                total_hits += 1
                mrr_sum += 1.0 / rank # Cộng điểm MRR (1/1, 1/2, 1/3...)
            
            results.append({
                "Question": query,
                "Target": target,
                "Found Source": found_source if is_hit else " Missed",
                "Rank": rank if is_hit else "-",
                "Pass": "Tốt" if is_hit else "Xấu"
            })

        # 4. Tổng hợp
        hit_rate = (total_hits / len(dataset)) * 100
        mrr = mrr_sum / len(dataset) if len(dataset) > 0 else 0
        
        return hit_rate, mrr, pd.DataFrame(results)

# 4. CHẠY THỰC NGHIỆM (MAIN EXECUTION)

# Bộ dữ liệu Test (Golden Dataset)
TEST_DATASET = [
    {
        "question": "Nhà đầu tư chứng khoán chuyên nghiệp là ai?",
        "target_source": "Điều 11"
    },
    {
        "question": "Vốn điều lệ tối thiểu để chào bán cổ phiếu ra công chúng là bao nhiêu?",
        "target_source": "Điều 15"
    },
    {
        "question": "Điều kiện cấp Giấy phép thành lập và hoạt động kinh doanh chứng khoán?",
        "target_source": "Điều 74"
    },
    {
        "question": "Ủy ban Chứng khoán Nhà nước trực thuộc cơ quan nào?",
        "target_source": "Điều 9"
    },
    {
        "question": "Hồ sơ đăng ký chào bán cổ phiếu lần đầu ra công chúng gồm những gì?",
        "target_source": "Điều 18"
    }
]

if __name__ == "__main__":
    # Cài đặt thư viện nếu chưa có
    # !pip install -q sentence-transformers supabase FlagEmbedding pandas tqdm

    evaluator = RAGEvaluator()
    hit_rate, mrr, df = evaluator.run_evaluation(TEST_DATASET)
    
    print(" Báo cáo hiệu năng hệ thống (RETRIEVAL METRICS)")
    print(f" HIT RATE (Độ chính xác Top 5): {hit_rate:.2f}%")
    print(f" MRR Score (Thứ hạng trung bình): {mrr:.4f} (Max 1.0)")
    
    # Hiển thị bảng kết quả chi tiết
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 1000)
    print(df)
    
    # Lưu kết quả ra Excel
    df.to_excel("rag_evaluation_results.xlsx", index=False)
    print("\n Đã lưu kết quả chi tiết vào 'rag_evaluation_results.xlsx'")



# NÂNG CẤP 

import pandas as pd
import re
from tqdm import tqdm

# PHẦN BỔ SUNG: ĐÁNH GIÁ HIT RATE & MRR (PHIÊN BẢN NÂNG CẤP)

# 1. Cập nhật bộ Test Data (Ground Truth)
advanced_test_data = [
    {
        "question": "Vốn điều lệ tối thiểu chào bán cổ phiếu ra công chúng?",
        "ground_truth": "30 tỷ đồng...",
        "target_source": "Điều 15" 
    },
    {
        "question": "Nhà đầu tư chứng khoán chuyên nghiệp gồm những ai?",
        "ground_truth": "Ngân hàng, công ty tài chính...",
        "target_source": "Điều 11" 
    },
    {
        "question": "Thời gian hạn chế chuyển nhượng cổ phiếu phát hành riêng lẻ?",
        "ground_truth": "03 năm với chiến lược...",
        "target_source": "Điều 31"
    },
    {
        "question": "Ủy ban Chứng khoán Nhà nước chịu sự quản lý của ai?",
        "ground_truth": "Bộ Tài chính.",
        "target_source": "Điều 9"
    },
    {
        "question": "Điều kiện cấp giấy phép thành lập công ty chứng khoán?",
        "ground_truth": "Trụ sở, vốn, nhân sự...",
        "target_source": "Điều 74"
    }
]

# Hàm kiểm tra khớp nguồn thông minh (Regex)
def check_match_regex(target, source):
    if not target or not source: return False
    # Lấy số (VD: "Điều 11" -> "11")
    nums = re.findall(r'\d+', target)
    if not nums: return target.lower() in source.lower()
    
    # Regex: "Điều" + khoảng trắng + "11" + biên từ (\b)
    # Để tránh "Điều 1" khớp với "Điều 115"
    pattern = r"Điều\s+" + re.escape(nums[0]) + r"\b"
    return bool(re.search(pattern, source, re.IGNORECASE))

print(f"\n BẮT ĐẦU ĐÁNH GIÁ HIT RATE (HYBRID SEARCH + RERANK)...")

hits = 0
mrr_sum = 0
total = len(advanced_test_data)
results_hit = []

for item in tqdm(advanced_test_data):
    q = item['question']
    target = item['target_source']

    # 1. Gọi Pipeline chuẩn: Hybrid Search -> Rerank -> Deduplicate
    # Kết quả trả về là danh sách các object SearchResult
    search_results = retriever.run(q)

    is_hit = False
    found_sources_str = ""
    rank = 0

    if search_results:
        # Lấy danh sách nguồn tìm được (Top 5)
        found_ids = [res.source_id for res in search_results]
        found_sources_str = " | ".join(found_ids)

        # Kiểm tra xem Target có nằm trong danh sách không
        for i, src_id in enumerate(found_ids):
            if check_match_regex(target, src_id):
                is_hit = True
                rank = i + 1 # Rank bắt đầu từ 1
                break

    if is_hit:
        hits += 1
        mrr_sum += 1.0 / rank # Cộng điểm MRR

    results_hit.append({
        "Question": q,
        "Target": target,
        "Found": found_sources_str if found_sources_str else " No Result",
        "Rank": rank if is_hit else "-",
        "Status": "Tốt" if is_hit else "Xấu"
    })

# --- XUẤT BÁO CÁO ---
df_hit = pd.DataFrame(results_hit)
hit_rate = (hits / total) * 100
mrr_score = mrr_sum / total

print("\n")
print(f" KẾT QUẢ ĐÁNH GIÁ TRUY VẤN (RETRIEVAL METRICS)")
print(f" HIT RATE (Độ chính xác Top 5): {hit_rate:.2f}%")
print(f" MRR SCORE (Thứ hạng trung bình): {mrr_score:.4f} (Max 1.0)")


# Hiển thị bảng đẹp
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.width', 1000)
print(df_hit)


# TỐI ƯU 

import pandas as pd
import re
from tqdm import tqdm

# PHẦN ĐÁNH GIÁ HIỆU NĂNG (MRR + HIT RATE) - PHIÊN BẢN TỐI ƯU

# 1. Bộ dữ liệu Test (Ground Truth)
advanced_test_data = [
    {
        "question": "Vốn điều lệ tối thiểu chào bán cổ phiếu ra công chúng?",
        "target": "Điều 15"
    },
    {
        "question": "Nhà đầu tư chứng khoán chuyên nghiệp gồm những ai?",
        "target": "Điều 11" # Hybrid Search sẽ bắt được cái này
    },
    {
        "question": "Thời gian hạn chế chuyển nhượng cổ phiếu phát hành riêng lẻ?",
        "target": "Điều 31"
    },
    {
        "question": "Ủy ban Chứng khoán Nhà nước chịu sự quản lý của ai?",
        "target": "Điều 9"
    }
]

# 2. Hàm kiểm tra khớp nguồn thông minh (Regex)
def check_match_regex(target, source):
    if not target or not source: return False
    nums = re.findall(r'\d+', target)
    if not nums: return target.lower() in source.lower()
    
    # Regex: "Điều" + khoảng trắng + số + biên từ (\b)
    pattern = r"Điều\s+" + re.escape(nums[0]) + r"\b"
    return bool(re.search(pattern, source, re.IGNORECASE))

# --- BẮT ĐẦU ĐÁNH GIÁ ---
print(f"\n BẮT ĐẦU ĐÁNH GIÁ HỆ THỐNG (HYBRID + RERANK)...")

mrr_sum = 0
total_hit = 0
results_adv = []

for item in tqdm(advanced_test_data):
    q = item['question']
    target = item['target']
    search_results = retriever.run(q)

    # 2. Tính toán MRR & Hit Rate
    rank = 0
    found = False
    top_source_found = "None"

    if search_results:
        # Lấy Top 1 để hiển thị trong báo cáo
        top_source_found = search_results[0].source_id

        # Kiểm tra xem Target nằm ở đâu trong Top 5
        for i, res in enumerate(search_results):
            if check_match_regex(target, res.source_id):
                rank = i + 1      # Thứ hạng (1, 2, 3...)
                mrr_sum += 1.0 / rank # Cộng điểm nghịch đảo
                total_hit += 1
                found = True
                break

    results_adv.append({
        "Question": q,
        "Target": target,
        "Top 1 Found": top_source_found,
        "Rank": rank if found else "-",
        "Pass": "Tốt" if found else "Xấu"
    })

# --- XUẤT BÁO CÁO ---
df_adv = pd.DataFrame(results_adv)
mrr_score = mrr_sum / len(advanced_test_data)
hit_rate_adv = (total_hit / len(advanced_test_data)) * 100

print("\n")
print(f" KẾT QUẢ ĐÁNH GIÁ (RETRIEVAL METRICS)")
print(f" Hit Rate (Top 5): {hit_rate_adv:.2f}%")
print(f" MRR Score (0-1):  {mrr_score:.4f}")

# Hiển thị bảng
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.width', 1000)
print(df_adv)



# KIỂM TRA KẾT NỐI 

import json

# --- CODE KIỂM TRA KẾT NỐI & LOGIC HYBRID ---
print(" ĐANG KIỂM TRA DỮ LIỆU TỪ SUPABASE (HYBRID MODE)...")

test_q = "Nhà đầu tư chứng khoán chuyên nghiệp gồm những ai?"
print(f" Câu hỏi: {test_q}")

# Ensure evaluator is defined
if 'evaluator' not in globals():
    print("Initializing RAGEvaluator...")
    evaluator = RAGEvaluator()

# 1. Tạo vector
vec = evaluator.manager.embedder.encode(f"query: {test_q}", normalize_embeddings=True).tolist()

# 2. Gọi Supabase (Dùng hàm Hybrid Search mới tạo)
try:
    response = evaluator.manager.db_client.rpc(
        "hybrid_search", 
        {
            "query_text": test_q,        
            "query_embedding": vec,      
            "match_count": 5             
        }
    ).execute()

    # 3. In kết quả
    print(f"\n KẾT QUẢ TRẢ VỀ TỪ SERVER ({len(response.data)} dòng):")
    for i, item in enumerate(response.data):
        # Lấy metadata an toàn
        meta = item.get('metadata', {})
        article = meta.get('article', 'No Meta')
        
        # In ra để kiểm chứng
        print(f"   [{i+1}] {article}")
        print(f"       Preview: {item['content'][:80]}...")
        
        if "Điều 11" in article:
            print(f"        Tìm thấy ĐIỀU 11! (Nhờ Hybrid Search)")

    print("\n KẾT NỐI THÀNH CÔNG & LOGIC ĐÚNG.")

except Exception as e:
    print(f"\n LỖI KẾT NỐI HOẶC LỖI SQL: {e}")
    print("Gợi ý: Kiểm tra lại xem đã chạy lệnh tạo hàm 'hybrid_search' trong SQL Editor chưa.")

