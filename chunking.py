# 1.Phân tích cấu trúc phân cấp (Hierarchical Parsing)

import re
import pandas as pd
import json
from typing import List, Dict, Any

# --- CẤU HÌNH ĐƯỜNG DẪN ---
INPUT_FILE = "/content/drive/MyDrive/[NLP - LAW_CHATBOT] /NLP_HNam_Data/Luật CK_cleaned.txt"
OUTPUT_JSON = "final_rag_data.json"

def parse_law_hierarchy(text: str) -> List[Dict[str, Any]]:
    """
    Phân tích văn bản luật thành cấu trúc cây (Hierarchy Tree).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    patterns = {
        'chapter': re.compile(r"^Chương\s+([IVXLCDM]+)\b[:\.]?(.*)", re.IGNORECASE),
        'section': re.compile(r"^Mục\s+(\d+)\b[:\.]?(.*)", re.IGNORECASE),
        'article': re.compile(r"^Điều\s+(\d+)\.\s*(.*)", re.IGNORECASE),
        'clause':  re.compile(r"^(\d+)\.\s+(.*)"),
        'point':   re.compile(r"^([a-zđ])\)\s+(.*)", re.IGNORECASE)
    }

    structure = []
    cur_chap = None; cur_sec = None; cur_art = None; cur_clause = None

    for line in lines:
        # 1. Chương
        m = patterns['chapter'].match(line)
        if m:
            cur_chap = {
                "number": m.group(1).strip(), "title": line, "children": []
            }
            structure.append(cur_chap)
            cur_sec = None; cur_art = None; cur_clause = None
            continue

        # 2. Mục
        m = patterns['section'].match(line)
        if m:
            cur_sec = {
                "number": m.group(1).strip(), "title": line, "children": []
            }
            if not cur_chap:
                cur_chap = {"number": "0", "title": "Quy định chung", "children": []}
                structure.append(cur_chap)
            cur_chap['children'].append(cur_sec)
            cur_art = None; cur_clause = None
            continue

        # 3. Điều
        m = patterns['article'].match(line)
        if m:
            cur_art = {
                "number": m.group(1).strip(),
                "title_text": m.group(2).strip(), # Chỉ lấy tên tiêu đề
                "full_title": line,               # Lấy cả cụm "Điều X..."
                "body": "", "children": []
            }
            parent = cur_sec if cur_sec else cur_chap
            if not parent:
                cur_chap = {"number": "0", "title": "Quy định chung", "children": []}
                structure.append(cur_chap)
                parent = cur_chap
            parent['children'].append(cur_art)
            cur_clause = None
            continue

        # 4. Khoản
        m = patterns['clause'].match(line)
        if m and cur_art:
            cur_clause = {"number": m.group(1).strip(), "content": m.group(2).strip()}
            cur_art['children'].append(cur_clause)
            continue

        # 5. Điểm & Nội dung nối tiếp
        m = patterns['point'].match(line)
        if m and cur_clause:
            cur_clause['content'] += f"\n{m.group(1)}) {m.group(2).strip()}"
        elif cur_clause:
            cur_clause['content'] += " " + line
        elif cur_art:
            cur_art['body'] += " " + line

    return structure


# 2.Làm phẳng & Mở rộng ngữ cảnh (Flattening & Context Expansion)

try:
    from underthesea import sent_tokenize
except ImportError:
    import re
    def sent_tokenize(text): return re.split(r'[.!?]+', text)

def flatten_to_dataframe_format(structure: List[Dict]) -> List[Dict]:
    rows = []

    for chap in structure:
        chap_code = f"Ch{chap['number']}"

        def process_article_node(art, sec_title=None, sec_code=""):
            # Tạo Parent Context
            full_context = f"{art['full_title']}\n{art['body']}"
            for c in art['children']:
                full_context += f"\n{c['number']}. {c['content']}"

            base_row = {
                "Chương": chap['title'],
                "Mục": sec_title if sec_title else "None",
                "Điều": f"Điều {art['number']}",
                "Tiêu_đề_Điều": art['title_text'], # Giữ tiêu đề sạch
                "full_context": full_context
            }

            dieu_code = f"D{art['number']}"

            # A. Xử lý Intro (Lời dẫn)
            if art['body'].strip():
                id_path = "|".join(filter(None, [chap_code, sec_code, dieu_code, "Intro"]))
                content = art['body'].strip()
                rows.append({
                    **base_row,
                    "Khoản": "Intro",
                    "Nội_dung": content, # Intro không có số khoản
                    "id_path": id_path,
                    "So_tu": len(content.split()),
                    "So_cau": len(sent_tokenize(content))
                })

            # B. Xử lý Khoản (SỬA Ở ĐÂY)
            for clause in art['children']:
                id_path = "|".join(filter(None, [chap_code, sec_code, dieu_code, f"K{clause['number']}"]))

                # Cũ: content = clause['content']
                # Mới: Thêm số thứ tự vào trước
                raw_content = clause['content']
                display_content = f"{clause['number']}. {raw_content}"

                rows.append({
                    **base_row,
                    "Khoản": clause['number'],
                    "Nội_dung": display_content, # <--- Đã thêm số "1. ", "2. "
                    "id_path": id_path,
                    "So_tu": len(display_content.split()),
                    "So_cau": len(sent_tokenize(raw_content))
                })

        for child in chap['children']:
            if 'title_text' in child:
                 process_article_node(child)
            else:
                 sec_code = f"M{child['number']}"
                 for art in child['children']:
                     process_article_node(art, sec_title=child['title'], sec_code=sec_code)

    return rows

def convert_df_to_rag_json(df_rows):
    rag_data = []
    for row in df_rows:
        # Tạo nội dung để Embed (Small Chunk)
        embed_text = f"{row['Chương']} > {row['Điều']} {row['Tiêu_đề_Điều']} > Khoản {row['Khoản']}\nNội dung: {row['Nội_dung']}"

        rag_data.append({
            "content": embed_text,
            "metadata": {
                "id_path": row['id_path'],
                "chapter": row['Chương'],
                "article": f"{row['Điều']} {row['Tiêu_đề_Điều']}",
                "clause": row['Khoản'],
                "full_context": row['full_context'], # <--- Lưu ngữ cảnh mở rộng vào đây
                "word_count": row['So_tu']
            }
        })
    return rag_data


# 3.Display các đoạn đã được Chunking

import pandas as pd
from IPython.display import display # Import display for rich output

# 1. Đọc file
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_text = f.read()
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại {INPUT_FILE}")
    raw_text = ""

if raw_text:
    # 2. Xử lý dữ liệu
    print("⏳ Đang xử lý Chunking & Context Expansion...")
    structure = parse_law_hierarchy(raw_text)
    flat_rows = flatten_to_dataframe_format(structure)

    # 3. Tạo DataFrame để hiển thị
    df = pd.DataFrame(flat_rows)

    # 4. IN RA KẾT QUẢ
    print(f" TỔNG SỐ ĐOẠN (CHUNKS) ĐÃ TẠO: {len(df)}")

    # Cấu hình hiển thị bảng đẹp (áp dụng cho tất cả DataFrame hiển thị sau đó)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 50) # Giới hạn độ rộng cột nội dung cho dễ nhìn

    # Hiển thị DataFrame trực tiếp
    display(df.head(100))

    # 5. Lưu file JSON cho bước Embedding
    rag_ready_data = convert_df_to_rag_json(flat_rows)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rag_ready_data, f, ensure_ascii=False, indent=2)
    print(f"\n-> Đã lưu file '{OUTPUT_JSON}' (chứa metadata mở rộng) thành công.")

# 4.Testing & Display output

from IPython.display import display
df_chunks = df

print("Các khoản của Điều 9:")

# Lọc dữ liệu
dieu_9 = df_chunks[df_chunks["Điều"] == "Điều 9"]

# Hiển thị bảng (chỉ hiện các cột quan trọng cho gọn)
display(dieu_9[["Điều", "Tiêu_đề_Điều", "Khoản", "Nội_dung", "id_path"]])

print("\n * Nội dung chi tiết Khoản 1 Điều 9:")
mask_9_1 = (df_chunks["Điều"] == "Điều 9") & (df_chunks["Khoản"] == "1")

if not df_chunks.loc[mask_9_1].empty:
    print(df_chunks.loc[mask_9_1, "Nội_dung"].values[0])
else:
    print(" Không tìm thấy Khoản 1 Điều 9 (Có thể do Điều này không có khoản, hoặc lỗi parsing).")

dieu_11 = df_chunks[df_chunks["Điều"] == "Điều 11"]
display(dieu_11[["Điều", "Tiêu_đề_Điều", "Khoản", "Nội_dung", "id_path"]])

mask_11_1 = (df_chunks["Điều"] == "Điều 11") & (df_chunks["Khoản"] == "1")
if not df_chunks.loc[mask_11_1].empty:
    print(df_chunks.loc[mask_11_1, "Nội_dung"].values[0])
else:
    print(" Không tìm thấy Khoản 1 Điều 11.")
