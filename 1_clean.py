import re
import os
import unicodedata
import logging
from typing import List

# Cấu hình Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- CẤU HÌNH ĐƯỜNG DẪN ---
INPUT_FILE_PATH = "/content/drive/MyDrive/[NLP - LAW_CHATBOT] /NLP_HNam_Data/Luật CK_cleaned.txt"
OUTPUT_FILE_PATH = "/content/drive/MyDrive/[NLP - LAW_CHATBOT] /NLP_HNam_Data/Luật CK_final_clean.txt"


class LegalTextCleaner:
    """
    Class chịu trách nhiệm làm sạch và chuẩn hóa văn bản luật.
    Được thiết kế để xử lý các lỗi đặc thù khi trích xuất text từ PDF/Docx.
    """

    def __init__(self, text: str):
        self.raw_text = text
        self.clean_text = ""

    def process(self) -> str:
        """
        Thực thi pipeline làm sạch dữ liệu theo tuần tự.
        """
        if not self.raw_text:
            return ""

        text = self.raw_text

        # Bước 1: Chuẩn hóa Unicode và khoảng trắng
        text = self._normalize_unicode_and_spaces(text)

        # Bước 2: Sửa lỗi dấu câu bị dính liền với chữ
        text = self._fix_sticky_punctuation(text)

        # Bước 3: Loại bỏ các thành phần rác (số trang, dòng trống)
        text = self._remove_artifacts(text)

        # Bước 4: Nối các dòng bị ngắt sai (Line Merging)
        text = self._merge_broken_lines(text)

        self.clean_text = text
        return self.clean_text

    def _normalize_unicode_and_spaces(self, text: str) -> str:
        """
        Chuyển đổi văn bản về chuẩn Unicode NFC và xử lý khoảng trắng thừa.
        """
        logger.info("Processing: Normalizing Unicode and whitespace...")

        # Chuẩn hóa về NFC (Normalization Form C)
        text = unicodedata.normalize('NFC', text)

        # Loại bỏ ký tự ẩn (Zero-width space, BOM, Non-breaking space)
        text = text.replace('\u200b', '').replace('\ufeff', '').replace('\xa0', ' ')

        # Loại bỏ các ký tự điều khiển không in được (trừ \n và \t)
        text = "".join([c for c in text if c == '\n' or c == '\t' or c >= ' '])

        # Thay thế nhiều khoảng trắng liên tiếp bằng 1 khoảng trắng
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    def _fix_sticky_punctuation(self, text: str) -> str:
        """
        Thêm khoảng trắng sau dấu câu nếu bị dính liền với ký tự tiếp theo.
        Ví dụ: "quy định.Điều 1" -> "quy định. Điều 1"
        """
        logger.info("Processing: Fixing sticky punctuation...")

        # Regex pattern:
        # Group 1: Ký tự chữ cái (bao gồm tiếng Việt)
        # Group 2: Dấu câu (.,;:)
        # Group 3: Ký tự chữ hoặc số liền kề
        pattern = r'([a-zA-ZăâđêôơưĂÂĐÊÔƠƯ])([.,;:])(\w)'

        return re.sub(pattern, r'\1\2 \3', text)

    def _remove_artifacts(self, text: str) -> str:
        """
        Lọc bỏ các dòng số trang, header/footer và các dòng quá ngắn vô nghĩa.
        """
        logger.info("Processing: Removing artifacts (page numbers, noise)...")

        lines = text.split('\n')
        cleaned_lines = []

        # Regex nhận diện số trang (VD: "12", "- 12 -", "Trang 12")
        page_number_pattern = re.compile(r'^(\-?\s*)?(Trang\s*)?\d+(\s*\-?)?$', re.IGNORECASE)

        # Regex nhận diện điểm khoản (a), b), c)...) để tránh xóa nhầm
        list_item_pattern = re.compile(r'^[a-zđ]\)$')

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if page_number_pattern.match(line):
                continue

            # Bỏ qua dòng quá ngắn (< 3 ký tự) trừ khi nó là điểm khoản (a, b, c)
            if len(line) < 3 and not list_item_pattern.match(line):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _merge_broken_lines(self, text: str) -> str:
        """
        Nối các dòng bị ngắt sai do định dạng PDF.
        Logic: Nếu dòng trên không kết thúc bằng dấu ngắt câu và dòng dưới
        không phải là tiêu đề mới -> Nối lại.
        """
        logger.info("Processing: Merging broken lines...")

        lines = text.split('\n')
        if not lines:
            return ""

        merged_lines = []
        current_line = lines[0]

        # Regex kiểm tra dấu kết thúc câu (. ! ? ;)
        end_punct_pattern = re.compile(r'[.!?;:]$')

        # Regex kiểm tra tiêu đề mới (Chương, Điều, Mục)
        header_pattern = re.compile(r'^(Chương|Điều|Mục)\s')

        # Regex kiểm tra đầu mục (1. hoặc a))
        bullet_pattern = re.compile(r'^(\d+\.|[a-zđ])\)')

        for i in range(1, len(lines)):
            next_line = lines[i].strip()

            # Điều kiện để nối dòng:
            # 1. Dòng hiện tại KHÔNG kết thúc bằng dấu ngắt câu.
            # 2. Dòng tiếp theo KHÔNG phải là tiêu đề (Chương/Điều).
            # 3. Dòng tiếp theo KHÔNG phải là đầu mục (1., a)).
            should_merge = (
                not end_punct_pattern.search(current_line) and
                not header_pattern.match(next_line) and
                not bullet_pattern.match(next_line)
            )

            if should_merge:
                current_line += " " + next_line
            else:
                merged_lines.append(current_line)
                current_line = next_line

        # Thêm dòng cuối cùng
        merged_lines.append(current_line)

        return "\n".join(merged_lines)

def main():
    """Hàm thực thi chính."""

    # 1. Kiểm tra file đầu vào
    if not os.path.exists(INPUT_FILE_PATH):
        logger.error(f"Input file not found: {INPUT_FILE_PATH}")
        # Tạo file dummy để test nếu cần
        return

    # 2. Đọc file
    try:
        logger.info(f"Reading file: {INPUT_FILE_PATH}")
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_content = f.read()
    except IOError as e:
        logger.error(f"Failed to read file: {e}")
        return

    # 3. Xử lý làm sạch
    cleaner = LegalTextCleaner(raw_content)
    global clean_text # Declare clean_text as global
    clean_text = cleaner.process() # Assign to global variable

    # 4. Lưu kết quả
    try:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        logger.info(f"Successfully saved cleaned text to: {OUTPUT_FILE_PATH}")

        # Preview kết quả
        print(clean_text[:2000])
        print("="*40 + "\n")

    except IOError as e:
        logger.error(f"Failed to write output file: {e}")

if __name__ == "__main__":
    main()
  
