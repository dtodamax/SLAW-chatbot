import json
import re
import requests
import streamlit as st
from typing import List, Dict

from rag import RAGConfig, create_supabase_client, retrieve_context

import os

#Đặt key trong biến môi trường để bảo mật (để public được trên GitHub)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY env var")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE env vars")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-32b"

MODEL = DEFAULT_MODEL
TEMPERATURE = 0 # Temp để bằng 0 để tránh LLM trả lời dài dòng quá
MAX_TOKENS = 6000
RPC_NAME = "hybrid_search"

st.set_page_config(page_title="SLAW Chatbot", page_icon="⚖️", layout="centered")


# SESSION
if "messages" not in st.session_state:
    st.session_state.messages = []  #Hỗ trợ lưu lại lịch sử hội thoại với người dùng để chatbot trả lời theo ngữ cảnh trước đó đang diễn ra
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = GROQ_API_KEY


# TEXT CLEANUP
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def clean_model_text(text: str) -> str: #Xóa phần thinking của LLM hiển thị trên giao diện
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


# Step này là gửi prompt lên API rồi nhận về câu trả lời
def groq_chat_stream(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: int = 0,
    max_tokens: int = 6000,
    timeout: int = 60,
):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    with requests.post(GROQ_URL, headers=headers, json=payload, stream=True, timeout=timeout) as resp:
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise RuntimeError(f"Groq HTTP {resp.status_code}: {err}")

        for raw in resp.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content

def build_system_prompt() -> str:
    return (
        "Bạn là trợ lý pháp lý SLAW.\n"
        "Mục tiêu: trả lời chính xác, dễ hiểu, bám sát nội dung pháp luật trong CONTEXT.\n\n"

        "QUY TẮC BẮT BUỘC:\n"
        "1) Chỉ sử dụng thông tin có trong CONTEXT. Không tự bịa hoặc viện dẫn quy định không có trong CONTEXT.\n"
        "2) Không hiển thị hoặc nhắc tới thẻ <think> hay suy nghĩ nội bộ.\n"
        "3) Nếu CONTEXT không đủ để kết luận, nói rõ 'Chưa đủ thông tin trong tài liệu được cung cấp' và nêu 1–3 câu hỏi cần bổ sung.\n\n"

        "CÁCH TRẢ LỜI:\n"
        "- Mở đầu bằng 1–2 câu trả lời chung/kết luận tổng quát, đi thẳng vào trọng tâm câu hỏi.\n"
        "- Sau đó nêu căn cứ cụ thể theo cấu trúc: 'Điều …, Khoản …' (nếu có Điểm thì nêu thêm) theo dữ liệu bạn lấy được, tuyệt đối không bịa điều và khoản, diễn giải ngắn gọn, dễ hiểu.\n"
        "- Nếu câu hỏi liên quan đến điều kiện, thủ tục hoặc quy trình, trình bày dưới dạng gạch đầu dòng hoặc checklist.\n"
        "- Thêm mục 'Lưu ý:' chỉ khi trong CONTEXT có ngoại lệ, điều kiện kèm theo hoặc giới hạn áp dụng.\n"
        "- Chỉ khi CONTEXT không đủ để kết luận: nói rõ 'Chưa đủ thông tin trong tài liệu được cung cấp' và hỏi 1–3 câu để làm rõ thêm.\n"
        "- Với các tình huống pháp lý cụ thể hoặc nhạy cảm (xử phạt, tranh chấp, khiếu nại, kiện tụng…), khuyến nghị tham vấn luật sư.\n"
    )


# Khởi tạo RAG
rag_cfg = RAGConfig(
    supabase_url=SUPABASE_URL,
    supabase_key=SUPABASE_KEY,
    rpc_function=RPC_NAME,
    device="cuda",
)
supabase = create_supabase_client(rag_cfg)


# UI
st.title("⚖️ SLAW Chatbot")
st.caption("Giải đáp mọi thắc mắc về luật pháp cùng SLAW — bạn của mọi nhà.")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander("📚 Sources"): #Hỗ trợ hiển thị sources cho câu trả lời
                for s in m["sources"]:
                    st.markdown(f"- {s}")

user_text = st.chat_input("Nhập câu hỏi của bạn...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        system_prompt = build_system_prompt()

        # Step này sẽ loại bỏ câu hỏi user vừa hỏi khỏi history, sau đó readd lại câu hỏi đó nhưng lúc này đã kèm context cho LLM trả lời kĩ hơn
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-10:]
            if m["role"] in ("user", "assistant")
        ]
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        # RAG retrieve
        try:
            context, sources = retrieve_context(supabase, rag_cfg, user_text)
        except Exception as e:
            context, sources = "", []
            st.warning(f"⚠️ Lỗi retrieval: {e}")

        # Đưa cho LLM câu hỏi kèm theo context pháp luật để trả lời chính xác hơn
        user_with_context = f"CONTEXT:\n{context}\n\nCÂU HỎI:\n{user_text}"

        messages_for_llm = [{"role": "system", "content": system_prompt}] + history + [
            {"role": "user", "content": user_with_context}
        ]

        placeholder = st.empty()
        acc = ""

        try:
            for delta in groq_chat_stream(
                api_key=st.session_state.groq_api_key,
                model=MODEL,
                messages=messages_for_llm,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            ):
                acc += delta
                placeholder.markdown(clean_model_text(acc))

            final = clean_model_text(acc) or "(Không có phản hồi từ mô hình)"
            st.session_state.messages.append(
                {"role": "assistant", "content": final, "sources": sources}
            )

        except Exception as e:
            err = f"❌ Lỗi gọi Groq API: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

    st.rerun()
