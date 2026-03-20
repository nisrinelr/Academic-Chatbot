import streamlit as st
import os
import hashlib

from engine import process_pdf, initialize_chatbot

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="University Chatbot",
    page_icon="🎓"
)

st.title("🎓 University Academic Advisor")
st.markdown("Ask me anything about university regulations and standards!")

# --------------------------------------------------
# Setup
# --------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Session state init
if "bot" not in st.session_state:
    st.session_state.bot = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None


# --------------------------------------------------
# Sidebar – PDF Upload
# --------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader(
        "Upload Academic PDF",
        type="pdf"
    )


# --------------------------------------------------
# Helper: hash PDF to detect changes
# --------------------------------------------------
def hash_file(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


# --------------------------------------------------
# Handle PDF upload
# --------------------------------------------------
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    current_hash = hash_file(file_bytes)

    # Only reprocess if PDF changed
    if st.session_state.pdf_hash != current_hash:
        pdf_path = os.path.join(DATA_DIR, uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        with st.spinner("Processing PDF and indexing knowledge..."):
            chunks = process_pdf(pdf_path)
            st.session_state.bot = initialize_chatbot(chunks, current_hash)

        st.session_state.pdf_hash = current_hash
        st.session_state.messages = []  # reset chat
        st.success("PDF indexed successfully. Ready to chat! 🎉")


# --------------------------------------------------
# Chat history display
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------
if prompt := st.chat_input("What is the validation criteria for a Master?"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.bot:
        with st.chat_message("assistant"):
            response = st.session_state.bot({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
    else:
        st.warning("Please upload a PDF first!")