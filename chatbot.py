import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv(override=True)

# ----------------------------
# Initialize Session State
# ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# App Config & Title
# ----------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📖", layout="wide")
st.title("PDF Q&A Chatbot")

# ----------------------------
# Sidebar: Upload + Chat Summary
# ----------------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    st.markdown("---")
    st.subheader("Chat Summary")
    if st.session_state.messages:
        for msg in st.session_state.messages[-5:]:  
            role = "You" if msg["role"] == "user" else "Assistant"
            st.markdown(f"**{role}:** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    else:
        st.markdown("_No messages yet._")

# ----------------------------
# Set API Key and Load LLM
# ----------------------------
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GROQ_MODEL_NAME"]= os.getenv("GROQ_MODEL_NAME")

@st.cache_resource
def load_llm():
    return ChatGroq(
        model=os.environ["GROQ_MODEL_NAME"],
        temperature=0,
        max_tokens=1024
    )

llm = load_llm()

# ----------------------------
# Load PDF and Build Vectorstore
# ----------------------------
if uploaded_file and st.session_state.vectorstore is None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    st.session_state.vectorstore = db
    st.sidebar.success("PDF processed!")

# ----------------------------
# Display Full Chat History in Main Screen
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Chat Input & Response
# ----------------------------
if st.session_state.vectorstore is not None:
    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever()
                docs = retriever.get_relevant_documents(user_input)
                context = "\n\n".join(doc.page_content for doc in docs[:3])

                prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
You are a helpful assistant. Answer the question using only the context provided.
If the answer is not in the context, respond with: "I don't know. Please ask a question related to the contents of the PDF."

Context:
{context}

Question:
{question}

Answer:
"""
                )
                final_prompt = prompt_template.format(context=context, question=user_input)

                try:
                    response = llm.invoke([
                        {"role": "system", "content": "You are a helpful assistant that answers questions about PDFs."},
                        {"role": "user", "content": final_prompt}
                    ])
                    assistant_response = response.content
                except Exception as e:
                    assistant_response = f"❌ Error: {e}"

                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
else:
    st.info("Upload a PDF to begin.")
