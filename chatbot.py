import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain.prompts import PromptTemplate
import os

# ----------------------------
# App Configuration & Header
# ----------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📖")
st.title("Ask Questions About Your PDF")
st.write("Upload a PDF and ask any question!")

# ----------------------------
# Load the Language Model
# ----------------------------
@st.cache_resource
def load_llm():
    """
    Load a FLAN-T5 HuggingFace pipeline with set generation parameters.

    Returns:
        HuggingFacePipeline: LLM pipeline for text generation.
    """
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# ----------------------------
# PDF Upload and VectorStore
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file and st.session_state.vectorstore is None:
    # Save the uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF and create embeddings
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    st.session_state.vectorstore = db
    st.success("PDF Loaded Successfully!")

# ----------------------------
# Initialize Chat History
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# Question Answering Section
# ----------------------------
if st.session_state.vectorstore is not None:
    query = st.text_input("Ask a question about the PDF:")

    if query:
        retriever = st.session_state.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs[:3]])

        # Prompt construction for the LLM
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

        prompt = prompt_template.format(context=context, question=query)

        with st.spinner("Thinking..."):
            try:
                result = llm(prompt)
                if isinstance(result, list):
                    result = result[0].get('generated_text', result[0].get('text', ''))

                st.session_state.chat_history.append((query, result))
                st.write("### Answer:")
                st.write(result)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

    # ----------------------------
    # Display Chat History
    # ----------------------------
    if st.session_state.chat_history:
        st.write("---")
        st.write("### Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**{i}. Q:** {q}")
            st.markdown(f"**A:** {a}")
