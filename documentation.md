PDF Chatbot Documentation

Overview:
This app allows users to upload a PDF and ask questions about its content via a natural language interface. It uses a large language model (LLaMA 3 via Groq), LangChain for orchestration, FAISS for document retrieval, and Streamlit for the UI.

Main Components:
1. load_llm()
- Loads the llama3-8b-8192 model from Groq for answering questions.
- Uses LangChain’s ChatGroq integration.

2. PDF Upload and Processing
- Users can upload a PDF from the sidebar.
- PDF is split into text chunks using RecursiveCharacterTextSplitter.
- Embeddings are generated using sentence-transformers/all-MiniLM-L6-v2.
- Chunks are indexed using FAISS for fast similarity search.

3. Question Answering
- Users type questions in the chat input field.
- The app retrieves relevant document chunks using vector similarity.
- A prompt is generated using retrieved context and passed to the LLM.
- The LLM responds based only on the context provided.

How It Works:
- Upload PDF
The user uploads a PDF via the sidebar.

- Document Processing
The app extracts, chunks, and embeds the text into a FAISS vectorstore.

- Ask Questions
The user asks a question. Relevant text chunks are retrieved.

- Answer Generation
The app constructs a prompt with the context and calls the Groq LLM to get an answer.

- Chat History
The conversation history is displayed in the main screen and summarized in the sidebar.