PDF Q&A Chatbot with Streamlit, LangChain, and Groq

This is a Streamlit web application that allows users to upload a PDF and chat with it using natural language. The app uses LangChain for document processing, FAISS for vector storage, and Groq's LLM (LLaMA 3) for answering questions based on the content of the uploaded PDF.

Features:

- Upload a PDF from the sidebar
- Ask questions about the PDF content
- Uses HuggingFace sentence embeddings and FAISS for document retrieval
- Powered by Groq's `llama3-8b-8192` large language model
- Chat history is shown in the main screen and sidebar

Installation:

1. Clone this repository
```bash
git clone https://github.com/FizaAli13/pdf-chatbot.git
cd pdf-chatbot
2. Create a virtual environment 
3. Install dependencies
pip install -r requirements.txt
4. Run the app
streamlit run c:/Users/DELL/Desktop/iternship/"pdf chatbot"/chatbot.py

Contact:
Created by Fiza Ali (https://github.com/FizaAli13) 
