PDF Q&A Chatbot with Streamlit, LangChain, and Groq

This is a Streamlit web application that allows users to upload a PDF and chat with it using natural language. The app uses LangChain for document processing, FAISS for vector storage, and Groq's LLM (LLaMA 3) for answering questions based on the content of the uploaded PDF.

Features:

- Upload a PDF from the sidebar
- Ask questions about the PDF content
- Uses HuggingFace sentence embeddings and FAISS for document retrieval
- Powered by Groq's `llama3-8b-8192` large language model
- Chat history is shown in the main screen and sidebar

Installation
 1. Clone this repository

```bash
git clone https://github.com/FizaAli13/pdf-chatbot.git
cd pdf-chatbot
```
2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # For Windows
source venv/bin/activate  # For macOS/Linux
```
4. Install dependencies
```bash
pip install -r requirements.txt
```
5. Run the app
```bash
streamlit run chatbot.py
```


Contact:
Created by Fiza Ali 
GitHub: @FizaAli13 (https://github.com/FizaAli13) 
