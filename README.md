🚗 Motor Vehicle Legal Advisor

A simple Flask-based AI chatbot that provides state-specific motor vehicle law information using OpenAI, LangChain, and FAISS.

⚙️ Features

Answers traffic rules, fines, licensing queries

State & language selection

PDF-based knowledge base

FAISS vector search + reranking

Clean web chat interface

📦 Setup
1. Install dependencies
pip install -r requirements.txt
2. Add API key
Create .env:
OPENAI_API_KEY=your_key



▶ Start the app

python app.py


Open:
http://127.0.0.1:5000/
