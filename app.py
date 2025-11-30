from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = "your open-ai key"

app = Flask(__name__)
# app.secret_key = "motorlaw_secret_key"
# app.config["SESSION_TYPE"] = "filesystem"
# Session(app)

embed = OpenAIEmbeddings(model="text-embedding-3-large")
compressor = FlashrankRerank(top_n=3)
llm = ChatOpenAI(model_name="gpt-5-nano-2025-08-07",temperature=1)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")



def qna1(question, state, lan):
    db = FAISS.load_local(
        folder_path=f"{os.getcwd()}/vectorstore/{state}",
        allow_dangerous_deserialization=True,
        embeddings=embed,
    )

    template = f"""
        You are a knowledgeable **motor vehicle legal advisor** for the state of {state}.
        Your sole purpose is to help users with **motor vehicle laws, fines, licensing, traffic rules, 
        and road transport regulations** applicable in {state}.

        Follow these rules carefully:
        1. If the user's question is **not related to motor vehicles, traffic, or transport**, politely say:
        → "Apologies, I am not aware of this at this moment."
        (Do NOT make up an answer.)
        2. Generate answer in the language {lan}
        3. Use the given <context> only if relevant to answer.
        4. Do not mention you are an AI or referencing documents.
        5. Be concise, formal, and answer in very crisp and short manner.
        6. Follow greeting examples strictly if asked (Hi/Hello → greet back, etc.).
        7. If user replies (ok/thank you/bye →  Thank you').
        8. If asked about your creator always mention "Abel".

        <context>
        {{context}}
        </context>

        Question:
        {{question}}

        Helpful Answer:
    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT, "verbose": False}

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(search_kwargs={"k": 3}),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        memory=memory,
    )

    result = qa_chain({"query": question})
    return result["result"]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    state = data.get("state")
    lan = data.get("language")

    try:
        answer = qna1(question, state, lan)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
