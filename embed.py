from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#path for the pdf
def load_embed_save():
        texts=[]
        for states in os.listdir(f"{os.getcwd()}/data/"):
        # for states in os.listdir(data):
            if os.path.isdir(f"{os.getcwd()}/data/{states}"):
                for files in os.listdir(f"{os.getcwd()}/data/{states}"):
                    pages=PyPDFLoader(f"{os.getcwd()}/data/{states}/{files}")
                    texts.extend(pages.load_and_split())
                db = FAISS.from_documents(texts, embeddings)
                db.save_local(folder_path = f"{os.getcwd()}/vectorstore/{states}")
            else:
                pages=PyPDFLoader(f"{os.getcwd()}/data/{states}")
                texts.extend(pages.load_and_split())
                db = FAISS.from_documents(texts, embeddings)
                db.save_local(folder_path = f"{os.getcwd()}/vectorstore/Central Government/")


        
                
        return "Database created/updated"
        # if os.path.exists(f"{os.getenv('ROOT_DIRECTORY')}/vectorstore/index.faiss"):
        #     local_db = FAISS.load_local(folder_path=f"{os.getenv('ROOT_DIRECTORY')}/vectorstore/",embeddings=embeddings)
        #     #merging the new embedding with the existing index store
        #     local_db.merge_from(db)
        #     print("Merge completed")
        #     local_db.save_local(folder_path=f"{os.getenv('ROOT_DIRECTORY')}/vectorstore/")
        #     print("Updated index saved")
        # else:
        #     db.save_local(folder_path=f"{os.getenv('ROOT_DIRECTORY')}/vectorstore/")
        #     print("New store created...")
        

# out=load_embed_save("./new_rule.pdf")
if __name__=="__main__":
    #  for file_name in os.listdir(f"{os.getenv('ROOT_DIRECTORY')}/data"):
    #       print(file_name)
    # load_embed_save(f"{os.getenv('ROOT_DIRECTORY')}/data/{file_name}")
     load_embed_save()