from flask import Flask,render_template,jsonify,request
from src.helper import download_higging_face_embeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from src.prompt import *
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')

embeddings = download_higging_face_embeddings()

Pinecone(
    api_key=PINECONE_API_KEY,
)

docsearch = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX,
    embeddings=embeddings
)

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"]
)

chain_type_kwargs = {"prompt":PROMPT}

llm = CTransformers(
    model="",
    config={
        'max_new_tokens':512,
        'temperature':0.9
    }
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents = True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")