from dotenv import load_dotenv
import os
from src.helper import load_pdf,text_split,download_higging_face_embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX = os.environ.get('PINECONE_INDEX')

pc = Pinecone(
    api_key=PINECONE_API_KEY,
)

PINECONE_INDEX_NAME = pc.Index(PINECONE_INDEX)

print("Initialised Pinecone")

extracted_data = load_pdf("data")
text_chunks = text_split(extracted_data)
print(len(text_chunks))
embeddings = download_higging_face_embeddings()


docsearch = PineconeVectorStore(
    index=PINECONE_INDEX_NAME,
    embedding=embeddings
)

for chunk in text_chunks:
    doc = Document(
        page_content=chunk.page_content,
        metadata=chunk.metadata
    )

    docsearch.add_documents([doc])
