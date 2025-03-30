import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from consts import INDEX_NAME
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone



def ingest_docs():

    # INGESTING HTML FILES INTO VECTORSTORE
    # loader = ReadTheDocsLoader("langchain-docs/langchain.readthedocs.io/en/latest", encoding='utf8')
    #
    # raw_documents = loader.load()
    # print(f"loaded {len(raw_documents) }documents")
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    # documents = text_splitter.split_documents(documents=raw_documents)
    # print(f"Split into {len(documents)} chunks")

    # for doc in documents:
    #     old_path = doc.metadata["source"]
    #     new_url = old_path.replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})

    # -------------------------------------------------------------------

    # INGESTING TXT FILES INTO VECTORSTORE

    print('Ingesting')
    # loader = TextLoader("docs", encoding='UTF-8')
    directory = "docs"

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            loader = TextLoader(filepath, encoding='utf-8')
            # Perform your ingestion process here
            print(f"Ingesting {filename}")
            # Process the loaded documents
            document = loader.load()
            print("splitting")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
            texts = text_splitter.split_documents(document)
            print(f"created {len(texts)} chunks")

            print(f"Going to add {len(texts)} to Pinecone")
            # embeddings = OpenAIEmbeddings()
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            PineconeVectorStore.from_documents(texts, embeddings, index_name=INDEX_NAME)
            print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()

