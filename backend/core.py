import os
from dotenv import load_dotenv
from typing import Any
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from consts import INDEX_NAME

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI


pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm(query: str) -> Any:
    # chat = ChatOpenAI(verbose=True, temperature=0)
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Setting llm, embeddings model and vectorstore
    chat = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, embedding=embeddings
    )

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    template = """Използвайте само следните части от контекста, за да отговорите на въпроса в края.
                Ако не знаете отговора, кажете само, "не мога да отговоря на този въпрос" и 
                че може да се свържете със наш консултант който ще Ви съдейства, не се опитвайте да измисляте отговор.
                Използвайте максимум три изречения и дръжте отговора възможно най-сбит.
                Винаги казвайте „благодаря че попитахте“ в края на отговора.

        <context>
        {context}
        </context>

        Question: {input}
        """

    # Method 1
    custom_rag_prompt = PromptTemplate.from_template(template)

    stuff_documents_chain = create_stuff_documents_chain(chat, custom_rag_prompt)

    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=stuff_documents_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    # Formatting result, so it complies with Streamlit
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result

    # Method 2
    # prompt = PromptTemplate.from_template(template)
    # retriever = vectorstore.as_retriever()
    # doc_chain = create_stuff_documents_chain(chat, prompt)
    # chain = create_retrieval_chain(retriever, doc_chain)
    # result = chain.invoke({"input": query})
    # new_result = {
    #     "query": result["input"],
    #     "result": result["answer"],
    #     "source_documents": result["context"],
    # }
    # return new_result


if __name__ == "__main__":
    res = run_llm(query="Как да управлявам активите?")
    print(res['result'])

