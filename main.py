import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def google_search(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        search_results = []
        for result in soup.find_all("div", class_="tF2Cxc"):
            link = result.find("a")["href"]
            search_results.append(link)
        return search_results
    else:
        print("Failed to retrieve search results.")
        return []

query = input("Enter your search query: ")
print("====== Performing google search ======")
results = google_search(query)

print("====== Loading html documents ======")
loader = AsyncChromiumLoader(results)
html_documents = loader.load()

print("====== Extracting content From html documents ======")
bs_transformer = BeautifulSoupTransformer()
text_documents = bs_transformer.transform_documents(html_documents, tags_to_extract=["p", "li", "div", "a"])

splitter = RecursiveCharacterTextSplitter(
     chunk_size=300,
    chunk_overlap=30,
)

print("====== Splitting text documents ======")
docs_split = splitter.transform_documents(text_documents)

embedding = HuggingFaceEmbeddings()

print("====== Creating and persisting vector database ======")
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=docs_split,
    embedding=embedding,
    persist_directory=persist_directory
)

vectordb.persist()

print("====== Performing retrieval and answering by llm ======")
llm = Ollama(model='llama2')

retrieval_qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(search_type="mmr"),
    memory=memory
)

result = retrieval_qa({"question": query})
print(result["answer"])

while True:
    query = input("Ask a follow up question: ")
    result = retrieval_qa({"question": query})
    print(result["answer"])