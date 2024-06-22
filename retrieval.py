import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def load_data_from_db(file_path):
    db = chromadb.PersistentClient(path=file_path)
    chroma_collection = db.get_or_create_collection("FAQs")
    chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return chroma_vector_store


if __name__ == "__main__":
    # Load data from database
    load_data_from_db("./storage/chroma")