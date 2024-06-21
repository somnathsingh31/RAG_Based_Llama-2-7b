import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def ingest_data(file_path):
    #Read PDF data
    documents = SimpleDirectoryReader(file_path).load_data()
    #Embedding
    embed_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    #Vector Database (Chroma)
    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection("FAQs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    chroma_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model, show_progress=True)
    print("Data successfully ingested into the database")


if __name__ == "__main__":
    ingest_data('/Data')
