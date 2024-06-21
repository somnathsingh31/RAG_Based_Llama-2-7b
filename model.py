import chromadb
import torch
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from load_data import load_data_from_db


def model_llm(vector_store):
    system_prompt="""
    You are a Q&A assistant. Your goal is to answer questions as
    accurately as possible based on the instructions and context provided.
    """
    #Embedding Model
    embed_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # Initialize the LLM
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        # loading model in 8bit for reducing memory
        model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
    )

    service_context=ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )
    index=VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context, similarity_top_k=20)
    query_engine=index.as_query_engine()
    return query_engine



if __name__ == "__main__":
    vector_store = load_data_from_db("./storage/chroma")
    query_engine = model_llm(vector_store)
    response=query_engine.query("what are the price for your plan?")
    print(response)