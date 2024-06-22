# RAG-Based Chatbot Using LLaMa 2

## Overview

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) based chatbot using the LLaMa 2 model. The chatbot is designed to respond to user queries using a company FAQ as the knowledge base. The FAQ content is ingested, vectorized, and indexed to facilitate efficient and accurate query responses.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Model Details](#model-details)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Efficient Query Handling**: Utilizes vectorized embeddings for rapid query responses.
- **Scalable Knowledge Base**: FAQ content is stored in a scalable vector store.
- **Advanced NLP**: Employs the LLaMa 2 model for high-quality language understanding and generation.
- **Secure**: Implements SSL 128-bit encryption for data security.

## Installation

### Prerequisites

Ensure you have Python 3.6 or later installed. You also need to have `pip` for installing Python packages.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/somnathsingh31/RAG_Based_Llama_2_7b.git
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

- `pypdf`
- `chromadb`
- `transformers`
- `einops`
- `accelerate`
- `langchain`
- `bitsandbytes`
- `sentence_transformers`
- `llama_index`
- `llama-index-embeddings-langchain`
- `llama-index-llms-huggingface`
- `langchain-community`
- `llama-index-vector-stores-chroma`

## Usage

### Loading the FAQ Document

1. Place your FAQ document in the `/content` directory.
2. Run the ingestion pipeline to load and process the document.

### Running the Chatbot

To start the chatbot, use the provided scripts or notebook to initialize the model, load the index, and query the chatbot.

## Project Structure

```
RAG-Based-Chatbot-Using-LLaMa-2/
│
├── data/
│   └── (Place your FAQ documents here)
│
├── notebooks/
│   └── RAG_Chatbot_Notebook.ipynb  # Interactive notebook for chatbot setup and testing
│
├── src/
│   ├── ingestion.py  # Scripts for ingesting and processing documents
│   ├── retrieval.py  # Scripts for setting up the retrieval pipeline
│   └── chatbot.py  # Main script for running the chatbot
│
├── requirements.txt  # List of required Python packages
└── README.md  # This file
```

## Ingestion Pipeline

The ingestion pipeline processes and indexes the FAQ document for efficient retrieval. It includes steps for loading the document, vectorizing the text, and storing the embeddings in a vector store.

## Retrieval Pipeline

The retrieval pipeline handles incoming queries and fetches relevant responses from the indexed FAQ data. It loads the pre-indexed data from the vector store and utilizes a query engine to fetch and rank relevant responses.

## Model Details

- **Model**: LLaMa 2 (Llama-2-7b-chat-hf)
- **Embeddings**: Hugging Face's sentence-transformers
- **Vector Store**: Chroma for efficient storage and retrieval

## Limitations

- **Data Scope**: The chatbot is limited to the provided FAQ document. It may not handle questions outside this context.
- **Model Size**: Running the LLaMa 2 model requires significant computational resources.

## Future Work

- **Expand Knowledge Base**: Incorporate additional documents or data sources.
- **Improve Response Generation**: Enhance the natural language generation capabilities of the chatbot.
- **Optimize Performance**: Explore ways to reduce the computational load of the chatbot.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---