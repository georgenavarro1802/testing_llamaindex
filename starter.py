"""
    LlamaIndex Tutorial
    Installation & Setup Ref: https://docs.llamaindex.ai/en/stable/getting_started/installation.html
    High Level Concepts  Ref: https://docs.llamaindex.ai/en/stable/getting_started/concepts.html
    chroma Ref: https://docs.trychroma.com/getting-started
"""
import os
import sys
import logging
import time

import chromadb
from dotenv import load_dotenv

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage, ServiceContext, download_loader
)
from llama_index.llms import PaLM
from llama_index.vector_stores import ChromaVectorStore

# load env vars
load_dotenv()

# OpenAI api key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Internal testing (measure time)
start = time.perf_counter()


# # Prueba de SimpleWeb Reader
# SimpleWebPageReader = download_loader("SimpleWebPageReader")
#
# loader = SimpleWebPageReader()
#
# if not os.path.exists("storage_tt"):
#     documents = loader.load_data(urls=[
#         'https://www.tychetools.com',
#         'https://www.tychetools.com/company',
#         'https://www.tychetools.com/news',
#         'https://www.tychetools.com/solutions/ecoaas',
#         'https://www.tychetools.com/solutions/aqaas',
#         'https://www.tychetools.com/solutions/iris',
#         'https://www.tychetools.com/solutions/soter',
#     ])
#     index = VectorStoreIndex.from_documents(documents)
#     index.storage_context.persist("storage_tt")
# else:
#     storage_context = StorageContext.from_defaults(persist_dir="storage_tt")
#     index = load_index_from_storage(storage_context)
#
# query_engine = index.as_query_engine()
# response = query_engine.query('What is the Tychetools mission?')
# # response = query_engine.query('Who is the CEO of Tychetools?')
# # response = query_engine.query('What is ECOaaS product about?')
# print("Response:\n", response)

# # check if storage already exists
# if not os.path.exists("storage"):
#     # load the documents and create the index
#     print('Loading ...')
#     documents = SimpleDirectoryReader("data").load_data()
#     print('Indexing ...')
#     index = VectorStoreIndex.from_documents(documents)
#     print('Storing ...')
#     index.storage_context.persist("storage")
# else:
#     print('Loading Index from Storage ...')
#     storage_context = StorageContext.from_defaults(persist_dir="storage")
#     index = load_index_from_storage(storage_context)

# Query Engine (either way we can now query the index)
# print('Querying ...')
# query_engine = index.as_query_engine()
# question = "What did the author do growing up?"
# print(f"Question: {question}")
# response = query_engine.query(question)
# print("Response:\n", response)

# # # # # Customizations # # # # #
# # 1) I want to PARSE MY DOCUMENTS INTO SMALLER CHUNKS
# print('ServiceContext chunk_size=1000 ...', end=' ')
# service_context = ServiceContext.from_defaults(chunk_size=1000)
# print('DONE')
# print('Stage - Loading ...', end=' ')
# documents = SimpleDirectoryReader('data').load_data()
# print('DONE')
# print('Stage - Indexing and Storing ...', end=' ')
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)
# print('DONE')
# print('Stage - Querying ...', end=' ')
# query_engine = index.as_query_engine()
# print('DONE')

# # 2) I want to USE A DIFFERENT VECTOR STORE
# # Chroma is a database for building AI applications with embeddings
# print('Chroma - Creating Client ...', end=' ')
# chroma_client = chromadb.PersistentClient()
# print('DONE')
# print('Chroma - Creating Collection ...', end=' ')
# chroma_collection = chroma_client.create_collection('pabloyjoseluis')
# print('DONE')
# print('Vector Store - from chroma collection ...', end=' ')
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# print('DONE')
# print('Storage Context - Loading ...', end=' ')
# # StorageContext defines the storage backend for where the documents, embeddings, and indexes are stored
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# print('DONE')
# print('Stage - Loading ...', end=' ')
# documents = SimpleDirectoryReader('data').load_data()
# print('DONE')
# print('Stage - Indexing and Storing ...', end=' ')
# index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
# print('DONE')
# print('Stage - Querying ...', end=' ')
# query_engine = index.as_query_engine()
# print('DONE')

# # 3) I want to RETRIEVE MORE CONTEXT WHEN I QUERY
# print('Stage - Loading ...', end=' ')
# documents = SimpleDirectoryReader('data').load_data()
# print('DONE')
# print('Stage - Indexing and Storing ...', end=' ')
# index = VectorStoreIndex.from_documents(documents)
# print('DONE')
# print('Stage - Querying ...', end=' ')
# # Here, we configure the retriever to return the top 5 most similar documents (instead of the default of 2)
# query_engine = index.as_query_engine(similarity_top_k=5)
# print('DONE')

# # 4) I want to USE A DIFFERENT LLM (this will need auth in google)
# print('ServiceContext llm=PaLM() ...', end=' ')
# service_context = ServiceContext.from_defaults(llm=PaLM())
# print('DONE')
# print('Stage - Loading ...', end=' ')
# documents = SimpleDirectoryReader('data').load_data()
# print('DONE')
# print('Stage - Indexing and Storing ...', end=' ')
# index = VectorStoreIndex.from_documents(documents)
# print('DONE')
# print('Stage - Querying (using service context) ...', end=' ')
# query_engine = index.as_query_engine(service_context=service_context)
# print('DONE')

# # 5) I want to USE A DIFFERENT RESPONSE MODE
# print('Stage - Loading ...', end=' ')
# documents = SimpleDirectoryReader('data').load_data()
# print('DONE')
# print('Stage - Indexing and Storing ...', end=' ')
# index = VectorStoreIndex.from_documents(documents)
# print('DONE')
# print('Stage - Querying (response_mode="tree_summarize") ...', end=' ')
# # Ref: https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes.html
# query_engine = index.as_query_engine(response_mode='tree_summarize')
# print('DONE')

# # 6) I want to STREAM THE RESPONSE BACK
# print('Stage - Loading ...', end=' ')
# documents = SimpleDirectoryReader('data').load_data()
# print('DONE')
# print('Stage - Indexing and Storing ...', end=' ')
# index = VectorStoreIndex.from_documents(documents)
# print('DONE')
# print('Stage - Querying (streaming=True) ...', end=' ')
# # Ref: https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/streaming.html
# query_engine = index.as_query_engine(streaming=True)
# print('DONE')

# # Question & Answer (if query_engine is as_query_engine)
# question = 'What did the author do growing up?'
# print('Question:', question)
# response = query_engine.query(question)
# print('Response:\n', response)

# # # 6) I want a CHATBOT INSTEAD OF Q&A
# print('Stage - Loading ...', end=' ')
# documents = SimpleDirectoryReader('data').load_data()
# print('DONE')
# print('Stage - Indexing and Storing ...', end=' ')
# index = VectorStoreIndex.from_documents(documents)
# print('DONE')
# print('Stage - Querying (as_chat_engine) ...', end=' ')
# # Ref: https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern.html
# query_engine = index.as_chat_engine()
# print('DONE')
# question1 = 'What did the author do growing up?'
# print('Question1:', question1)
# response1 = query_engine.chat(question1)
# print('Response1:\n', response1)
# question2 = 'Oh interesting, tell me more.'
# print('Question2:', question2)
# response2 = query_engine.chat(question2)
# print('Response2:\n', response2)

# Internal testing (measure time)
end = time.perf_counter()
print(f"ExecTime {(end - start):.2f}s")
