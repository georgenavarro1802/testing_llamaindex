import os
import time

from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, VectorStoreIndex

# load env vars
load_dotenv()

# OpenAI api key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# helper function (non-related with llamaindex)
def type_effect(text, delay=0.03):
    """
    Prints the text to the console, simulating typing by adding a delay after each character.
    Args:
    text (str): The text to be printed.
    delay (float, optional): The delay in seconds between each character. Default is 0.05 seconds.
    """
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)


if __name__ == '__main__':

    # # # Semantic Search
    print('Loading Documents...')
    documents = SimpleDirectoryReader('data').load_data()
    print('Creating Chunks, Embeddings and Nodes...')
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    print('Query Question: Qué es Uptime Institute?')
    response = query_engine.query('Qué es Uptime Institute?')
    print('Response:')
    print(type_effect(response.response))
