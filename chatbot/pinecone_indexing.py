import os
from pinecone import Pinecone, PodSpec
from langchain_pinecone import Pinecone as pineconestore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from sentence_transformers import SentenceTransformer
from typing import List
from dotenv import load_dotenv
import logging
import pdb
from utils import *

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(asctime)s - %(message)s',
                    filename='classifier.log')  # Specify the log file path here

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']

# Directory containing the documents
directory = 'QnA\data'


def load_docs(directory: str) -> List:
    """Loads all documents from a directory.

    Args:
        directory (str): The directory containing the documents.

    Returns:
        List[Document]: A list of Document objects.
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    """
    Splits a list of Documents into smaller chunks of text.

    Args:
        documents (List[Document]): A list of Document objects.
        chunk_size (int, optional): The size of each chunk, in characters. Defaults to 1000.
        chunk_overlap (int, optional): The number of characters to overlap between chunks. Defaults to 20.

    Returns:
        List[Document]: A list of Document objects, where each Document is a smaller chunk of text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def get_similar_docs(query: str, k: int = 2, score: bool = False) -> List:
    """
    Returns a list of the k most similar documents to the query, along with their similarity scores.

    Args:
        query (str): The query vector.
        k (int, optional): The number of results to return. Defaults to 2.
        score (bool, optional): Whether to return similarity scores. Defaults to False.

    Returns:
        List[Tuple[Document, float]]: A list of tuples, where each tuple contains a Document and a similarity score.
    """
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Load documents
logger.info('Loading documents...')
documents = load_docs(directory)
logger.info(f'Total documents loaded: {len(documents)}')

# Split documents into smaller chunks
logger.info('Splitting documents...')
chunks = split_docs(documents)
logger.info(f'Total chunks after splitting: {len(chunks)}')


# Initialize OpenAI Embeddings
logger.info('Initializing OpenAI Embeddings...')
embeddings = OpenAIEmbeddings(deployment='ada')
logger.info('OpenAI Embeddings initialized successfully')

# Embed a sample query for testing
query_result = embeddings.embed_query("Hello world")
logger.info(f'Total query results: {len(query_result)}')

# Initialize Pinecone
logger.info('Initializing Pinecone...')
index_name = "langchain-demo"
pc = Pinecone(PINECONE_API_KEY)

try:
    # Create index if it does not exist
    if index_name not in pc.list_indexes():
        pc.create_index(name=index_name,
                        dimension=len(query_result),
                        metric="cosine",
                        spec=PodSpec(environment="gcp-starter"))
        logger.info(f'Index "{index_name}" created successfully')
except:
    logger.info(f'Index "{index_name}" already exists')

logger.info('Pinecone initialized successfully')

# Load documents into Pinecone index
logger.info('Loading documents into Pinecone index...')
index = pineconestore.from_documents(chunks, embeddings, index_name=index_name)
logger.info('Documents loaded into Pinecone index successfully')

query = "How is India economy"
similar_docs = get_similar_docs(query)
logger.info(f'Total documents loaded: {similar_docs}')

model = SentenceTransformer('all-MiniLM-L6-v2')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
