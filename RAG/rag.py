import os
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import logging
import pdb
from langsmith import Client


# Configure logging to write to a file
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(asctime)s - %(message)s',
                    filename='classifier.log')  # Specify the log file path here

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = Client()

os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']
LANGCHAIN_ENDPOINT = os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_PROJECT = os.environ['LANGCHAIN_PROJECT']

#### INDEXING ####

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
# Embed
vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question

answer = rag_chain.invoke("Summarize the content for me in easy words ")
print(answer)
