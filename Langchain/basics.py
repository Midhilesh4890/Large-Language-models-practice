from langchain_openai import OpenAI, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']

# llm = OpenAI()

# print(llm('what is a fun fact about India'))


# Chat functions
chat = ChatOpenAI()

# result = chat([HumanMessage(content = 'Tell me what you know planets')])

# print(result.content)

result = chat([SystemMessage(content = 'You are a friendly assistant'), HumanMessage(content= 'Tell me what you know planets')])
print(result.llm_output)


