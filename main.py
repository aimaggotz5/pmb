from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import json
import uvicorn
from pyngrok import ngrok, conf, installer

import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

my_secret = os.environ['openaikey']
os.environ["OPENAI_API_KEY"] = my_secret

# Advanced method - Split by chunk

# Step 1: Convert PDF to text
import textract

doc = textract.process("./attention_is_all_you_need.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('attention_is_all_you_need.txt', 'w') as f:
  f.write(doc.decode('utf-8'))

with open('attention_is_all_you_need.txt', 'r') as f:
  text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
  return len(tokenizer.encode(text))


# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
  # Set a really small chunk size, just to show.
  chunk_size=512,
  chunk_overlap=24,
  length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
type(chunks[0])

# Get embedding model
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import Pinecone
import pinecone

# Get embedding model
embeddings = OpenAIEmbeddings()

# Load Pinecone API key
pinecone.init(
  api_key='db595835-7e78-42fb-a61b-6357d9391aa5',
  environment="us-east-1-aws"  # find next to API key in console
)

index_name = "langchain-demo"

#docsearch = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

# if you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Check similarity search is working
query = "terakhir kapan kak?"
docs = docsearch.similarity_search(query)
docs[0]

cc = ColabCode(port=14000, code=False)

app = FastAPI()


class input_parameters(BaseModel):
  phone: int
  name: str
  message: str


@app.get("/")
async def read_root():
  return {"message": "selamat datang di chatbot ittp"}


@app.head("/")
async def head():
  return {"message": "Success head"}


@app.get("/greet/{name}")
async def greet_name(name: str):
  return {"greeting": f"hallo {name}"}


@app.get("/item/")
async def greet_name(input_parameters: input_parameters):
  return {
    'name': input_parameters.name,
    'phone': input_parameters.phone,
    'message': input_parameters.message
  }


@app.get("/")
async def read_root():
  return {
    "message": "selamat datang di chatbot Institut Teknologi Telkom Purwokerto"
  }


@app.get("/greet/{name}")
async def greet_name(name: str):
  return {"greeting": f"hallo {name}"}


@app.get("/item/")
async def greet_name(input_parameters: input_parameters):
  return {
    'name': input_parameters.name,
    'phone': input_parameters.phone,
    'message': input_parameters.message
  }


@app.post('/chatbot')
async def diabetes_predd(input_parameters: input_parameters):

  input_data = input_parameters.json()
  input_dictionary = json.loads(input_data)

  phone = input_dictionary['phone']
  name = input_dictionary['name']
  message = input_dictionary['message']

  print(message)

  from langchain.chains import ConversationalRetrievalChain
  from langchain.memory import ConversationBufferMemory

  memory = ConversationBufferMemory(memory_key="chat_history",
                                    return_messages=True)
  qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0),
                                             docsearch.as_retriever(),
                                             memory=memory)

  query = message
  result = qa({"question": query})

  answer = result["answer"]
  return {answer}


#    return {message}
# sudah jadi

cc.run_app(app=app)
