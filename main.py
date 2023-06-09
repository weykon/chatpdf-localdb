# -*- coding: utf-8 -*-

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from getpass import getpass
from dotenv import load_dotenv
import sys
import os
load_dotenv() 
# 获取环境变量
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    print("OPENAI_API_KEY is not set")
    OPENAI_API_KEY = getpass()
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
else:
    print(f"OPENAI_API_KEY = {OPENAI_API_KEY}")

persist_directory = 'db'

loader = UnstructuredPDFLoader("11.pdf")
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()

# to get local file ?
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

vectordb = db.from_documents(pages,embeddings,persist_directory=persist_directory)

vectordb.persist()

chain = load_qa_chain(OpenAI(temperature=0.6), chain_type="stuff")

def run_qa():
    while True:
        query = input()
        print('type your question: \n')
        docs = vectordb.as_retriever().get_relevant_documents(query)
        output = chain.run(input_documents=docs, question=query)
        print(output)
        print('------------------')
try:
    run_qa()
except Exception as e:
    print(f"execute error：{e}")
    run_qa()
