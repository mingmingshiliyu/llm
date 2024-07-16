import getpass
import os

from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

result = model.invoke(messages)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

parser.invoke(result)

chain = model | parser

chain.invoke(messages)

from langchain.chains import SimpleSequentialChain



prompt1=ChatPromptTemplate.from_template("what is the best name to describe a company ")

chain_one = LLMChain(llm=model,prompt=prompt1)

SimpleSequentialChain(chains=[chain_one],verbose=True)
