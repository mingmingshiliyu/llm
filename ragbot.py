# build your own rag bot

#load html/pdf/powerpoint document into a vector db
# query vector database
#insert the results of the query into a prompt
#pass the prompt into llm
#ask question to the llm,response to you

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md
from unstructured.partition.pptx import partition_pptx
from unstructured.staging.base import dict_to_elements

import unstructured_client
from unstructured_client.models import operations, shared

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


from langchain.prompts.prompt import  PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain,LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

#https://api.unstructuredapp.io/general/v0/general
def unstructuredCall():
    # Before calling the API, replace filename and ensure sdk is installed: "pip install unstructured-client"
    # See https://docs.unstructured.io/api-reference/api-services/sdk for more details



    client = unstructured_client.UnstructuredClient(
        api_key_auth="LwShkNu866kJmqZrDWymF7jua2rNP4",
        server_url="https://api.unstructuredapp.io",
    )

    filename = "/Users/wtst45x/Documents/eBook-How-to-Build-a-Career-in-AI.pdf"
    with open(filename, "rb") as f:
        data = f.read()
    #解析ppt文件
    pptx_elements=partition_pptx(filename=filename)

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=data,
                file_name=filename,
            ),
            # --- Other partition parameters ---
            # Note: Defining 'strategy', 'chunking_strategy', and 'output_format'
            # parameters as strings is accepted, but will not pass strict type checking. It is
            # advised to use the defined enum classes as shown below.
            strategy=shared.Strategy.AUTO,
            languages=['eng'],
        ),
    )

    try:
        res = client.general.partition(request=req)
        print(res.elements[0].get("text"))


        #过滤掉不想搜索的内容,比如header
        # tables=[el for el in res.elements if el.category=="Table"]

        #将文档element分块为chunk
        elements=chunk_by_title(res.elements+pptx_elements)
        #分块后,将文档加载到向量数据库
        documents=[]
        for element in elements:
            metadata=element.metadata.to_dict()
            del metadata["languages"]
            metadata["source"]=metadata["filename"]
            documents.append(Document(page_content=element.text,metadata=metadata["metadata"]))
        #上面这个documents就是要存储到乡里那个数据库的
        # 嵌入这个文档到数据库
        embeddings=OpenAIEmbeddings()
        vectorstore=Chroma.from_documents(documents,embeddings)

        #从数据库中搜索,得到很多个结果,这些结果会作为prompt的一部分结合后传递给llm
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k":6}
        )
        #设置prompt template,用langchain管理prompt template
        template="""
        you are an ai assistant for answering questions about the document,you are given the following extracted parts of a long document and a questions,if you don`t know the answer,just say "i`m not sure",don`t try to make it yourself
        if the question is not about ai,politely inform them that you are the document about ai robot.
        question: {question}
        =======
        {context}
        =======
        answer in markdown
        """
        prompt=PromptTemplate(template=template,input_variables=["question","context"])
        llm=OpenAI(temperature=0)
        doc_chain=load_qa_with_sources_chain(llm,chain_type="map_reduce")
        question_generator_chain=LLMChain(llm=llm,prompt=prompt)
        qa_chain=ConversationalRetrievalChain(
            retriever=retriever,
            question_generator=question_generator_chain,
            combine_docs_chain=doc_chain
        )
        print(qa_chain.invoke({
            "question":"how do you think about ai?",
            "chat_history":[],
        })["answer"])

        #之前设定了source,所以也可以用source进行过滤,source可以是文件名,可以是类别
        filter_retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k":1,"filter":{"source":"ai"}}
        )
        filter_chain = ConversationalRetrievalChain(
            retriever=filter_retriever,
            question_generator=question_generator_chain,
            combine_docs_chain=doc_chain
        )
        print(filter_chain.invoke({
            "question":"how do you think about ai?",
            "chat_history":[],
            # "filter":filter,
        })["answer"])



    except SDKError as e:
        print(e)

if __name__ == '__main__':
    unstructuredCall()