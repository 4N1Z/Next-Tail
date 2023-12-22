from django.shortcuts import render
import time
from rest_framework.views import APIView
# from rest_framework.renderers import JSONRenderer
# from rest_framework.response import Response
from django.views.generic import View  
from django.shortcuts import render
from django.http import StreamingHttpResponse
from rest_framework import status
from openai import OpenAI  # for OpenAI API calls
import time  # for measuring time duration of API calls
import os
from django.conf import settings
#--------------------------------------------------
# AI PART
#--------------------------------------------------
from dotenv import load_dotenv

import cohere
import re
from langchain.vectorstores import AstraDB
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from .nextTail import gemini_Chain
from .multiModalDemo import generate_code
from .models import image

import google.generativeai as genai
import environ
env = environ.Env()


ASTRA_DB_APPLICATION_TOKEN = settings.ASTRA_DB_APPLICATION_TOKEN
ASTRA_DB_ID = settings.ASTRA_DB_ID
ASTRA_DB_API_ENDPOINT = settings.ASTRA_DB_API_ENDPOINT
COHERE_API_KEY = settings.COHERE_API_KEY

# Google client
GOOGLE_API_KEY = settings.GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Cohere client
COHERE_API_KEY = settings.COHERE_API_KEY
co = cohere.Client(COHERE_API_KEY)
# Vector DB already loaded.(Already indexed)
# Get the vectors from the DB (using Retriever)


def split_and_convert(string):

    pattern = r'\([^)]+\)\s*>\s*'
    components = re.split(pattern, string)
    components = [c.strip() for c in components if c.strip()]
    return [{"text": component} for component in components]


def get_retriever(question):

    cohereEmbedModel = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

    astra_vector_store = AstraDB(
        embedding=cohereEmbedModel,
        collection_name="tailwind_docs_embeddings",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        # batch_size = 20,
    )

    # astraOP = astra_vector_store.as_retriever(search_kwargs={'k': 10})
    return rerank_documents(question,astra_vector_store.as_retriever(search_kwargs={'k': 10}))
    
    # return astra_vector_store.as_retriever(search_kwargs={'k': 2})


def rerank_documents(question,astraOP):

    # print(astraOP.get_relevant_documents("What is Tailwind ?"))
    retrieved_documents = astraOP.get_relevant_documents(question)
    # print(type(retrieved_documents[0]))
    # Convert the retrieved documents to the required format for rerank
    rerank_documents = [{"id": str(i), "text": doc} for i, doc in enumerate(retrieved_documents)]
    # print(rerank_documents)
    # print(type([rerank_documents]))
    docs = []
    for i in rerank_documents:
        docs.append(i['text'].page_content)

    results = co.rerank(query=question, documents=docs, top_n=3, model="rerank-multilingual-v2.0")
    # print(results)
    return results


def get_prompt_template():
    """ Prompt Template from Langchain
        with context and question. Can change to give dfferent format output
    """

    template = """You are an experienced senior TAILWIND-CSS developer, recognized for my efficiency in building websites and converting plain CSS into TAILWIND-CSS code. 
    Your expertise extends to NEXT JS development as well. You are eager to assist anyone with any TAILWIND-CSS related inquiries or tasks others may have.
    The context to answer the questions:
    {context}
    Return you answer as markdown.
    And if you only receive CSS as input, first create a simple HTML template and then convert that CSS into TAILWIND-CSS with reference to the relevant docs from above context 
    and add it in HTML. Only give HTML with TAILWIND-CSS as reply.

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    return prompt


def gemini_Chain(question):

    Gemini_llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True)

    """RETRIEVER WORKING - ✅"""
    # print(astra_retriever.get_relevant_documents("what is Tailwind?"))
    # astra_retriever.get_relevant_documents("what is Tailwind?")

    """RERANK THE RETREIVED DOCUEMENTS WORKING - ✅"""
    # use cohere reranker

    """PROMPT WORKING - ✅"""
    gemini_prompt = get_prompt_template()

    """OUTPUT PARSER WORKING - ✅"""
    output_parser = StrOutputParser()

    """If reranker working add the OP of rereanker in context.
        As of now it is the top n from relevant documnets.
    """
    # print(reranked)
    chain = RunnableMap({
        "context": lambda x: get_retriever(x["question"]),
        "question": lambda x: x["question"]
    }) | gemini_prompt | Gemini_llm | output_parser

    res = chain.stream({"question": question})
    for r in res:
        yield r

#--------------------------------------------------
        # Gemini CSS to Tailwind Converter
#--------------------------------------------------
class Gemini(APIView):
    # Vector DB already loaded.(Already indexed)
    # Get the vectors from the DB (using Retriever)
    def main_test():

        """To run the main code run : gemini_Chain()"""
        gemini_Chain(question="What is Tailwind ?")

        """To check if reranker is wokriing run get_retriever"""
        # get_retriever()
    
    def post(self,request):
        print(request.data)
        if request.data['message'] == '':
            chat = gemini_Chain(question='Send a greetings message for me and ask me to ask you a question to continue a conversation')
        else:
            chat = gemini_Chain(question=request.data['message'])
        response =  StreamingHttpResponse(chat,status=200, content_type='text/event-stream')
        return response
    
#--------------------------------------------------
        # UI to Code
#--------------------------------------------------

class UItoCode(APIView):

    def post(self,request):
        print(request.data)
        
        # url = request.data['url']
        url = "https://next-tail-space.blr1.digitaloceanspaces.com/next-tail/images/website.jpg"
        if request.data['message'] == '':
            chat = generate_code(url,'Send a greetings message for me and ask me to ask you a question to continue a conversation')
        else:
            chat = generate_code(url,request.data['message'])
        response =  StreamingHttpResponse(chat,status=200, content_type='text/event-stream')
        return response

# -------------------------------------------------- 
# --------------------------------------------------    
# client = OpenAI(
#     api_key='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',
# )

# class StreamGeneratorView(APIView):

#     def openaichatter(self,message):

#         stream = client.chat.completions.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content":message}],
#             stream=True,
#         )

#         print("here")
#         for chunk in stream:
#             yield chunk.choices[0].delta.content or ""
    
#     def post(self,request):
#         print(request.data)
#         if request.data['message'] == '':
#             chat = self.openaichatter('Send a greetings message for me and ask me to ask you a question to continue a conversation')
#         else:
#             chat = self.openaichatter(request.data['message'])
#         response =  StreamingHttpResponse(chat,status=200, content_type='text/event-stream')
#         return response

class HomeView(View):

    def get(self,request):
        return render(request,'index.html')
    
class cssToTailwindView(View):

    def get(self,request):
        return render(request,'tailwind_conversion.html')
    
class nextAIView(View):

    def get(self,request):
        return render(request,'next.html')
    
class UItoCodeView(View):

    def get(self,request):
        return render(request,'ui_to_code.html')
