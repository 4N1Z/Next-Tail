import os
from dotenv import load_dotenv

import cohere
import re
from langchain.vectorstores import AstraDB
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

import google.generativeai as genai

load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

# Google client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Cohere client 
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

set_llm_cache(InMemoryCache())

def split_and_convert(string):

    pattern = r'\([^)]+\)\s*>\s*'
    components = re.split(pattern, string)
    components = [c.strip() for c in components if c.strip()]
    return [{"text": component} for component in components]


def get_retriever(question):

    cohereEmbedModel = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
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
    Return you answer as :
    IF the question is to convert CSS into TAILWIND-CSS then return plane HTML as markdown;
    ELSE IF the question is to generate an UI for the prompt or anything related to CSS or TAILWIND give JSX code by default with TAILWIND-CSS ;

    And if you only receive CSS as input, first create a simple HTML template and then convert that CSS into TAILWIND-CSS with reference to the relevant docs from above context 
    and add it in HTML. Only give HTML with TAILWIND-CSS as reply and no need to give any reply to the answer, JUST THE HTML CODE IS ENOUGH. Thanks !
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def gemini_Chain(question):

    Gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
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

    # print(reranked)
    """Chain is of Runnable meaning:
        - It first calls the prompt and gives the context and question to it
        - Then it gives this prompt to gemini_llm
        - Then it takes the  O/P of Gemini_llm and parses it.
    """
    print("<<<<<<<< NOW ENTERING CHAIN >>>>>>>>>> \n\n")
    
    chain = RunnableMap({
        "context": lambda x: get_retriever(x["question"]),
        "question": lambda x: x["question"]
    }) | gemini_prompt | Gemini_llm | output_parser 

    # tc = TruChain(chain, app_id = "TruLens Dmo", feedbacks = [f_context_relevance, f_qa_relevance, f_groundedness])

    # tc({"question": question})

    res = chain.stream({"question": question})
    print("<<<<<<<<<<<< NOW GOT THE RESLT >>>>>>>>> \n\n")

    for r in res :
        print(r)
        
    # print(res)

def main():

    """To run the main code run : gemini_Chain()"""
    gemini_Chain(question="""Can you give tailwind for this CSS : background-image: linear-gradient(
		135deg,
		rgba(#752e7c, 0.35),
		rgba(#734a58, 0.1) 15%,
		#1b2028 20%,
		#1b2028 100% """)

    """To check if reranker is wokriing run get_retriever"""
    # get_retriever()


if __name__ == "__main__":
    main()