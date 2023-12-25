from trulens_eval import Feedback, Huggingface, Tru, TruChain, Select
from trulens_eval.feedback import Groundedness, Langchain as fLangchain
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
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory, ConversationSummaryMemory, ConversationBufferMemory

from langchain.schema.output_parser import StrOutputParser
import google.generativeai as genai
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI

from langchain_core.runnables import RunnablePassthrough
from trulens_eval import Feedback, Huggingface, Tru, TruChain

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

hugs = Huggingface()
tru = Tru()


def truLens(chain):

    Langchain_tru = fLangchain(chain=chain)
    grounded = Groundedness(groundedness_provider=Langchain_tru)
    import numpy as np
    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons,
                 name="Groundedness")
        .on(Select.RecordCalls.retrieve.rets.collect())
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = (
        Feedback(Langchain_tru.relevance_with_cot_reasons,
                 name="Answer Relevance")
        .on(Select.RecordCalls.retrieve.args.query)
        .on_output()
    )

    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(Langchain_tru.qs_relevance_with_cot_reasons,
                 name="Context Relevance")
        .on(Select.RecordCalls.retrieve.args.query)
        .on(Select.RecordCalls.retrieve.rets.collect())
        .aggregate(np.mean)
    )

    return f_groundedness, f_qa_relevance, f_context_relevance


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

    return rerank_documents(question, astra_vector_store.as_retriever(search_kwargs={'k': 10}))


def get_retriever_Chat():

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

    # return rerank_documents(question,astra_vector_store.as_retriever(search_kwargs={'k': 10}))
    return astra_vector_store.as_retriever()


def rerank_documents(question, astraOP):

    retrieved_documents = astraOP.get_relevant_documents(question)

    rerank_documents = [{"id": str(i), "text": doc}
                        for i, doc in enumerate(retrieved_documents)]

    docs = []
    for i in rerank_documents:
        docs.append(i['text'].page_content)

    results = co.rerank(query=question, documents=docs,
                        top_n=3, model="rerank-multilingual-v2.0")
    # print(type(results))
    print(results)

    return results


def get_prompt_template():
    """ Prompt Template from Langchain
        with context and question. Can change to give dfferent format output
    """

    template = """You are an experienced senior TAILWIND-CSS developer, recognized for my efficiency in building websites and converting plain CSS into TAILWIND-CSS code. 
    Your expertise extends to NEXT JS development as well. You are eager to assist anyone with any TAILWIND-CSS related inquiries or tasks others may have.
    The context to answer the questions:
    {context}
    Also this is the memory so far :
    {chat_history}
    Return you answer as markdown.
    And if you only receive CSS as input, first create a simple HTML template and then convert that CSS into TAILWIND-CSS with reference to the relevant docs from above context 
    and add it in HTML. Only give HTML with TAILWIND-CSS as reply and no need to give any reply to the answer, JUST THE HTML CODE IS ENOUGH. Thanks !
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def chat_RAG(question):

    Gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True)

    gemini_prompt = get_prompt_template()
    output_parser = StrOutputParser()

    retriever = get_retriever_Chat()

    memory = ConversationSummaryMemory(
        llm=Gemini_llm, memory_key="chat_history", return_messages=True
    )

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=Gemini_llm, retriever=retriever, memory=memory)
    # f_context_relevance, f_qa_relevance, f_groundedness = truLens(qa)

    # tc = TruChain(qa, app_id = "TruLens Dmo", feedbacks = [f_context_relevance, f_qa_relevance, f_groundedness])
    # print(tc(question))
    format_docs = get_retriever(question)
    # print(format_docs)

    # Extract documents from the results
    extracted_documents = dict(result.document['text'] for result in format_docs)
    formatted_documents = [{'document': {'text': doc['text']}} for doc in extracted_documents]

    # print(extracted_documents)

    rag_chain = (
        {"context": formatted_documents, "question": RunnablePassthrough()}
        | gemini_prompt
        | Gemini_llm
        | output_parser
    )

    # tc_recorder = TruChain(rag_chain, app_id="app_id")
    # with tc_recorder as recording:
    #     resp = rag_chain(question)
    # tru_record = recording.records[0]

    # print(resp)
    # print(tru_record)

    # chain = RunnableMap({
    #     "context": lambda x: get_retriever(x["question"]),
    #     "memory" :  memory,
    #     "question": lambda x: x["question"]
    # }) | gemini_prompt | Gemini_llm | output_parser

    # print(qa(question))

from trulens_eval.feedback.provider import OpenAI
def main():
    question = "What is tailwind ?"
    chat_RAG(question)
    # get_retriever([question])


if __name__ == "__main__":
    main()
