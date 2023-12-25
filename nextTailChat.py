import os
import cohere

from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import AstraDB
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser

from langchain.chains import LLMChain
from langchain.chat_models import ChatCohere

from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv()
# Google client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")


"""
1. Load Gemini model
2. Set prompt template
3. Get the query from the user
4. Set retriever to get the documents related to that query
5. Add the message to Message schema of Langchain
6. Create a chain and add the llm Gemini to it.
7. Invoke the chain and get the answer
8. Append the result to Messages as AI
9. Also return query.
"""


def rerank_documents(question, astraOP):

    retrieved_documents = astraOP.get_relevant_documents(question)
    rerank_documents = [{"id": str(i), "text": doc}
                        for i, doc in enumerate(retrieved_documents)]
    docs = []
    for i in rerank_documents:
        docs.append(i['text'].page_content)

    results = co.rerank(query=question, documents=docs,
                        top_n=3, model="rerank-multilingual-v2.0")
    print("<<<<<<<<<<<< RERANK >>>>>>>>>>>>>\n\n",results)
    return results


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


class getPrompt():

    """ Prompt Templates with context and question. Can change to give dfferent format output.

        general_prompt() => Given to change the behaviour of the model
        answer_prompt() => Given the relevant docs and the question(can also give chat history)
        question_prompt() => Given the chat history and the question generate a suitable quesition to give in answer chain.

    """

    def general_prompt():
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

    def answer_prompt():

        template = """You are an experienced senior TAILWIND-CSS developer, recognized for my efficiency in building websites and converting plain CSS into TAILWIND-CSS code. 
        Your expertise extends to NEXT JS development as well. You are eager to assist anyone with any TAILWIND-CSS related inquiries or tasks others may have.
        The context to answer the questions:
        {context}
        Return your answer based on the above context only.

        As the question you will get Question + a summarry of the conversation so far.
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template=template)
 
        return prompt
    
    def question_prompt():

        template = """Here is the chat history so far as a List of Dict with roles: USER and AI .
                Chat History : {context}
                here is the latest question from USER . , 
                USER : {question}
                Give a summary of the Chat history along with the latest question.Do not diverge from the context and no need to add any thing new.
                ONLY SUMMARIZE THE CONVERSATION SO FAR. 
                Query:"""
        prompt = PromptTemplate.from_template(template=template)
        # prompt = PromptTemplate(
        #     input_variables=["chat_history", "human_input"], template=template
        # )
        return prompt

class conversationConfig ():
    """Conversation settings for the chat.
    """

    def get_messages():
        """This schema can only be followed if we are using a chain from lanchain
        Here i am creating  cusom runnable interface so this is not needed.
        Can rather creata a simple dictionary for the memory.
        """
        # messages = [
        #     SystemMessage(content="You are a Helpful "),
        #     HumanMessage(content="Hi AI, how are you today?"),
        #     AIMessage(content="I'm great thank you. How can I help you?"),
        #     HumanMessage(content="I'd like to understand string theory.")
        # ]

        """This is the default message history for our chat, created by Aniz
            [List[Dict[str, str]]]
        """
        messages = [
            {'Role': 'USER', 'message': 'You are MR.Softie,a higly intelligent NEXT JS and TAILWIND-CSS Developer ?'},
            {'Role': 'AI', 'message': 'Yeaooh, You are absolutely correct! I only answer doubts regarding TAILWIND-CSS and NEXT JS from their documentation only and will give reply.'},
            # {'sender': 'Human', 'message': 'Yeah sure !'}
        ]

        return messages


    def make_history(typeReq, message, message_history):
        message_history.append({"Role":typeReq,"message":message})
        return message_history

def coChat():
    # """Cohere is used to summarize the chatHistory to get memory of the conversation.and generate a question"""
    # cohere_llm = co.chat(
    #     message="Summarize the following conversation.Also take chat_history without loosing any context the question. Here is the question: "+question,
    #     model="command-light",
    #     chat_history = message_history,
    #     preamble_override= """You are an expert in summarizing without loosing context. When you are summarizing, dont forget to take the chat history and when summarizing do not loose 
    #     context.Your context is the Chat History. Only summarize the contents of CHAT HISTORY along with the question. Also keep in mind to not loose the question."""  
    # )
    # print(cohere_llm)
    ...

def chatSummary(question,message_history):
    """Takes in the question and message_history and passes it to a custom chain to generate a question with context"""

    Gemini_llm = ChatGoogleGenerativeAI(
        model='gemini-pro',
        temperature=0.1,
        convert_system_message_to_human=True
    )
    cohere_llm = ChatCohere(
        model="command-light",
        temperature=0.1
    )
    queustionPrompt = getPrompt.question_prompt()

    chain = RunnableMap({
        "context": lambda x : conversationConfig.make_history("USER",x["question"],message_history),
        "question": lambda x : x["question"],
        # "chat_history":memory,
    }) | queustionPrompt | Gemini_llm | StrOutputParser()

    summary = chain.invoke({"question": question})
    print("<<<<<<<<<<<<<< SUMMARY >>>>>>>>>>>>>\n\n", summary)
    return summary


def chain(question,message_history):

    Gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True)
    
    """Prompt Temp - """
    chatPrompt = getPrompt.answer_prompt()

    """Set Message and creating a human Query Template : - """

    # humanQuery = HumanMessage(
    #     content=question
    # )
    # messages.append(humanQuery)
    # messages = []
    # # messages["USER"] = {"message": question}
    # user={"sender": "USER", "message": "Hello there!"}
    # messages.append(user)
    # print(messages)

    """Either give context, human_input and chat_history(using memory) together.
        or give context and messages or far(chat History + current)
    """

    chain = RunnableMap({
        "context": lambda x: get_retriever(x["question"]),
        "question": lambda x: chatSummary(x["question"],message_history),
    }) | chatPrompt | Gemini_llm | StrOutputParser()

    res = chain.invoke({"question": question})
    # res = chain.invoke({"question": question})
    # print(get_retriever(question))
    # print(res)

    conversationConfig.make_history("AI", res, message_history)
    print(message_history)

    # for r in res:
    #     print(r)
    # messages.append()

if __name__ == "__main__":

    message_history = conversationConfig.get_messages()
    chain("How can we add custom colors in TailWind ?", message_history)
    # chatSummary("What is Tailwind", message_history)
    # chain("What is Tailwind")
