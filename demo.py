import os
import PIL

import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyAZskhE6J9-Df_dlrLfnC3gKWcNYpOKtgM"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)
from llama_index.llms import Gemini


# GEMINI_MODELS = (
#     "models/gemini-pro",
#     "models/gemini-ultra",
# )

# # Normal Gemini/stream
# resp = Gemini(
# ).stream_complete("Sell me a pen")
# print(resp)

# for i in resp:
#     print("EACH STREAM",i)

model = genai.GenerativeModel('models/gemini-pro-vision')
res = model.generate_content([
    'Can you give the tailwind code for this ? IMAGE : ', PIL.Image.open('ssw.png')])

# for r in res :
#     print(r)
print(res.text)

# for r in res:
#     print(r)

# for m in genai.list_models():
#     if "generateContent" in m.supported_generation_methods:
#         print(m.name)

# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

# Gemini chat
# from llama_index.llms import ChatMessage, Gemini

# messages = [
#     ChatMessage(role="user", content="Hello friend!"),
#     ChatMessage(role="assistant", content="GUMM GUMM DOO ?!! Hooi, Whatya needee ...!??"),
#     ChatMessage(
#         role="user", content="Help me decide what to have for dinner."
#     ),
# ]\
# # resp = Gemini().chat(messages)
# llm  = Gemini()
# # print(resp)

# resp = llm.stream_chat(messages)
# for r in resp:
#     print(r)
# # async completion
# resp = await llm.acomplete("Llamas are famous for ")
# print(resp)
# # async streaming (completion)
# resp = await llm.astream_complete("Llamas are famous for ")
# async for chunk in resp:
#     print(chunk.text, end="")


# TO MARKDOWN
# from IPython.display import display
# from IPython.display import Markdown
# import textwrap

# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# retriever = vectorstore.as_retriever()

# prompt = hub.pull("rlm/rag-prompt")
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )