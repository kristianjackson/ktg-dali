from PyPDF2 import PdfReader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

import pickle
from pathlib import Path
import os

import streamlit as st
from streamlit_chat import message

import io
import asyncio 

st.set_page_config(page_title="KTG DALI", page_icon="🤖")

initial_user_message = """I am an analyst at Kearney & Company and I need to extract key information from the uploaded document.

Founded in 1985, Kearney is the premier CPA firm focused on the Government, providing services across the financial management spectrum. 
Kearney has helped the Federal Government improve its financial operations’ overall effectiveness and efficiency; increase its level of accountability and compliance with laws, regulations, and guidance; and protect its funds from fraud, waste, and abuse. 
We understand the Federal Government’s need for efficiency and transparency.
"""

async def main():

    async def storeDocEmbeds(file, filename):
    
        reader = PdfReader(file)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])
        
        splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
        chunks = splitter.split_text(corpus)
        
        embeddings = OpenAIEmbeddings(openai_api_key = api_key)
        vectors = FAISS.from_texts(chunks, embeddings)
        
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)

        
    async def getDocEmbeds(file, filename):
        
        if not os.path.isfile(filename + ".pkl"):
            await storeDocEmbeds(file, filename)
        
        with open(filename + ".pkl", "rb") as f:
            global vectores
            vectors = pickle.load(f)
            
        return vectors
    

    async def conversational_chat(query):
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        # print("Log: ")
        # print(st.session_state['history'])
        return result["answer"]

    st.sidebar.header("Configurations:")
    st.sidebar.divider()
    llm_model = st.sidebar.radio("Which LLM would you like to use: ", ("gpt-4", "gpt-3.5-turbo"))
    api_key = st.sidebar.text_input(label="API Key:", type="password", help="API keys can be obtained at: https://platform.openai.com/account/api-keys", value=os.getenv('OPENAI_API_KEY'))
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("Choose a file", type="pdf")

    llm = ChatOpenAI(model_name=llm_model)
    chain = load_qa_chain(llm, chain_type="stuff")

    if 'history' not in st.session_state:
        st.session_state['history'] = []


    #Creating the chatbot interface
    st.title("KTG Document Analysis Language Interface (DALI)")

    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    if uploaded_file is not None:

        with st.spinner("Processing..."):
        # Add your code here that needs to be executed
            uploaded_file.seek(0)
            file = uploaded_file.read()
            # pdf = PyPDF2.PdfFileReader()
            vectors = await getDocEmbeds(io.BytesIO(file), uploaded_file.name)
            qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name=llm_model), retriever=vectors.as_retriever(), return_source_documents=True)

        st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome! You can now ask any questions regarding " + uploaded_file.name]

        if 'past' not in st.session_state:
            st.session_state['past'] = [initial_user_message]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="e.g: Summarize the paper in a few sentences", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = await conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i), avatar="https://images.emojiterra.com/google/noto-emoji/unicode-15/color/svg/1f916.svg")


if __name__ == "__main__":
    asyncio.run(main())