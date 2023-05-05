# .streamlit folder was created to store the confi.toml file that contains the theme
import configparser
config = configparser.ConfigParser()
config.read('config.toml')
from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
#load_qa_chain takes llm as parameter. we can use any llm
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI # langchain is wrapper for llms from openAI

# Monitor how much money spent for question answered by the llm
# currently only work for opeai
from langchain.callbacks import get_openai_callback

#pip install faiss-cpu
#pipinstall tiktoken
#!pip install openai
#pip install langchain
#pip install streamlit




def main():
    load_dotenv()
    # Add custom CSS to set the background color to navy blue
    # define parameters for the webpage
    # Set the page configuration
    st.set_page_config(page_title="PDF Answer Pro", page_icon=":books:", layout="wide", initial_sidebar_state="auto")
    st.header("PDF Answer Pro")
    st.subheader("Get answers to your PDF questions")

    # Upload pdf file
    pdf = st.file_uploader("Upload your PDF:", type ="pdf")

    #extract text from the pdf
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text() # concatenate texts
        # st.write(text)

        # splitting the data into chunks
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size=1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        # check the chunks
        #st.write(chunks)

        # Create embeddings
        # With this embeddings we can use for similarity search
        embeddings = OpenAIEmbeddings()

        # create knowledge base: on which we are going to look for chunks and embeddings that are related to question
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show the user text input in streamlit
        def run_qa(user_question):
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cbk:
                response = chain.run(input_documents=docs, question=user_question)
            return response

        with st.form("my_form"):
            user_question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')
            if submit_button and user_question:
                response = run_qa(user_question)
                st.write(response)

    # Add a footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("Contact us: youremail@email.com")

if __name__ == '__main__':
    main()







