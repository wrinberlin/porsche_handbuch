# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:41:40 2024

@author: Wolfgang Reuter

USAGE: Run from command line: 
    streamlit run c:\porsche_demos\src\pdf_agent_handbuch.py

"""

# =============================================================================
# Imports
# =============================================================================

import streamlit as st

OPENAI_API_KEY = st.secrets["openai"]["api_key"]

import os
from pathlib import Path


from PyPDF2 import PdfReader

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import tiktoken  # Library for counting tokens

# Get the current working directory (will be the project root in Streamlit Cloud)
project_root = Path(os.getcwd())

# =============================================================================
# Paths and Variables
# =============================================================================

# Define constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300
MAX_TOKENS = 4096  # Adjust this based on the model's token limit
RESPONSE_BUFFER = 500  # Reserve tokens for the response

IMAGE_PATH = project_root / "Porsche_Demos" / "illustrations" / "taycan.jpg"   # Taycan Image

def calculate_token_length(text, model_name="gpt-4"):
    """Calculate the token length of a text."""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def main():
    st.set_page_config(page_title="Ask your PDF with System Prompt")
    st.header("Chatte mit Deinem Handbuch")
    
    # Display an image
    st.image(IMAGE_PATH)
    
    # Define the system prompt
    system_prompt = """
    Du bist ein erfahrener Porsche Ingenieur, der Nutzern hilft, sich in dem 
    Handbuch, was Dir bereitgestellt ist, zu orientieren. Du beantwortest die 
    Nutzerfragen möglichst einfach und verständlich - und sagst Ihnen auch, 
    wo Du die entsprechenden Informationen gefunden hast.  
    """
    
    pdf = st.file_uploader("Upload your PDF file", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator='\n', 
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len)
        chunks = text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) 
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        if "question_count" not in st.session_state:
            st.session_state.question_count = 1  # Initialize the question counter
            
        llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)
        
        # Loop to display questions and answers
        for i in range(st.session_state.question_count):
            user_question = st.text_input(f"Question {i + 1}:", key=f"question_{i}")
            
            if user_question:
                st.write(f"Antwort auf Frage {i + 1}: ")
                docs = knowledge_base.similarity_search(user_question)
                
                # Combine the most relevant context dynamically within the token limit
                context = ""
                token_count = calculate_token_length(system_prompt + user_question, model_name="gpt-4")
                for doc in docs:
                    doc_tokens = calculate_token_length(doc.page_content, model_name="gpt-4")
                    if token_count + doc_tokens + RESPONSE_BUFFER < MAX_TOKENS:
                        context += doc.page_content + "\n\n"
                        token_count += doc_tokens
                    else:
                        break
                
                # Create a chat-based prompt template
                system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                human_message = HumanMessagePromptTemplate.from_template("Kontext:\n{context}\n\nFrage: {question}")
                chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                
                # Use LLMChain to execute the chat prompt
                qa_chain = LLMChain(llm=llm, prompt=chat_prompt)
                response = qa_chain.run(context=context.strip(), question=user_question)
                
                # Handle potentially cut-off answers
                if response.strip().endswith("..."):
                    st.warning("Die Antwort kann auf Grund von Textlängenbeschränkungen unvollständig sein.")
                
                # Display the response
                st.write(response)
        
        # Automatically add a new question input field below the last one
        st.session_state.question_count += 1  # Increment the question count for the next question slot
        
        # Exit button
        if st.button('Exit'):
            st.write("Exiting the application...")
            os._exit(0)

if __name__ == "__main__":
    main()
