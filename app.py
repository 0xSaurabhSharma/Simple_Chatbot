# ## Libraries

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

## Env &  LLM Set Up
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HF_TOKEN')

llm = ChatGroq(
    model='Gemma2-9b-It',
    temperature=0.8
)
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ## prompt | chain
# template = '''
# You are chatbot that remembers the previous chat and understand the context provided by it.
# Provide the appropriate response to given input.
# '''
# prompt = ChatPromptTemplate([
#     ('system', template),
#     MessagesPlaceholder(variable_name = 'chat_history'),
#     ('human', '{input}')
# ])
# chain = ...

# ## User, sessions & chat_history | get_chat_history
# chat_store = {}

# def get_chat(user_id: str, session_id: str):
#     """Retrieve or create a chat history for a given user and session."""
#     # If user doesn't exist, create a new entry
#     if user_id not in chat_store:
#         chat_store[user_id] = {}  # Initialize session dict for user

#     # If session doesn't exist, create a new ChatMessageHistory
#     if session_id not in chat_store[user_id]:
#         chat_store[user_id][session_id] = ChatMessageHistory()  # New session

#     return chat_store[user_id][session_id]  # Return chat history


# ## chat_history_chain
# ## UI Integration

# =========================================================

# Configure logging
# logging.basicConfig(level=logging.INFO)

class ChatManager:
    def __init__(self, llm):
        """Initialize the chat manager with a language model and chat storage."""
        self.llm = llm
        self.chat_store = {}  # Stores users & their session-wise chat histories
    
    def get_chat_history(self, user_id: str, session_id: str):
        """Retrieve or create chat history for a given user and session."""
        if user_id not in self.chat_store:
            self.chat_store[user_id] = {}  # Initialize session dict for the user
        
        if session_id not in self.chat_store[user_id]:
            self.chat_store[user_id][session_id] = ChatMessageHistory()  # New session
        
        return self.chat_store[user_id][session_id]

    def generate_response(self, user_id: str, session_id: str, input_text: str):
        """Generate response using LLM with stored chat history."""
        # Retrieve chat history for user/session
        chat_history = self.get_chat_history(user_id, session_id)

        # Define Chat Prompt Template
        template = '''
        You are a chatbot that remembers previous chat history and understands context.
        Provide an appropriate response to the given input.
        '''
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name='chat_history'),
            ("human", "{input}")
        ])

        # Create LLM Chain
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Generate response
        response = chain.run(input=input_text, chat_history=chat_history.messages)

        # Store the latest human & AI message
        chat_history.add_user_message(input_text)
        chat_history.add_ai_message(response)

        return response