import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OctoAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set page config at the start
st.set_page_config(page_title='Virtual Shopping Assistant', page_icon=':robot_face:')

# Load environment variables
load_dotenv()
OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

# Load the datasets
@st.cache_data
def load_data():
    user_purchases = pd.read_csv('User_Past_Purchase.csv')
    product_info = pd.read_csv('Product_Database.csv')
    return user_purchases, product_info

user_purchases, product_info = load_data()

# Merge the datasets on 'Product ID'
merged_data = pd.merge(user_purchases, product_info, on="Product ID")

# Create documents from merged data
documents = merged_data.apply(lambda x: f"User {x['User ID']} bought {x['Product Name']} ({x['Category']}) for ${x['Price']} on {x['Purchase Date']}. Remaining stock: {x['Stock Level']}.", axis=1).tolist()

# Initialize embeddings and vector store
embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/")
vector_store = FAISS.from_texts(documents, embedding=embeddings)
retriever = vector_store.as_retriever()

# Set up language model endpoint
llm = OctoAIEndpoint(
    model="meta-llama-3-70b-instruct",
    max_tokens=1024,
    presence_penalty=0,
    temperature=0.1,
    top_p=0.9,
)

template = """
You are a virtual shopping assistant. Based on the past purchases and a budget of $500, recommend a combination of electronics products that together stay within the budget.
Question: {question}
Context: {context}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Combining the retriever and language model to form a query chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit interface

# Sidebar for additional information or navigation
with st.sidebar:
    st.title('User Information')
    st.write('This is a virtual shopping assistant that helps you find electronics based on your past purchases and budget.')

# Main interface
st.title('Virtual Shopping Assistant')

# Chat-like interface
st.write("### Ask your shopping assistant:")

# Display previous result if available
if 'result' in st.session_state:
    st.write("### Previous Result:")
    st.write(st.session_state.result)

# Prompt input field at the bottom of the screen
user_question = st.text_input("Your question:", "Given my previous purchases and a $500 budget, which electronics should I consider buying to stay within the $500 mark?")

if st.button('Send'):
    if user_question:
        # Generate response
        result = chain.invoke(user_question)
        st.session_state.result = result  # Store result in session state

        # Clear previous result
        if 'chat_history' in st.session_state:
            del st.session_state['chat_history']

        # Display assistant's response in chat
        st.write("### Current Result:")
        st.write(result)
