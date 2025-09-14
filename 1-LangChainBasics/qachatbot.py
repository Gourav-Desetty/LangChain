import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os

# Page config

st.set_page_config(page_title="Simple langchain chatbot with groq", page_icon="Wit")

#Title
st.title("Simple langchain with groq")
st.markdown("Learn LangChain basics with groq's ultra-fast inference!")

with st.sidebar:
    st.header("Settings")

    # api key
    api_key = st.text_input("Groq API key", type="password", help="Get free api key at groq.com")

    # Model selection
    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0
    )


    # Clear Button
    if st.button("Clear"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


# Initialize LLM
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None

    # Init the groq model
    llm = ChatGroq(api_key=api_key, 
            model=model_name,
            temperature=0.7,
            streaming=True)
    
    # Chat template
    prompt = ChatPromptTemplate([
        ("system", "You are a helpful assistant powered by groq. Answer questions clearly and concisely"),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain

chain = get_chain(api_key, model_name)

if not chain:
    st.warning("Please enter your Groq api key to start chatting")
    st.markdown("[Get your free api key here](https://console.groq.com)")

else:
    # Display the chat message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): 
            st.write(message['content'])

    # chat input
    if question:= st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role":"user", "content":question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response)
                
                message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")