import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API Wrappers
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=100)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

search = DuckDuckGoSearchRun()

st.title("Langchain - Chat with Search")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm Jarvis who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if api_key and (prompt := st.chat_input(placeholder="Ask me anything...")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
        tools = [search, arxiv, wiki]
        search_agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

    except Exception as e:
        st.error(f"An error occurred: {e}")
