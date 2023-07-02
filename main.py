import streamlit as st
import openai
import os
import threading
import queue
import time

# Import necessary packages
from llama_hub.file.base import SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex, StorageContext, load_index_from_storage
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from streamlit_chat import message
from PIL import Image


response = queue.Queue()

image = Image.open('./images/wellness-logo.png')

os.environ['OPENAI_API_KEY'] = st.secrets['api_secret']
openai.api_key = os.environ['OPENAI_API_KEY']

# Setting page title and header
st.set_page_config(page_title="WellnessAgent", page_icon=":robot:")

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = [
        "Hello! how can I assist with you today?"]
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hello, Wellness Agent!"]
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "Hello! how can I assist with you today?"}
    ]
if 'index' not in st.session_state:
    st.session_state['index'] = 'LLama'
if 'words' not in st.session_state:
    st.session_state['words'] = ''

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
# st.sidebar.title("Ernesto Assistant")

with st.sidebar:
    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        pass
    with col2:
        st.image(image=image, caption="Medical Assistant")
    with col3:
        pass
    col4, col5, col6 = st.columns([2, 5, 2])
    with col4:
        pass
    with col5:
        clear_button = st.button("Clear Conversation", key="clear")
    with col6:
        pass
    st.text("History:")
    col7, col8, col9 = st.columns([2, 5, 2])
    with col7:
        pass
    with col8:
        pass       
    with col9:
        pass
    # model_name = st.sidebar.radio(
    #     "Choose a model:", ("GPT-3.5", "GPT-4 (Coming Soon)"))

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []

def update_history():
        with st.sidebar, col8:
            for i, past in enumerate(st.session_state['past']):
                st.session_state['words'] = str(i+1)+". "+st.session_state['words']
                i = st.empty()
                past = past+"\n"
                for char in past:
                    st.session_state['words']+=char
                    i.write(st.session_state['words'])
                    time.sleep(.01)
                st.session_state['words']=""

def update_messages(message):
    st.session_state['messages'].append(
        {"role": "assistant", "content": message})


def load_index(storage_context):
# load index
    index = load_index_from_storage(storage_context, index_id="vector_index")
    query_engine = index.as_query_engine(response_mode="compact")
    return query_engine


# check storage directory exist
def check_storage_exist():
    files = os.listdir()
    if "storage" not in files:
        loader = SimpleDirectoryReader(
            './data', recursive=True, exclude_hidden=True)
        documents = loader.load_data()
        index = GPTVectorStoreIndex.from_documents(documents)
        index.set_index_id("vector_index")
        index.storage_context.persist('storage')
         # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir='storage')
    return storage_context

# langchain tool setup
def langchain_tool(query_engine):
    tools = [
    Tool(
        name="Local Directory Index",
        func=lambda q: query_engine.query(q),
        description=f"Useful when you want answer questions about the files in your local directory.",
    ),
    ]
    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_chain = initialize_agent(
        tools, llm, agent="zero-shot-react-description", memory=memory
    )
    return agent_chain

# generate llama response
def llama_generate_response(prompt):
    print("Using llama to generate response...")
    storage_context = check_storage_exist()
    query_engine = load_index(storage_context)
    message = query_engine.query(prompt)
    # update_messages(message)
    response.put(message)

# generate langchain response
def langchain_generate_response(prompt):
    print("Using langchain to generate response...")
    storage_context = check_storage_exist()
    query_engine = load_index(storage_context)
    agent_chain = langchain_tool(query_engine)
    message = agent_chain.run(input=prompt)
    # update_messages(message)
    response.put(message)

def fetch_response(index):
    fetching = True
    st.session_state['past'].append(user_input)
    if index == "LLama":
        thread = threading.Thread(target=llama_generate_response(user_input))
    elif index == "Langchain":
        thread = threading.Thread(target=langchain_generate_response(user_input))
    thread.start()
    while fetching:
        try:
            output = response.get()
            st.session_state['generated'].append(output)
            update_messages(output)
            if output:
                fetching = False
        except Exception as e:
            print("Error: " + str(e))
            thread.join()

    thread.join()


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        index_radio = st.radio("Choose the indexer:",('LLama','Langchain'), key=st.session_state['index'], horizontal=True)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        if index_radio == 'LLama':
          fetch_response(index_radio)
        elif index_radio == 'Langchain':
          fetch_response(index_radio)
        

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(str(st.session_state["generated"][i]).strip(), key=str(i))
        update_history()