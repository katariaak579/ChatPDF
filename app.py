import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template ,user_template



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()

    return text

def get_chunks_text(raw_textis):
    chunks = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len 
    )
    text = chunks.split_text(raw_textis)
    return text

def get_embeddings(textis):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=textis,embedding=embeddings)
    return vectorstore

def get_conversation_memory(vectorstoreis):
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    llm = ChatOpenAI()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstoreis.as_retriever(),
    )
    return conversation_chain

def response_ques(questionis):
    res = st.session_state.conversation_chain({'question':questionis})
    st.session_state.chat_his = res['chat_history']
    for i, mes in enumerate(reversed(st.session_state.chat_his)):
        if i % 2 != 0:
            st.write(user_template.replace("{{MSG}}", mes.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", mes.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain=None

    if "chat_his" not in st.session_state:
        st.session_state.chat_his=None

    st.set_page_config(page_title="Chat with PDF",page_icon=":books:")
    st.header("Chat with PDF")
    usr_ques = st.text_input("Enter the prompt here")
    if usr_ques:
        response_ques(usr_ques)
    st.write(css,unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader("Upload your file and click 'Process",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks_text(raw_text)
                embeds = get_embeddings(text_chunks)
                st.session_state.conversation_chain = get_conversation_memory(embeds)

    # st.write(user_template.replace("{{MSG}}","Hello Robot"),unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}","Hello Hooman"),unsafe_allow_html=True)


if __name__ == "__main__":
    main()