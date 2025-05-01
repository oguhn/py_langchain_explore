import streamlit as st
import os
from langchain.embeddings import CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(page_title="Document GPT", page_icon="📰")

# 캐시 디렉토리 생성
cache_dir = os.path.join(os.getcwd(), ".cache")
files_dir = os.path.join(cache_dir, "files")
embeddings_dir = os.path.join(cache_dir, "embeddings")

os.makedirs(files_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

# API 키 입력 받기
with st.sidebar:
    api_key = st.text_input("OpenAI API 키를 입력하세요", type="password", key="api_key")
    if st.button("확인"):
        if api_key:
            st.session_state.api_key_valid = True
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            st.error("API 키를 입력해주세요.")
    st.markdown("https://github.com/oguhn/py_langchain_explore/blob/main/pages/DocumentGPT.py")

if not api_key or not st.session_state.get("api_key_valid", False):
    st.error("OpenAI API 키를 입력하고 확인 버튼을 눌러주세요.")
    st.stop()

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    model_name="gpt-4o-mini-2024-07-18",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()]
)

@st.cache_resource(show_spinner="Embedding...")
def embed_file(file):
    file_content = file.read()
    file_path = os.path.join(files_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(os.path.join(embeddings_dir, file.name))
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
[
    (
        "system",
        """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
        
        Context: {context}
        """,
    ),
    ("human", "{question}"),
]
)

st.title("Document GPT")
st.markdown("""
## Hi!
##### 파일을 업로드하면,
##### 파일의 내용을 분석하고,
##### 질문에 답변해드리는 기능입니다.
""")

with st.sidebar:
    file = st.file_uploader("Upload a file Only .txt", type=["txt"])


if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                response = chain.invoke(message)
else:
    st.session_state["messages"] = []