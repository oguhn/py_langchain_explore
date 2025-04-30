import streamlit as st

st.title("Document GPT")

if "messages" not in st.session_state:
    st.session_state.messages = []

def send_message(message, role, save):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"role": role, "message": message})

for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)

message = st.chat_input("Ask a question about the document")

if message:
    send_message(message, "human", save=True)
    send_message(f"{message}", "ai", save=True)

# with st.sidebar:
#     st.write(st.session_state)