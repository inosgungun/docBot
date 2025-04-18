import os
import streamlit as st
import re

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def format_text_for_display(text):
    """Format text to display properly in Streamlit with line breaks."""
    text = re.sub(r'\s+', ' ', text)

    numbered_pattern = r'(\d+\.\s)'
    if re.search(numbered_pattern, text):
        parts = re.split(r'(\d+\.\s)', text)
        if parts[0].strip() == '':
            parts = parts[1:]  
        
        formatted_text = ''
        for i in range(0, len(parts), 2):
            if i+1 < len(parts):
                point_num = parts[i]
                content = parts[i+1].strip()
                formatted_text += f"{point_num}{content}\n\n"
            else:
                formatted_text += parts[i]
        
        return formatted_text.strip()

    paragraphs = [p.strip() for p in text.split('.') if p.strip()]
    return '. '.join(paragraphs) + '.'


def is_greeting(text):
    greetings = [
        'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 
        'good evening', 'howdy', 'what\'s up', 'whats up', 'hiya'
    ]
    
    text_lower = text.lower().strip().rstrip('!.,?')
    
    for greeting in greetings:
        if greeting in text_lower or text_lower in greeting:
            return True
            
    docbot_patterns = ['docbot', 'doc bot', 'bot', 'assistant']
    for pattern in docbot_patterns:
        for greeting in greetings:
            combined = f"{greeting} {pattern}"
            if combined in text_lower or text_lower in combined:
                return True
    
    return False


def get_greeting_response():
    import random
    responses = [
        "Hello! I'm DocBot, your medical assistant. How can I help you today?",
        "Hi there! I'm ready to help with your medical questions.",
        "Greetings! I'm DocBot, here to provide medical information. What would you like to know?",
        "Hello! How can I assist you with medical information today?",
        "Hi! I'm your medical chatbot assistant. What medical questions do you have?"
    ]
    return random.choice(responses)


def main():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    st.markdown('<h1 class="docbot-title">Ask docBot!</h1>', unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        if is_greeting(prompt):
            greeting_response = get_greeting_response()
            st.chat_message('assistant').markdown(greeting_response)
            st.session_state.messages.append({'role':'assistant', 'content': greeting_response})
        else:
            CUSTOM_PROMPT_TEMPLATE = """
                    Use the pieces of information provided in the context to answer user's question.
                    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                    Dont provide anything out of the given context

                    Context: {context}
                    Question: {question}

                    Start the answer directly. No small talk please.
                    """
            
            HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
            HF_TOKEN=os.environ.get("HF_TOKEN")

            try: 
                vectorstore=get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")

                qa_chain=RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response=qa_chain.invoke({'query':prompt})

                result=response["result"]
                if 'format_text_for_display' in globals():
                    formatted_result = format_text_for_display(result)
                else:
                    formatted_result = result
                
                st.chat_message('assistant').markdown(formatted_result)
                st.session_state.messages.append({'role':'assistant', 'content': formatted_result})

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()