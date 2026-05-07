import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate


```python OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") ``` 
with st.sidebar:
    st.title("MY PDF")
    file=st.file_uploader("Upload notes PDF and start asking questions",type="pdf")

if file is not None:
    my_pdf=PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text+=page.extract_text()
        st.write(text)
    splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50,length_function=len)
    chunks=splitter.split_text(text)
    st.write(chunks)
    embedding=OpenAIEmbeddings(api_key="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")
    vector_store=FAISS.from_texts(chunks,embedding)
    userquery=st.text_input("Type your query here")
    if userquery:
        matching_chunks=vector_store.similarity_search(userquery)
        llm=ChatOpenAI(api_key="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",max_tokens=300,temperature=1,model="gpt-3.5-turbo")
        customized_prompt=ChatPromptTemplate.from_template("""You are my assistant tutor.Answer the question based on the following content and if you did not get the context simply say "I don't know jenny:
        {context}
        Question:{input}
        """)
        chain=create_stuff_documents_chain(llm,customized_prompt)
        output=chain.invoke({"input":userquery,"input_documents":matching_chunks})
        st.write(output)


