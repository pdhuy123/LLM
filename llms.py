from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("Token Hugging Face không được tìm thấy trong file .env!")
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

def load_llm(model_name):
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
        model_kwargs={'max_length': 512}
    )
    return llm

template = '''
Use the pieces of information provided in the context to answer user's question.
if you dont know the answer, just say that you dont know, dont try to make up an answer.
Context: {context}
Question: {question}
Start the answer directly.
'''

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt

db_faiss_path = './vectorstores'
embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local(db_faiss_path, embedding, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(model_name=model_name),
    chain_type = 'stuff',
    retriever = db.as_retriever(search_kwarg={'k': 3}),
    chain_type_kwargs = {'prompt': create_prompt(template)}
)

user_query = input("Write the question here: ")
response = qa_chain.invoke(user_query)
print('Result: ', response['result'])