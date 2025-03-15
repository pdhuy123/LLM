from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

data_path = './data'
db_path = './vectorstores'
def load_pdf_files(path: str):
    loader = DirectoryLoader(
        path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    return docs



def create_chunks(document: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=512
    )
    chunks = text_splitter.split_documents(document)
    return chunks


def vectordb(data_path, db_path):
    docs = load_pdf_files(data_path)
    chunks = create_chunks(docs)

    embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(db_path)

vectordb(data_path, db_path)