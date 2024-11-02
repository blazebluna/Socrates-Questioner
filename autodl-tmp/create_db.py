import os
from tqdm import tqdm
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_files(dir_path):
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".md"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
    return file_list

def get_text(dir_path):
    file_list = get_files(dir_path)
    docs = []
    for one_file in tqdm(file_list):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        else:
            continue
        docs.extend(loader.load())
    return docs

tar_dir = [
    "/root/autodl-tmp/data_mining"
]

docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

persist_directory = 'data_base/vector_db/chroma'
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

vectordb.persist()

