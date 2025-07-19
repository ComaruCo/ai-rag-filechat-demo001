#向量生成 # utils/embed.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore

def create_vectorstore(chunks, source_name, persist_directory="chroma_db"):
    """
    将文本段 embedding 成向量，并存入 Chroma 向量数据库，带 metadata
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    metadatas = [{"source": source_name} for _ in chunks]
    #为每个文本段创建一个 metadata 字典，内容都是 {"source": source_name}。
    #这些 metadata 会作为向量一起存入 Chroma 向量数据库，方便后续按文件名删除或查询。
    # _ 是占位符，表示我们不关心每个 chunk 的内容，只需要知道它的数量。

    vectorstore = Chroma.from_texts(
        texts=chunks,           # 待处理的文本段
        embedding=embeddings,   # OpenAI embedding 模型
        metadatas=metadatas,    # 每段文本绑定元数据
        persist_directory=persist_directory # 存储路径
    )
    vectorstore.persist()       # 持久化保存到磁盘
    return vectorstore

def delete_vector_by_source(source_name, persist_directory="chroma_db"):
    """
    删除指定 source 的所有向量（如上传的某个文件）
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    vectorstore.delete(where={"source": source_name})   # 条件删除
    vectorstore.persist()    # 保存变更

#从磁盘加载已经存在的 Chroma 向量数据库
def load_vectorstore(persist_directory: str) -> VectorStore:
    """
    用向量数据库路径，加载已存在的 Chroma 向量数据库。返回Chroma 向量数据库对象
    """
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )