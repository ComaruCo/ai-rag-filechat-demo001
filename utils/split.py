#文本切分处理
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(content: str, chunk_size=300, chunk_overlap=50):
    """将大段文本切分为多个小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(content)
    return chunks
