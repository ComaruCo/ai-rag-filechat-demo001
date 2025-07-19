#问答逻辑模块
# utils/qa.py

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def get_qa_chain(vectorstore, model_name="gpt-3.5-turbo"):
    """
    构建问答链（基于向量数据库 + ChatGPT 回答）
    """
    #初始化 LLM
    llm = ChatOpenAI(model_name=model_name)
    #从向量库中检索最相关的 3 段文本（k=3）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    #构建问答链（LLM + 检索器）
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True # 返回源文档内容，便于调试或展示
    )
    return qa_chain
