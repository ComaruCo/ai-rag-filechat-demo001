#主程序（Python，网页入口）
# app.py
import streamlit as st #建网页用的
from dotenv import load_dotenv #加载key用的
import os

from utils.split import split_text #文本切分处理
from utils.embed import create_vectorstore #文本向量化，生成向量库
from utils.embed import delete_vector_by_source #删除向量库信息
from utils.embed import load_vectorstore #加载已有向量数据库
from utils.qa import get_qa_chain #用户交互处理

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 本地开发 加载.env文件中的API Key
load_dotenv()
# 部署时读取 secrets（优先使用 secrets 中的 key）
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

#设置浏览器标签页文字 页面布局为宽屏模式
st.set_page_config(page_title="文件问答 Demo", layout="wide")
#页面顶部显示大标题
st.title("文件问答 Demo001-1")

# 1 画面加载时候就有简易对话框
# 2 对话框可以进行彼此单次100字符以内的问答
# 3 画面要保留对话过程
# 4 对话框里面追加按钮
#   4-1 发送按钮 清除按钮 改成多行入力框按钮 可进行彼此300字符以内的问答
# 5 对话框旁边追加按钮
#   5-1 清除对话历史按钮
# 6 对话框下面增加几个功能按钮
#   6-0 查看已有文档 点击变成 收起文档列表
#   6-1 文件内容预览 点击变成 关闭文件预览
# 7 可上传多个文档
# 8 上传的文档前面加个按钮 
#   8-1 存入向量数据库
#   8-2 从向量数据库删除
# 9 每次问答下面加一行小字 显示本次回答/提问使用多少token
# 10 回答框后面追加一个按钮 
#   10-1 显示回答引用原文 
#   10-2 显示回答引用原文件的行数
# 11 可以跨多文件回答问题 比如问题答案在多个文件里都有涉猎
# 12 上传的文档可以是txt,pdf,excel 
# 13 chunk切割与成本调整
# 14 部署上去，外人可访问
# 15 多用户使用模式
# 16 用户权限设定
# 17 用户用量设定

# 上传文件
uploaded_file = st.file_uploader("请上传一个 txt 文本文件", type=["txt"])

if uploaded_file:

    # --0. 上传文本
    # 读取文本内容
    content = uploaded_file.read().decode("utf-8")
    filename = uploaded_file.name
    # 显示成功信息
    st.success(f"✅ 文件已上传：{filename}")
    # 预览前100字符
    #st.text_area("文件内容预览（前100字符）", content[:100], height=100)
    st.markdown("#### 文件内容预览（前100字符）")
    st.markdown(f"<div style='font-size:12px; background-color:#f8f9fa; padding:10px;'>{content[:100]}</div>", unsafe_allow_html=True)

    # --1. 切分文本（分段）
    chunks = split_text(content)
    st.write(f"成功将文本分为 {len(chunks)} 段。")
    #st.text_area("示例段落（第1段）", chunks[0], height=150)
    st.markdown("#### 示例段落（第1段）")
    st.markdown(f"<div style='font-size:12px; background-color:#f0f0f0; padding:10px;'>{chunks[0]}</div>", unsafe_allow_html=True)

    # --2. 文本 embedding，先判断此文件是否已有向量库 生成向量库
    
    #将向量数据库的保存路径设定为 "chroma_db/文件名"
    persist_path = os.path.join("chroma_db", filename)
    if os.path.exists(persist_path):
        #调用加载已有数据库方法
        vectorstore = load_vectorstore(persist_path)
        st.info("已加载已有向量数据库")
    else:
        #页面中显示一个“加载中”的旋转动画
        with st.spinner("🔍 正在生成向量数据库，请稍候..."):
            #调用生成向量数据库方法
            vectorstore = create_vectorstore(chunks, source_name=filename, persist_directory=persist_path)
        st.success("✅ 向量数据库构建成功！")

    # --3. 用户问答处理
    # 用户输入问题
    question = st.text_input("💬 输入你的问题：")

    if question:
        # 创建问答链
        qa_chain = get_qa_chain(vectorstore)        
        # 执行问答
        with st.spinner("🤖 正在思考回答..."):
            result = qa_chain({"query": question})
            answer = result["result"]
        # 显示回答
        st.success("回答如下：")
        st.write(answer)
        # 显示引用的文档段落
        st.markdown("#### 引用内容：")
        #enumerate在遍历可迭代对象（如列表、元组）时，同时获取元素的索引和值。
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**段落 {i+1}:**")
            st.markdown(f"<div style='font-size:13px; background-color:#f9f9f9; padding:10px;'>{doc.page_content}</div>", unsafe_allow_html=True)
            st.caption(f"来源: {doc.metadata.get('source', '未知')}")

    # --4. 用户删除上传的文件信息处理
    if st.button(f"🗑️ 删除向量：{filename}"):
        delete_vector_by_source(filename)
        st.warning(f"已删除与 {filename} 相关的向量数据！")

