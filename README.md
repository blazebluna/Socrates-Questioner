# Socrates-Questioner
Data Mining Assignment
## Step1：环境配置与模型下载
- 使用chatglm-4-9b-chat作为预训练语言模型，下载langchain，开源词向量模型sentence-transformers
## Step2：构建向量检索库
- 选择数据挖掘专业课讲义作为语料库来源，将pdf讲义转为markdown，便于大模型理解
- 使用FileLoader对象加载得到纯文本列表，依次对文本分块、向量化后，构建Chroma向量数据库
## Step3：将ChatGLM接入LangChain
- 继承并重写LLM类的构造函数与_call函数
## Step4：构建检索问答链
- 导入向量数据库，实例化自定义ChatGLM
- 人工设计一个优秀的Prompt，调用检索问答链
## Step5：部署Web Demo
- 基于Gradio框架将检索问答链对象部署到Web网页
