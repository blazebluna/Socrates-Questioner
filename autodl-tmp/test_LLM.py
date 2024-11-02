from LLM import ChatGLM4_LLM

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
llm = ChatGLM4_LLM(model_name_or_path="/root/autodl-tmp/ZhipuAI/glm-4-9b-chat", gen_kwargs=gen_kwargs)
print(llm.invoke("你是谁"))

