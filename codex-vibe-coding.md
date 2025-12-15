## 参考项目readme 了解项目的能力
- 支持如下模型
    - d3LLM-Dream 实现参考：chat_d3llm_dream.py
    - d3LLM-LLaDA 实现参考：chat_d3llm_llada.py

## 参考 api-example.py 实现如下接口
- @app.get("/v1/models")
- @app.get("/health")
- @app.post("/v1/chat/completions") 支持stream模式

## 请求参数 （API请求参数 对应 模型参数）
- 参考 api-example.py 
- body 参数
    - temperature -> temperature
    - max_tokens -> max_new_tokens

- extra_body 参数
    - block_length -> block_size
    - diffusion_threshold -> threshold
    - diffusion_steps -> steps

## 启动端口自定义 或者 18091

## 编写 api_requirements.txt
- 能够链接项目的requirements.txt
