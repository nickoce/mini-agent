# Agent MVP

一个最小可上线的 Agent MVP：

```text
用户输入问题
-> Intent Detection 判断是否需要搜索
-> 需要搜索时调用 Tavily Search
-> 将搜索结果交给 LLM 总结
-> 显示 Agent Trace 和最终答案
```

不使用 LangChain / CrewAI 等 Agent 框架。

## 功能

- Streamlit Web UI
- Intent Detection
- Tavily Search Tool
- LLM 总结
- 可见 Agent Trace 日志
- Run Metrics：Token、Latency、Search Calls

## 项目结构

```text
.
├── app.py
├── agent.py
├── requirements.txt
├── .env.example
├── tools/
│   └── search.py
└── utils/
    └── logger.py
```

## 环境变量

复制 `.env.example` 为 `.env`，然后填写真实 Key。

百炼模式：

```text
DASHSCOPE_API_KEY=your_dashscope_api_key_here
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BAILIAN_MODEL=qwen-plus
TAVILY_API_KEY=your_tavily_api_key_here
```

OpenAI 兼容模式：

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4.1-mini
TAVILY_API_KEY=your_tavily_api_key_here
```

`.env` 已被 `.gitignore` 忽略，不要提交到 GitHub。

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

## 文件用途

| 文件 | 用途 |
|---|---|
| `app.py` | Streamlit Web UI 入口 |
| `agent.py` | Agent 主逻辑：意图判断、搜索、总结、指标统计 |
| `tools/search.py` | Tavily Search API 工具 |
| `utils/logger.py` | Agent Trace 日志工具 |
| `requirements.txt` | Python 依赖 |
