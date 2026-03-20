# Report Scratch

AI 驱动的研究前置报告生成工具。输入研究课题，自动通过网络检索、多轮 LLM 推理，生成结构化的研究前置报告（支持 IEEE 等格式）。

## 功能

- 多 LLM 配置管理（支持 OpenAI、Gemini 等兼容 OpenAI 协议的服务）
- 自动网络检索 + 多轮推理生成研究报告
- 报告支持 Markdown 预览与导出
- 对话式交互，支持追问与重新生成
- 后台任务异步生成，SSE 实时推送进度

## 技术栈

- **后端**：Python / FastAPI / SQLite
- **前端**：React / TypeScript / Vite / Tailwind CSS

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+

### 启动

```bash
bash start.sh
```

脚本会自动完成：
1. 安装 Python 依赖
2. 安装前端依赖
3. 启动后端（`http://localhost:8000`）
4. 启动前端（`http://localhost:5173`）

### 手动启动

**后端**

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**前端**

```bash
cd webui
npm install
npm run dev
```

## 配置 LLM

1. 打开 `http://localhost:5173`
2. 点击右上角「配置」
3. 添加 LLM 配置：
   - **名称**：自定义标识
   - **Provider**：OpenAI / Gemini 等
   - **API Key**：对应服务的密钥
   - **Base URL**：自定义接口地址（使用官方服务可留空）
   - **Model**：模型名称，如 `gpt-4o`、`gemini-2.5-flash`
   - **报告语言**：默认 `zh-CN`

## 环境变量（可选）

| 变量 | 说明 | 默认值 |
|---|---|---|
| `OPENAI_MODEL` | 默认模型 | `gemini-2.5-flash` |
| `OPENAI_BASE_URL` | 默认 Base URL | Google Gemini 接口 |

## 项目结构

```
.
├── api/
│   ├── main.py            # FastAPI 路由与数据库
│   ├── research_prereq.py # 报告生成 Agent
│   └── llm_logging.py     # LLM 调用日志
├── webui/                 # React 前端
├── reports/               # 生成的报告文件（gitignore）
├── reports.db             # SQLite 数据库（gitignore）
├── requirements.txt
└── start.sh               # 一键启动脚本
```

## API 文档

启动后访问 `http://localhost:8000/docs` 查看完整 OpenAPI 文档。
