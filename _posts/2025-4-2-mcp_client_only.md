---
layout: post
title: "MCP Client 중심으로 MCP 구조 파악하기 (LLM 빼고)"
author: 김태영
date: 2025-04-02 04:00:00
categories: [MCP, MCPClient, Claude]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-4-2-mcp_client_only_title.jpg
---

# MCP Client 중심으로 MCP 구조 파악하기 (LLM 빼고)

## 목차
1. **MCP란 무엇인가?**
2. **프로젝트 구조 및 준비**
3. **mcp_server_config.json으로 서버 파라미터 관리**
4. **클라이언트 코드 구현**
5. **잠깐 Stdio이란?**
6. **실행 및 예시 출력**
7. **정리**

---

## 1. MCP란 무엇인가?

Model Context Protocol(MCP)는 LLM(예: Claude, GPT)과 다양한 툴(데이터베이스, 파일시스템, 기타 API 등)을 연결하는 **클라이언트-서버 프로토콜**입니다.

- **서버(MCP Server)**: 특정한 기능(‘도구’)을 제공하고, MCP 메시지를 받아 처리함.  
- **클라이언트(MCP Client)**: MCP 서버에 접속하여 필요한 작업을 요청함.

MCP는 **JSON-RPC 2.0** 스타일로 요청(Request)·응답(Response)·알림(Notification)을 주고받기 때문에 확장성과 호환성이 뛰어납니다.

---

## 2. 프로젝트 구조 및 준비

```plaintext
mcp-client/
 ┣ .venv/                   # 가상환경(uv venv)
 ┣ mcp_server_config.json   # MCP 서버 설정 파일
 ┣ client.py                # MCP 클라이언트 코드
 ┗ ...
```

### 가상환경 설치 예시 (uv CLI)

```bash
uv init mcp-client
cd mcp-client
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv add mcp
```

---

## 3. mcp_server_config.json으로 서버 파라미터 관리

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "."
      ]
    }
  }
}
```

- `"command"`: 서버 실행 명령 (예: `python`, `node`, `npx` 등)
- `"args"`: 실행 시 전달할 인자

---

## 4. 클라이언트 코드 구현

```python
import json, os, asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    with open("mcp_server_config.json") as f:
        config = json.load(f)["mcpServers"]["filesystem"]

    server_params = StdioServerParameters(
        command=config["command"],
        args=config["args"],
        env=None
    )

    stack = AsyncExitStack()
    async with stack:
        stdio, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        tools_response = await session.list_tools()
        tool_names = [tool.name for tool in tools_response.tools]
        print("도구:", ", ".join(tool_names))

        allowed_response = await session.call_tool("list_allowed_directories")
        allowed_text = allowed_response.content[0].text

        directories = []
        for line in allowed_text.split('\n'):
            if line.strip() and "Allowed" not in line:
                directories.append(line.strip())

        if not directories:
            directories = ['.']

        print(f"디렉토리: {', '.join(directories)}")

        extensions = ['.txt', '.md', '.py', '.json', '.csv', '.log', '.html', '.css', '.js']

        for directory in directories:
            print(f"\n--- {directory} ---")
            dir_response = await session.call_tool("list_directory", {"path": directory})
            dir_text = dir_response.content[0].text

            text_files = []
            for line in dir_text.split('\n'):
                if line.startswith('[FILE]'):
                    filename = line.replace('[FILE]', '').strip()
                    if any(filename.lower().endswith(ext) for ext in extensions):
                        text_files.append(filename)

            if not text_files:
                print("파일 없음")
                continue

            print(f"{len(text_files)}개: {', '.join(text_files[:3])}" +
                  ("..." if len(text_files) > 3 else ""))

            for filename in text_files[:2]:
                try:
                    file_path = os.path.join(directory, filename)
                    file_response = await session.call_tool("read_file", {"path": file_path})
                    content = file_response.content[0].text
                    lines = content.split('\n')

                    print(f"\n> {filename}:")
                    for i in range(min(3, len(lines))):
                        print(f"  {lines[i]}")
                    if len(lines) > 3:
                        print("  ...")
                except:
                    print(f"오류: {filename} 읽기 실패")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. 잠깐 Stdio이란?

**Stdio Transport**는 MCP에서 클라이언트와 서버가 **표준 입력/출력(stdin/stdout)**으로 통신하는 방식입니다.

### 장점
- 별도 네트워크 구성 없이 **로컬에서 테스트 가능**
- **프로세스 실행만으로 바로 연결**되므로 디버깅이 쉬움

### 한계
- 원격 서버 연결이 필요할 경우 부적합
- 분산 시스템 환경에서는 **SSE + HTTP** 기반 transport가 더 적합

---

## 6. 실행 및 예시 출력

```bash
uv run client.py
# 또는
python client.py
```

### 예시 출력
```
도구: list_allowed_directories, list_directory, read_file
디렉토리: ., ./data

--- . ---
3개: README.md, test.txt, example.py...

> README.md:
  # Project Readme
  This is an example README file.
  For demonstration.

> test.txt:
  test row1
  test row2
  test row3
```

---

## 7. 정리

이 글은 LLM 없이도 MCP의 구조를 이해할 수 있도록 **MCP Client만 떼어 구현**하고 테스트해본 사례입니다.

- `list_tools`, `call_tool` 같은 **MCP 핵심 호출 흐름**을 직접 확인 가능
- Stdio는 **빠른 테스트**와 **로컬 실습**에 적합
- 추후에는 **Claude, GPT 등 LLM**을 결합하여 자연어 인터페이스로 확장 가능

---

## 📎 참고 링크

- [MCP Quickstart for Clients](https://modelcontextprotocol.io/quickstart/client)
