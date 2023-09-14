---
layout: post
title: "PDF 파일기반 질의응답 챗봇 (랭체인, 그라디오, ChatGPT)"
author: 김태영
date: 2023-8-21 00:00:00
categories: ai
comments: true
---

```python
!pip install openai # openai 라이브러리를 설치합니다.
!pip install langchain # 랭체인 라이브러리를 설치합니다.
```

```
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting openai
  Downloading openai-0.27.7-py3-none-any.whl (71 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m72.0/72.0 kB[0m [31m5.1 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: requests>=2.20 in /usr/local/lib/python3.10/dist-packages (from openai) (2.27.1)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from openai) (4.65.0)
Collecting aiohttp (from openai)
  Downloading aiohttp-3.8.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.0 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m46.3 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2022.12.7)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2.0.12)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (3.4)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (23.1.0)
Collecting multidict<7.0,>=4.5 (from aiohttp->openai)
  Downloading multidict-6.0.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (114 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m114.5/114.5 kB[0m [31m7.8 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting async-timeout<5.0,>=4.0.0a3 (from aiohttp->openai)
  Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
Collecting yarl<2.0,>=1.0 (from aiohttp->openai)
  Downloading yarl-1.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (268 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m268.8/268.8 kB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting frozenlist>=1.1.1 (from aiohttp->openai)
  Downloading frozenlist-1.3.3-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (149 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m149.6/149.6 kB[0m [31m15.2 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting aiosignal>=1.1.2 (from aiohttp->openai)
  Downloading aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
Installing collected packages: multidict, frozenlist, async-timeout, yarl, aiosignal, aiohttp, openai
Successfully installed aiohttp-3.8.4 aiosignal-1.3.1 async-timeout-4.0.2 frozenlist-1.3.3 multidict-6.0.4 openai-0.27.7 yarl-1.9.2
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting langchain
  Downloading langchain-0.0.190-py3-none-any.whl (983 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m983.2/983.2 kB[0m [31m34.7 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: PyYAML>=5.4.1 in /usr/local/lib/python3.10/dist-packages (from langchain) (6.0)
Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.0.10)
Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /usr/local/lib/python3.10/dist-packages (from langchain) (3.8.4)
Requirement already satisfied: async-timeout<5.0.0,>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (4.0.2)
Collecting dataclasses-json<0.6.0,>=0.5.7 (from langchain)
  Downloading dataclasses_json-0.5.7-py3-none-any.whl (25 kB)
Requirement already satisfied: numexpr<3.0.0,>=2.8.4 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.8.4)
Requirement already satisfied: numpy<2,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain) (1.22.4)
Collecting openapi-schema-pydantic<2.0,>=1.2 (from langchain)
  Downloading openapi_schema_pydantic-1.2.4-py3-none-any.whl (90 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m90.0/90.0 kB[0m [31m10.4 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: pydantic<2,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain) (1.10.7)
Requirement already satisfied: requests<3,>=2 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.27.1)
Requirement already satisfied: tenacity<9.0.0,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (8.2.2)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (23.1.0)
Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (2.0.12)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.0.4)
Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.9.2)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.3)
Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.1)
Collecting marshmallow<4.0.0,>=3.3.0 (from dataclasses-json<0.6.0,>=0.5.7->langchain)
  Downloading marshmallow-3.19.0-py3-none-any.whl (49 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.1/49.1 kB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting marshmallow-enum<2.0.0,>=1.5.1 (from dataclasses-json<0.6.0,>=0.5.7->langchain)
  Downloading marshmallow_enum-1.5.1-py2.py3-none-any.whl (4.2 kB)
Collecting typing-inspect>=0.4.0 (from dataclasses-json<0.6.0,>=0.5.7->langchain)
  Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)
Requirement already satisfied: typing-extensions>=4.2.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<2,>=1->langchain) (4.5.0)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain) (2022.12.7)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain) (3.4)
Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.10/dist-packages (from SQLAlchemy<3,>=1.4->langchain) (2.0.2)
Requirement already satisfied: packaging>=17.0 in /usr/local/lib/python3.10/dist-packages (from marshmallow<4.0.0,>=3.3.0->dataclasses-json<0.6.0,>=0.5.7->langchain) (23.1)
Collecting mypy-extensions>=0.3.0 (from typing-inspect>=0.4.0->dataclasses-json<0.6.0,>=0.5.7->langchain)
  Downloading mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)
Installing collected packages: mypy-extensions, marshmallow, typing-inspect, openapi-schema-pydantic, marshmallow-enum, dataclasses-json, langchain
Successfully installed dataclasses-json-0.5.7 langchain-0.0.190 marshmallow-3.19.0 marshmallow-enum-1.5.1 mypy-extensions-1.0.0 openapi-schema-pydantic-1.2.4 typing-inspect-0.9.0

```

```python
import os

os.environ["OPENAI_API_KEY"] = "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" # 환경변수에 OPENAI_API_KEY를 설정합니다.
```

```python
#from langchain.chat_models import ChatOpenAI
```

```python
!pip install pypdf # pdf 로딩용
!pip install chromadb # 벡터스토어
!pip install tiktoken # 토큰 계산용
```

```
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting pypdf
  Downloading pypdf-3.9.1-py3-none-any.whl (249 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m249.3/249.3 kB[0m [31m10.0 MB/s[0m eta [36m0:00:00[0m
[?25hInstalling collected packages: pypdf
Successfully installed pypdf-3.9.1
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting chromadb
  Downloading chromadb-0.3.25-py3-none-any.whl (86 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m86.6/86.6 kB[0m [31m4.5 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: pandas>=1.3 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.5.3)
Collecting requests>=2.28 (from chromadb)
  Downloading requests-2.31.0-py3-none-any.whl (62 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.6/62.6 kB[0m [31m4.1 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: pydantic>=1.9 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.10.7)
Collecting hnswlib>=0.7 (from chromadb)
  Downloading hnswlib-0.7.0.tar.gz (33 kB)
  Installing build dependencies ... [?25l[?25hdone
  Getting requirements to build wheel ... [?25l[?25hdone
  Preparing metadata (pyproject.toml) ... [?25l[?25hdone
Collecting clickhouse-connect>=0.5.7 (from chromadb)
  Downloading clickhouse_connect-0.5.25-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (922 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m922.7/922.7 kB[0m [31m44.7 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: duckdb>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from chromadb) (0.7.1)
Collecting fastapi>=0.85.1 (from chromadb)
  Downloading fastapi-0.96.0-py3-none-any.whl (57 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.1/57.1 kB[0m [31m6.6 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting uvicorn[standard]>=0.18.3 (from chromadb)
  Downloading uvicorn-0.22.0-py3-none-any.whl (58 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m7.2 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: numpy>=1.21.6 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.22.4)
Collecting posthog>=2.4.0 (from chromadb)
  Downloading posthog-3.0.1-py2.py3-none-any.whl (37 kB)
Collecting onnxruntime>=1.14.1 (from chromadb)
  Downloading onnxruntime-1.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.9 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.9/5.9 MB[0m [31m88.1 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting tokenizers>=0.13.2 (from chromadb)
  Downloading tokenizers-0.13.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.8 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.8/7.8 MB[0m [31m105.6 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: tqdm>=4.65.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (4.65.0)
Requirement already satisfied: typing-extensions>=4.5.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (4.5.0)
Collecting overrides>=7.3.1 (from chromadb)
  Downloading overrides-7.3.1-py3-none-any.whl (17 kB)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from clickhouse-connect>=0.5.7->chromadb) (2022.12.7)
Requirement already satisfied: urllib3>=1.26 in /usr/local/lib/python3.10/dist-packages (from clickhouse-connect>=0.5.7->chromadb) (1.26.15)
Requirement already satisfied: pytz in /usr/local/lib/python3.10/dist-packages (from clickhouse-connect>=0.5.7->chromadb) (2022.7.1)
Collecting zstandard (from clickhouse-connect>=0.5.7->chromadb)
  Downloading zstandard-0.21.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.7 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.7/2.7 MB[0m [31m82.3 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting lz4 (from clickhouse-connect>=0.5.7->chromadb)
  Downloading lz4-4.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m71.9 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting starlette<0.28.0,>=0.27.0 (from fastapi>=0.85.1->chromadb)
  Downloading starlette-0.27.0-py3-none-any.whl (66 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.0/67.0 kB[0m [31m8.2 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting coloredlogs (from onnxruntime>=1.14.1->chromadb)
  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46.0/46.0 kB[0m [31m6.1 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: flatbuffers in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (23.3.3)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (23.1)
Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (3.20.3)
Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (1.11.1)
Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.3->chromadb) (2.8.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from posthog>=2.4.0->chromadb) (1.16.0)
Collecting monotonic>=1.5 (from posthog>=2.4.0->chromadb)
  Downloading monotonic-1.6-py2.py3-none-any.whl (8.2 kB)
Collecting backoff>=1.10.0 (from posthog>=2.4.0->chromadb)
  Downloading backoff-2.2.1-py3-none-any.whl (15 kB)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.28->chromadb) (2.0.12)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.28->chromadb) (3.4)
Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.10/dist-packages (from uvicorn[standard]>=0.18.3->chromadb) (8.1.3)
Collecting h11>=0.8 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading h11-0.14.0-py3-none-any.whl (58 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m6.5 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting httptools>=0.5.0 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading httptools-0.5.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (414 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m414.1/414.1 kB[0m [31m41.7 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting python-dotenv>=0.13 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading python_dotenv-1.0.0-py3-none-any.whl (19 kB)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from uvicorn[standard]>=0.18.3->chromadb) (6.0)
Collecting uvloop!=0.15.0,!=0.15.1,>=0.14.0 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading uvloop-0.17.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.1 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.1/4.1 MB[0m [31m100.5 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting watchfiles>=0.13 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading watchfiles-0.19.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m71.9 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting websockets>=10.4 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading websockets-11.0.3-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (129 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m129.9/129.9 kB[0m [31m14.4 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: anyio<5,>=3.4.0 in /usr/local/lib/python3.10/dist-packages (from starlette<0.28.0,>=0.27.0->fastapi>=0.85.1->chromadb) (3.6.2)
Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.14.1->chromadb)
  Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m86.8/86.8 kB[0m [31m12.5 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->onnxruntime>=1.14.1->chromadb) (1.3.0)
Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.4.0->starlette<0.28.0,>=0.27.0->fastapi>=0.85.1->chromadb) (1.3.0)
Building wheels for collected packages: hnswlib
  Building wheel for hnswlib (pyproject.toml) ... [?25l[?25hdone
  Created wheel for hnswlib: filename=hnswlib-0.7.0-cp310-cp310-linux_x86_64.whl size=2119868 sha256=885d0538eb46ba5515af232648dc17f3aa49ba5fff36d8d3b77b110b7a1f88a1
  Stored in directory: /root/.cache/pip/wheels/8a/ae/ec/235a682e0041fbaeee389843670581ec6c66872db856dfa9a4
Successfully built hnswlib
Installing collected packages: tokenizers, monotonic, zstandard, websockets, uvloop, requests, python-dotenv, overrides, lz4, humanfriendly, httptools, hnswlib, h11, backoff, watchfiles, uvicorn, starlette, posthog, coloredlogs, clickhouse-connect, onnxruntime, fastapi, chromadb
  Attempting uninstall: requests
    Found existing installation: requests 2.27.1
    Uninstalling requests-2.27.1:
      Successfully uninstalled requests-2.27.1
[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
google-colab 1.0.0 requires requests==2.27.1, but you have requests 2.31.0 which is incompatible.[0m[31m
[0mSuccessfully installed backoff-2.2.1 chromadb-0.3.25 clickhouse-connect-0.5.25 coloredlogs-15.0.1 fastapi-0.96.0 h11-0.14.0 hnswlib-0.7.0 httptools-0.5.0 humanfriendly-10.0 lz4-4.3.2 monotonic-1.6 onnxruntime-1.15.0 overrides-7.3.1 posthog-3.0.1 python-dotenv-1.0.0 requests-2.31.0 starlette-0.27.0 tokenizers-0.13.3 uvicorn-0.22.0 uvloop-0.17.0 watchfiles-0.19.0 websockets-11.0.3 zstandard-0.21.0
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting tiktoken
  Downloading tiktoken-0.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m46.5 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2022.10.31)
Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2.31.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2.0.12)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2022.12.7)
Installing collected packages: tiktoken
Successfully installed tiktoken-0.4.0

```

```python
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
```

```python
loader = PyPDFLoader("인공지능팩토리_복리후생_230516.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```

```python
texts
```

```
[Document(page_content='복리 후생\n1☕\n복리 후생\n💡지속적으로  복리후생  정책이  변경되고  있으며 , 시행  혹은  변경  시마다  메일로  공\n지되는  내용은  해당  페이지를  통해서도  확인할  수  있으니  메일을  놓치거나  확인\n이 필요한  경우에는  반드시  본  페이지를  확인해주세요 .\n건강검진\n직원들의  건강  증진을  위하여  회사에서  건강검진  비용을  지원합니다 .\n연도 규정\n2023• 지원  금액 : 20 만원  한도 , 초과분은  개인  지출  • 지원  대상 : 입사일로부터  6 개월이\n지난 자  • ‘23 년  7 월  이후  입사자는  당해년도  대상에서  제외  • 지원  만료  시기 : ‘23\n년 12월  31일까지  • 결제  방법 : 개별  소유한  법인카드로  20 만원까지  결제  후 , 나\n머지 추가  비용에  대해서는  개인  지불  • 휴가  처리 : 오전  혹은  오후  반차  (0.5 일 )\n공가 처리 , 어시  기안  및  구글  캘린더  등록  • 기타 : 체질개선 ( 당뇨 , 비만 , 고혈압\n등) 프로그램  이용도  가능  • 시행  : ‘23 년  4 월  12 일부터\n2022• 지원  금액 : 20 만원  한도 , 초과분은  개인  지출  • 지원  대상 : 입사일로부터  6 개월이\n지난 자  • ‘22 년  7 월  이후  입사자는  당해년도  대상에서  제외  • 지원  만료  시기 : ‘22\n년 12월  31일까지  • 결제  방법 : 개별  소유한  법인카드로  20 만원까지  결제  후 , 나\n머지 추가  비용에  대해서는  개인  지불  • 휴가  처리 : 오전  혹은  오후  반차  (0.5 일 )\n공가 처리 , 구글  캘린더  등록  및  슬랙  # 휴가  채널  작성  • 공가  휴가  신청은  전자결\n재 어시를  통해서  신청  (’22 년  6 월  1 일  이후 ) • 기타 : 체질개선 ( 당뇨 , 비만 , 고혈압\n등) 프로그램  이용도  가능\n2022년  기안서  참고\nhttps://s3-us-west-2.amazonaws.com/secure.noti\non-static.com/45a30f62-1ef1-44fd-ad3e-cf8ede8\n837e9/20220419-22 년 _ 건강검진비용 _ 지원 _ 및 _ 건\n강검진 _실시 _ 품의서 ( 승인본 ).pdf', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 0}),
 Document(page_content='복리 후생\n2근로자휴가지원사업\n정부와  기업이  함께  근로자의  국내여행  경비 ( 휴가비 ) 를  지원합니다 .\n연도 규정\n2022• 분담금액  : 정부  10 만원  + 기업  10 만원  + 근로자  20 만원  → 근로자  1 인당  적립\n금 40만원  조성  • 동  사업은  참여근로자에  한해  한국관광공사에  일괄  신청하여  휴\n가비를  지원하는  프로그램\n2022년  기안서  참고\nhttps://s3-us-west-2.amazonaws.com/secure.noti\non-static.com/c36f8488-9883-4540-9e78-541e66\n91e8ac/20220323-22 년 _ 근로자 _ 휴가지원사업 _ 신\n청_품의서 _ 서명본 .pdf\n회식비\n임직원간  업무효율성  제고와  소통을  위하여  사용되는  ‘ 회식비 ’ 를  지원합니다 .\n연도 규정\n2023• 지원금액  : 인당  분기별  3 만원  • 회식비  사용은  2 인  이상 ( 만 ) 충족되면  사용  가능\n하며 부서 ( 팀 ) 내  혹은  다른  부서 ( 팀 ) 간  회식에도  사용  가능함 . • 회식비  지급대상\n은 휴직  및  정직중인  자를  제외한  모든  직원 . • 당월에  사용하지  않은  회식비는  월\n단위로  자동  소멸되며  누적하여  사용할  수  없음 . • 법인카드  사용신청서  작성  시\n회식 참석자  명단  기명하여  제출 . • 결제방식  : 개인별  보유  중인  법인카드로  일괄\n결제. 인당  분기별  3 만원  초과분은  개인명의로  지출  • 지원기한  : ‘23 년  12 월  31 일\n까지 운영 . • 시행  : ‘23 년  2 사분기부터\n2022• 지원금액  : 인당  월  2 만원  • 2 인이상  소통을  위한  회식비  사용  시  본인  월  2 만원\n사용 한도내에서  지원 . • 회식비  사용은  2 인  이상 ( 만 ) 충족되면  사용  가능하며  부\n서(팀 ) 내  혹은  다른  부서 ( 팀 ) 간  회식에도  사용  가능함 . • 회식비  지급대상은  휴직\n및 정직중인  자를  제외한  모든  직원 . • 당월에  사용하지  않은  회식비는  월  단위로\n자동 소멸되며  누적하여  사용할  수  없음 . • 법인카드  사용신청서  작성  시  회식  참\n석자 명단  기명하여  제출 . • 결제방식  : 개인별  보유  중인  법인카드로  일괄  결제 .\n인당 월  2 만원  초과분은  개인명의로  지출 . • 지원기한  : ‘22 년  12 월  31 일까지  운\n영. • 시행  : ‘22 년  5 월부터 .', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 1}),
 Document(page_content='복리 후생\n32022년  기안서  참고\nhttps://s3-us-west-2.amazonaws.com/secure.noti\non-static.com/f61d0495-80ca-497c-81b9-0286bb\ndb1863/20220516-22 년 _ 회식비 _ 지원 _ 품의서 ( 승인\n본).pdf\n음료\n직원들이  편하게  맛있는  음료를  드실  수  있도록  지원합니다 .\n연도 규정\n2023• 지원금액  : 5,000( 원 ) / 1 건당  • 음료는  커피 , 티 , 탄산음료  등  마실  수  있는  모든\n것들이  해당  • 지원금액은  1 회당  사용할  수  있는  금액이며  지원금액을  나누어  사\n용할 수  없음  • 음료는  근무일 , 근무시간에만  드실  수  있고  재택근무  유무와  관계\n없이 이용할  수  있음  • 지원금액은  당일  사용  기준이며  사용치  않으면  자동  소멸되\n고 누적으로  사용할  수  없음  • 어시양식  제출  시  비목은  ‘ 오커찬 ’ 으로  사유는  ‘ 복리\n후생’을  선택해서  제출  • 결제방식  : 개인별  보유중인  법인카드로  결제  • 지원기한\n: 특별히  정함이  없고  별도  공지가  있을  때  까지  지원  • 시행  : ‘23 년  4 월  12 일부터\n2022• 지원금액  : 4,500( 원 ) / 1 건당  • 음료는  커피 , 티 , 탄산음료  등  마실  수  있는  모든\n것들이  해당  • 지원금액은  1 회당  사용할  수  있는  금액이며  지원금액을  나누어  사\n용할 수  없음  • 음료는  근무일 , 근무시간에만  드실  수  있고  재택근무  유무와  관계\n없이 이용할  수  있음  • 지원금액은  당일  사용  기준이며  사용치  않으면  자동  소멸되\n고 누적으로  사용할  수  없음  • 어시양식  제출  시  비목은  ‘ 다과비 ’ 로  사유는  ‘ 복리후\n생’을 선택해서  제출  • 결제방식  : 개인별  보유중인  법인카드로  결제  • 지원기한  :\n특별히  정함이  없고  별도  공지가  있을  때  까지  지원  • 시행  : ‘22 년  6 월  14 일부터\n방학 제도\n임직원  여러분들에게  충분한  휴식을  제공하고자  방학  제도를  시행합니다 . 지침에  맞춰  방학 \n제도를  적극  이용하시길  바랍니다 .\n연도 규정', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 2}),
 Document(page_content='복리 후생\n4연도 규정\n2023• 지원  대상  : 입사일로부터  6 개월이  지난  자 . • 방학은  총  3 일로  사용시기는  정함\n이 없으며  당해년도  12 월  말까지  사용  • 방학  사용은  연내에  자유롭게  사용할  수\n있으며 , 기간  내  사용치  않은  방학은  소멸됩니다 . • 방학은  반드시  연차와  이어서\n사용해야  함  • 연차와  함께  2+1 방학  제도  사용  가능  조합  ( 예시 ) 참고  • 예시  : 연\n차 1일  사용  시  방학  사용할  수  없음  • 예시  : 연차  2 일에  방학  1 일 ( 총 3 일 ), 연차  3\n일에 방학  1일 ( 총 4 일 ) • 예시  : 연차  4 일에  방학  2 일 ( 총 6 일 ), 연차  5 일에  방학  2 일\n(총7일 ) • 예시  : 연차  6 일에  방학  3 일 ( 총 9 일 ), 연차  7 일  이상  방학  3 일 ( 총 10 일 ) •\n시행 : ‘23 년  4 월  12 일부터\n2022• 지원  대상  : 입사일로부터  6 개월이  지난  자 . • 여름방학과  겨울방학은  총  5 일입\n니다. • 연차와  이어서  사용  가능합니다 . • 여름과  겨울방학은  여러  날을  계속해서\n붙여서  사용하며  띄워서  사용은  금합니다 . 예를  들어  월 , 수 , 금  이렇게  사용은  안\n됩니다 . • 여름방학은  7 월 , 8 월 , 겨울방학은  11 월 , 12 월  기간  내  사용해야  하며  이\n기간 사용치  않은  방학은  자동  소멸됩니다 . • 입사일로부터  6 개월이  경과한  시점\n에 여름방학을  이용  못하고  겨울  방학만  이용  시  최대  3 일을  제공합니다 . • 여름 ,\n겨울방학  총  5일은  여름과  겨울  계절에  모두  몰아서  사용할  수  없으며  아래  조합\n으로(만 ) 나누어서  사용  바랍니다 . • 여름방학 , 겨울방학  사용  가능  조합  • 여름 (7\n월, 8월 ) 3일  + 겨울 (11 월 , 12 월 ) 2 일  : 총  5 일  • 여름 (7 월 , 8 월 ) 2 일  + 겨울 (11 월 ,\n12월) 3일  : 총  5일  • 시행  : ‘22 년  7 월  1 일부터\n도서 구매\n업무적으로  도서가  필요하신  분은  아래  절차에  따라  구매를  진행합니다 .\n연도 규정\n2023 2022• 구매  전  또는  후에  대표에게  보고한다  • 보고  후  직접  도서  구매를  진행한다  ( 결\n재는 개인카드 , 법인카드 ( 오프라인  서점 ) 사용  가능 ) • 개인카드  구매  시  ‘ 입체금\n청구’를  법인카드 ( 오프라인  서점 ) 구매  시  ‘ 법인카드  사용 ’ 작성 제출  • 입체금  청\n구와 법인카드  사용  신청서  작성  시  비고란에  ‘ 대표승인 ’ 이라고  기입한다 . • 도서\n는 공유를  위해  행정팀에서  도서  목록을  관리한다 . • 시행  : ‘21 년  10 월  20 일부터\n회사 티셔츠\n소속감  향상을  위하여  회사  티셔츠를  제공합니다 .\n연도 규정\n2023 • 신청  시  지급\n2022 • 입사  시  지급', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 3}),
 Document(page_content='복리 후생\n5', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 4})]
```

```python
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(texts, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
```

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # Modify model_name if you have access to GPT-4

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = retriever,
    return_source_documents=True)
```

```python
query = "음료 지원은?"
result = chain(query)
print(result)
```

```
{'question': '음료 지원은?', 'answer': 'The company provides support for employees to have comfortable and delicious drinks, including coffee, tea, and carbonated drinks. The support amount is 5,000 KRW per use, and it can only be used during working hours on the day of use. The support amount cannot be divided and accumulated, and it will expire if not used on the day of use. The payment method is through individual corporate cards. There is no information about other benefits or support related to drinks.\n', 'sources': '인공지능팩토리_복리후생_230516.pdf', 'source_documents': [Document(page_content='복리 후생\n32022년  기안서  참고\nhttps://s3-us-west-2.amazonaws.com/secure.noti\non-static.com/f61d0495-80ca-497c-81b9-0286bb\ndb1863/20220516-22 년 _ 회식비 _ 지원 _ 품의서 ( 승인\n본).pdf\n음료\n직원들이  편하게  맛있는  음료를  드실  수  있도록  지원합니다 .\n연도 규정\n2023• 지원금액  : 5,000( 원 ) / 1 건당  • 음료는  커피 , 티 , 탄산음료  등  마실  수  있는  모든\n것들이  해당  • 지원금액은  1 회당  사용할  수  있는  금액이며  지원금액을  나누어  사\n용할 수  없음  • 음료는  근무일 , 근무시간에만  드실  수  있고  재택근무  유무와  관계\n없이 이용할  수  있음  • 지원금액은  당일  사용  기준이며  사용치  않으면  자동  소멸되\n고 누적으로  사용할  수  없음  • 어시양식  제출  시  비목은  ‘ 오커찬 ’ 으로  사유는  ‘ 복리\n후생’을  선택해서  제출  • 결제방식  : 개인별  보유중인  법인카드로  결제  • 지원기한\n: 특별히  정함이  없고  별도  공지가  있을  때  까지  지원  • 시행  : ‘23 년  4 월  12 일부터\n2022• 지원금액  : 4,500( 원 ) / 1 건당  • 음료는  커피 , 티 , 탄산음료  등  마실  수  있는  모든\n것들이  해당  • 지원금액은  1 회당  사용할  수  있는  금액이며  지원금액을  나누어  사\n용할 수  없음  • 음료는  근무일 , 근무시간에만  드실  수  있고  재택근무  유무와  관계\n없이 이용할  수  있음  • 지원금액은  당일  사용  기준이며  사용치  않으면  자동  소멸되\n고 누적으로  사용할  수  없음  • 어시양식  제출  시  비목은  ‘ 다과비 ’ 로  사유는  ‘ 복리후\n생’을 선택해서  제출  • 결제방식  : 개인별  보유중인  법인카드로  결제  • 지원기한  :\n특별히  정함이  없고  별도  공지가  있을  때  까지  지원  • 시행  : ‘22 년  6 월  14 일부터\n방학 제도\n임직원  여러분들에게  충분한  휴식을  제공하고자  방학  제도를  시행합니다 . 지침에  맞춰  방학 \n제도를  적극  이용하시길  바랍니다 .\n연도 규정', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 2}), Document(page_content='복리 후생\n1☕\n복리 후생\n💡지속적으로  복리후생  정책이  변경되고  있으며 , 시행  혹은  변경  시마다  메일로  공\n지되는  내용은  해당  페이지를  통해서도  확인할  수  있으니  메일을  놓치거나  확인\n이 필요한  경우에는  반드시  본  페이지를  확인해주세요 .\n건강검진\n직원들의  건강  증진을  위하여  회사에서  건강검진  비용을  지원합니다 .\n연도 규정\n2023• 지원  금액 : 20 만원  한도 , 초과분은  개인  지출  • 지원  대상 : 입사일로부터  6 개월이\n지난 자  • ‘23 년  7 월  이후  입사자는  당해년도  대상에서  제외  • 지원  만료  시기 : ‘23\n년 12월  31일까지  • 결제  방법 : 개별  소유한  법인카드로  20 만원까지  결제  후 , 나\n머지 추가  비용에  대해서는  개인  지불  • 휴가  처리 : 오전  혹은  오후  반차  (0.5 일 )\n공가 처리 , 어시  기안  및  구글  캘린더  등록  • 기타 : 체질개선 ( 당뇨 , 비만 , 고혈압\n등) 프로그램  이용도  가능  • 시행  : ‘23 년  4 월  12 일부터\n2022• 지원  금액 : 20 만원  한도 , 초과분은  개인  지출  • 지원  대상 : 입사일로부터  6 개월이\n지난 자  • ‘22 년  7 월  이후  입사자는  당해년도  대상에서  제외  • 지원  만료  시기 : ‘22\n년 12월  31일까지  • 결제  방법 : 개별  소유한  법인카드로  20 만원까지  결제  후 , 나\n머지 추가  비용에  대해서는  개인  지불  • 휴가  처리 : 오전  혹은  오후  반차  (0.5 일 )\n공가 처리 , 구글  캘린더  등록  및  슬랙  # 휴가  채널  작성  • 공가  휴가  신청은  전자결\n재 어시를  통해서  신청  (’22 년  6 월  1 일  이후 ) • 기타 : 체질개선 ( 당뇨 , 비만 , 고혈압\n등) 프로그램  이용도  가능\n2022년  기안서  참고\nhttps://s3-us-west-2.amazonaws.com/secure.noti\non-static.com/45a30f62-1ef1-44fd-ad3e-cf8ede8\n837e9/20220419-22 년 _ 건강검진비용 _ 지원 _ 및 _ 건\n강검진 _실시 _ 품의서 ( 승인본 ).pdf', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 0})]}

```

```python
result['answer']
```

```
'The company provides support for employees to have comfortable and delicious drinks, including coffee, tea, and carbonated drinks. The support amount is 5,000 KRW per use, and it can only be used during working hours on the day of use. The support amount cannot be divided and accumulated, and it will expire if not used on the day of use. The payment method is through individual corporate cards. There is no information about other benefits or support related to drinks.\n'
```

```python
result['sources']
```

```
'인공지능팩토리_복리후생_230516.pdf'
```

```python
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template="""Use the following pieces of context to answer the users question shortly.
Given the following summaries of a long document and a question, create a final answer with references ("SOURCES"), use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}

You MUST answer in Korean and in Markdown format:"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)
```

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

chain_type_kwargs = {"prompt": prompt}

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # Modify model_name if you have access to GPT-4

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
```

```python
query = "음료 지원은?"
result = chain(query)
print(result)
```

```
{'question': '음료 지원은?', 'answer': "2022년과 2023년 모두 근무일, 근무시간에만 음료를 드실 수 있고, 지원금액은 1회당 사용할 수 있는 금액이며 지원금액을 나누어 사용할 수 없습니다. 지원금액은 당일 사용 기준이며 사용치 않으면 자동 소멸되고 누적으로 사용할 수 없습니다. 어시양식 제출 시 비목은 '오커찬'으로 사유는 '복리 후생'을 선택해서 제출하면 됩니다. 결제 방식은 개인별 보유중인 법인카드로 결제하며, 지원기한은 특별히 정해져 있지 않고 별도 공지가 있을 때까지 지원됩니다. (참고: 인공지능이 이해한 내용입니다. 자세한 내용은 SOURCES를 참고해주세요.) \n\n**SOURCES:** \n- 복리 후생 32022년 기안서 참고\n- 복리 후생 1☕", 'sources': '', 'source_documents': [Document(page_content='복리 후생\n32022년  기안서  참고\nhttps://s3-us-west-2.amazonaws.com/secure.noti\non-static.com/f61d0495-80ca-497c-81b9-0286bb\ndb1863/20220516-22 년 _ 회식비 _ 지원 _ 품의서 ( 승인\n본).pdf\n음료\n직원들이  편하게  맛있는  음료를  드실  수  있도록  지원합니다 .\n연도 규정\n2023• 지원금액  : 5,000( 원 ) / 1 건당  • 음료는  커피 , 티 , 탄산음료  등  마실  수  있는  모든\n것들이  해당  • 지원금액은  1 회당  사용할  수  있는  금액이며  지원금액을  나누어  사\n용할 수  없음  • 음료는  근무일 , 근무시간에만  드실  수  있고  재택근무  유무와  관계\n없이 이용할  수  있음  • 지원금액은  당일  사용  기준이며  사용치  않으면  자동  소멸되\n고 누적으로  사용할  수  없음  • 어시양식  제출  시  비목은  ‘ 오커찬 ’ 으로  사유는  ‘ 복리\n후생’을  선택해서  제출  • 결제방식  : 개인별  보유중인  법인카드로  결제  • 지원기한\n: 특별히  정함이  없고  별도  공지가  있을  때  까지  지원  • 시행  : ‘23 년  4 월  12 일부터\n2022• 지원금액  : 4,500( 원 ) / 1 건당  • 음료는  커피 , 티 , 탄산음료  등  마실  수  있는  모든\n것들이  해당  • 지원금액은  1 회당  사용할  수  있는  금액이며  지원금액을  나누어  사\n용할 수  없음  • 음료는  근무일 , 근무시간에만  드실  수  있고  재택근무  유무와  관계\n없이 이용할  수  있음  • 지원금액은  당일  사용  기준이며  사용치  않으면  자동  소멸되\n고 누적으로  사용할  수  없음  • 어시양식  제출  시  비목은  ‘ 다과비 ’ 로  사유는  ‘ 복리후\n생’을 선택해서  제출  • 결제방식  : 개인별  보유중인  법인카드로  결제  • 지원기한  :\n특별히  정함이  없고  별도  공지가  있을  때  까지  지원  • 시행  : ‘22 년  6 월  14 일부터\n방학 제도\n임직원  여러분들에게  충분한  휴식을  제공하고자  방학  제도를  시행합니다 . 지침에  맞춰  방학 \n제도를  적극  이용하시길  바랍니다 .\n연도 규정', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 2}), Document(page_content='복리 후생\n1☕\n복리 후생\n💡지속적으로  복리후생  정책이  변경되고  있으며 , 시행  혹은  변경  시마다  메일로  공\n지되는  내용은  해당  페이지를  통해서도  확인할  수  있으니  메일을  놓치거나  확인\n이 필요한  경우에는  반드시  본  페이지를  확인해주세요 .\n건강검진\n직원들의  건강  증진을  위하여  회사에서  건강검진  비용을  지원합니다 .\n연도 규정\n2023• 지원  금액 : 20 만원  한도 , 초과분은  개인  지출  • 지원  대상 : 입사일로부터  6 개월이\n지난 자  • ‘23 년  7 월  이후  입사자는  당해년도  대상에서  제외  • 지원  만료  시기 : ‘23\n년 12월  31일까지  • 결제  방법 : 개별  소유한  법인카드로  20 만원까지  결제  후 , 나\n머지 추가  비용에  대해서는  개인  지불  • 휴가  처리 : 오전  혹은  오후  반차  (0.5 일 )\n공가 처리 , 어시  기안  및  구글  캘린더  등록  • 기타 : 체질개선 ( 당뇨 , 비만 , 고혈압\n등) 프로그램  이용도  가능  • 시행  : ‘23 년  4 월  12 일부터\n2022• 지원  금액 : 20 만원  한도 , 초과분은  개인  지출  • 지원  대상 : 입사일로부터  6 개월이\n지난 자  • ‘22 년  7 월  이후  입사자는  당해년도  대상에서  제외  • 지원  만료  시기 : ‘22\n년 12월  31일까지  • 결제  방법 : 개별  소유한  법인카드로  20 만원까지  결제  후 , 나\n머지 추가  비용에  대해서는  개인  지불  • 휴가  처리 : 오전  혹은  오후  반차  (0.5 일 )\n공가 처리 , 구글  캘린더  등록  및  슬랙  # 휴가  채널  작성  • 공가  휴가  신청은  전자결\n재 어시를  통해서  신청  (’22 년  6 월  1 일  이후 ) • 기타 : 체질개선 ( 당뇨 , 비만 , 고혈압\n등) 프로그램  이용도  가능\n2022년  기안서  참고\nhttps://s3-us-west-2.amazonaws.com/secure.noti\non-static.com/45a30f62-1ef1-44fd-ad3e-cf8ede8\n837e9/20220419-22 년 _ 건강검진비용 _ 지원 _ 및 _ 건\n강검진 _실시 _ 품의서 ( 승인본 ).pdf', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 0})]}

```

```python
result['answer']
```

```
"2022년과 2023년 모두 근무일, 근무시간에만 음료를 드실 수 있고, 지원금액은 1회당 사용할 수 있는 금액이며 지원금액을 나누어 사용할 수 없습니다. 지원금액은 당일 사용 기준이며 사용치 않으면 자동 소멸되고 누적으로 사용할 수 없습니다. 어시양식 제출 시 비목은 '오커찬'으로 사유는 '복리 후생'을 선택해서 제출하면 됩니다. 결제 방식은 개인별 보유중인 법인카드로 결제하며, 지원기한은 특별히 정해져 있지 않고 별도 공지가 있을 때까지 지원됩니다. (참고: 인공지능이 이해한 내용입니다. 자세한 내용은 SOURCES를 참고해주세요.) \n\n**SOURCES:** \n- 복리 후생 32022년 기안서 참고\n- 복리 후생 1☕"
```

```python
result['source_documents']
```

```
[Document(page_content='복리 후생\n32022년  기안서  참고\nhttps://s3-us-west-2.amazonaws.com/secure.noti\non-static.com/f61d0495-80ca-497c-81b9-0286bb\ndb1863/20220516-22 년 _ 회식비 _ 지원 _ 품의서 ( 승인\n본).pdf\n음료\n직원들이  편하게  맛있는  음료를  드실  수  있도록  지원합니다 .\n연도 규정\n2023• 지원금액  : 5,000( 원 ) / 1 건당  • 음료는  커피 , 티 , 탄산음료  등  마실  수  있는  모든\n것들이  해당  • 지원금액은  1 회당  사용할  수  있는  금액이며  지원금액을  나누어  사\n용할 수  없음  • 음료는  근무일 , 근무시간에만  드실  수  있고  재택근무  유무와  관계\n없이 이용할  수  있음  • 지원금액은  당일  사용  기준이며  사용치  않으면  자동  소멸되\n고 누적으로  사용할  수  없음  • 어시양식  제출  시  비목은  ‘ 오커찬 ’ 으로  사유는  ‘ 복리\n후생’을  선택해서  제출  • 결제방식  : 개인별  보유중인  법인카드로  결제  • 지원기한\n: 특별히  정함이  없고  별도  공지가  있을  때  까지  지원  • 시행  : ‘23 년  4 월  12 일부터\n2022• 지원금액  : 4,500( 원 ) / 1 건당  • 음료는  커피 , 티 , 탄산음료  등  마실  수  있는  모든\n것들이  해당  • 지원금액은  1 회당  사용할  수  있는  금액이며  지원금액을  나누어  사\n용할 수  없음  • 음료는  근무일 , 근무시간에만  드실  수  있고  재택근무  유무와  관계\n없이 이용할  수  있음  • 지원금액은  당일  사용  기준이며  사용치  않으면  자동  소멸되\n고 누적으로  사용할  수  없음  • 어시양식  제출  시  비목은  ‘ 다과비 ’ 로  사유는  ‘ 복리후\n생’을 선택해서  제출  • 결제방식  : 개인별  보유중인  법인카드로  결제  • 지원기한  :\n특별히  정함이  없고  별도  공지가  있을  때  까지  지원  • 시행  : ‘22 년  6 월  14 일부터\n방학 제도\n임직원  여러분들에게  충분한  휴식을  제공하고자  방학  제도를  시행합니다 . 지침에  맞춰  방학 \n제도를  적극  이용하시길  바랍니다 .\n연도 규정', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 2}),
 Document(page_content='복리 후생\n1☕\n복리 후생\n💡지속적으로  복리후생  정책이  변경되고  있으며 , 시행  혹은  변경  시마다  메일로  공\n지되는  내용은  해당  페이지를  통해서도  확인할  수  있으니  메일을  놓치거나  확인\n이 필요한  경우에는  반드시  본  페이지를  확인해주세요 .\n건강검진\n직원들의  건강  증진을  위하여  회사에서  건강검진  비용을  지원합니다 .\n연도 규정\n2023• 지원  금액 : 20 만원  한도 , 초과분은  개인  지출  • 지원  대상 : 입사일로부터  6 개월이\n지난 자  • ‘23 년  7 월  이후  입사자는  당해년도  대상에서  제외  • 지원  만료  시기 : ‘23\n년 12월  31일까지  • 결제  방법 : 개별  소유한  법인카드로  20 만원까지  결제  후 , 나\n머지 추가  비용에  대해서는  개인  지불  • 휴가  처리 : 오전  혹은  오후  반차  (0.5 일 )\n공가 처리 , 어시  기안  및  구글  캘린더  등록  • 기타 : 체질개선 ( 당뇨 , 비만 , 고혈압\n등) 프로그램  이용도  가능  • 시행  : ‘23 년  4 월  12 일부터\n2022• 지원  금액 : 20 만원  한도 , 초과분은  개인  지출  • 지원  대상 : 입사일로부터  6 개월이\n지난 자  • ‘22 년  7 월  이후  입사자는  당해년도  대상에서  제외  • 지원  만료  시기 : ‘22\n년 12월  31일까지  • 결제  방법 : 개별  소유한  법인카드로  20 만원까지  결제  후 , 나\n머지 추가  비용에  대해서는  개인  지불  • 휴가  처리 : 오전  혹은  오후  반차  (0.5 일 )\n공가 처리 , 구글  캘린더  등록  및  슬랙  # 휴가  채널  작성  • 공가  휴가  신청은  전자결\n재 어시를  통해서  신청  (’22 년  6 월  1 일  이후 ) • 기타 : 체질개선 ( 당뇨 , 비만 , 고혈압\n등) 프로그램  이용도  가능\n2022년  기안서  참고\nhttps://s3-us-west-2.amazonaws.com/secure.noti\non-static.com/45a30f62-1ef1-44fd-ad3e-cf8ede8\n837e9/20220419-22 년 _ 건강검진비용 _ 지원 _ 및 _ 건\n강검진 _실시 _ 품의서 ( 승인본 ).pdf', metadata={'source': '인공지능팩토리_복리후생_230516.pdf', 'page': 0})]
```

```python
for doc in result['source_documents']:
    print('내용 : ' + doc.page_content[0:100].replace('\n', ' '))
    print('파일 : ' + doc.metadata['source'])
    print('페이지 : ' + str(doc.metadata['page']))
```

```
내용 : 복리 후생 32022년  기안서  참고 https://s3-us-west-2.amazonaws.com/secure.noti on-static.com/f61d0495-80ca-497
파일 : 인공지능팩토리_복리후생_230516.pdf
페이지 : 2
내용 : 복리 후생 1☕ 복리 후생 💡지속적으로  복리후생  정책이  변경되고  있으며 , 시행  혹은  변경  시마다  메일로  공 지되는  내용은  해당  페이지를  통해서도  확인할 
파일 : 인공지능팩토리_복리후생_230516.pdf
페이지 : 0

```

```python
!pip install gradio # 그라디오 라이브러리를 설치합니다.
```

```
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting gradio
  Downloading gradio-3.33.1-py3-none-any.whl (20.0 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m20.0/20.0 MB[0m [31m75.7 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting aiofiles (from gradio)
  Downloading aiofiles-23.1.0-py3-none-any.whl (14 kB)
Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from gradio) (3.8.4)
Requirement already satisfied: altair>=4.2.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (4.2.2)
Requirement already satisfied: fastapi in /usr/local/lib/python3.10/dist-packages (from gradio) (0.96.0)
Collecting ffmpy (from gradio)
  Downloading ffmpy-0.3.0.tar.gz (4.8 kB)
  Preparing metadata (setup.py) ... [?25l[?25hdone
Collecting gradio-client>=0.2.4 (from gradio)
  Downloading gradio_client-0.2.5-py3-none-any.whl (288 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m288.1/288.1 kB[0m [31m28.9 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting httpx (from gradio)
  Downloading httpx-0.24.1-py3-none-any.whl (75 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.4/75.4 kB[0m [31m9.0 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting huggingface-hub>=0.14.0 (from gradio)
  Downloading huggingface_hub-0.15.1-py3-none-any.whl (236 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m236.8/236.8 kB[0m [31m26.2 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.1.2)
Requirement already satisfied: markdown-it-py[linkify]>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.2.0)
Requirement already satisfied: markupsafe in /usr/local/lib/python3.10/dist-packages (from gradio) (2.1.2)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from gradio) (3.7.1)
Collecting mdit-py-plugins<=0.3.3 (from gradio)
  Downloading mdit_py_plugins-0.3.3-py3-none-any.whl (50 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m50.5/50.5 kB[0m [31m6.3 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from gradio) (1.22.4)
Collecting orjson (from gradio)
  Downloading orjson-3.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (136 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m137.0/137.0 kB[0m [31m16.7 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from gradio) (1.5.3)
Requirement already satisfied: pillow in /usr/local/lib/python3.10/dist-packages (from gradio) (8.4.0)
Requirement already satisfied: pydantic in /usr/local/lib/python3.10/dist-packages (from gradio) (1.10.7)
Collecting pydub (from gradio)
  Downloading pydub-0.25.1-py2.py3-none-any.whl (32 kB)
Requirement already satisfied: pygments>=2.12.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.14.0)
Collecting python-multipart (from gradio)
  Downloading python_multipart-0.0.6-py3-none-any.whl (45 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.7/45.7 kB[0m [31m4.8 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from gradio) (6.0)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from gradio) (2.31.0)
Collecting semantic-version (from gradio)
  Downloading semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from gradio) (4.5.0)
Requirement already satisfied: uvicorn>=0.14.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (0.22.0)
Requirement already satisfied: websockets>=10.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (11.0.3)
Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair>=4.2.0->gradio) (0.4)
Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair>=4.2.0->gradio) (4.3.3)
Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair>=4.2.0->gradio) (0.12.0)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from gradio-client>=0.2.4->gradio) (2023.4.0)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from gradio-client>=0.2.4->gradio) (23.1)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.14.0->gradio) (3.12.0)
Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.14.0->gradio) (4.65.0)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py[linkify]>=2.0.0->gradio) (0.1.2)
Collecting linkify-it-py<3,>=1 (from markdown-it-py[linkify]>=2.0.0->gradio)
  Downloading linkify_it_py-2.0.2-py3-none-any.whl (19 kB)
Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->gradio) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->gradio) (2022.7.1)
Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.10/dist-packages (from uvicorn>=0.14.0->gradio) (8.1.3)
Requirement already satisfied: h11>=0.8 in /usr/local/lib/python3.10/dist-packages (from uvicorn>=0.14.0->gradio) (0.14.0)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->gradio) (23.1.0)
Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->gradio) (2.0.12)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->gradio) (6.0.4)
Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp->gradio) (4.0.2)
Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->gradio) (1.9.2)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->gradio) (1.3.3)
Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->gradio) (1.3.1)
Requirement already satisfied: starlette<0.28.0,>=0.27.0 in /usr/local/lib/python3.10/dist-packages (from fastapi->gradio) (0.27.0)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx->gradio) (2022.12.7)
Collecting httpcore<0.18.0,>=0.15.0 (from httpx->gradio)
  Downloading httpcore-0.17.2-py3-none-any.whl (72 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m72.5/72.5 kB[0m [31m9.3 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: idna in /usr/local/lib/python3.10/dist-packages (from httpx->gradio) (3.4)
Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from httpx->gradio) (1.3.0)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->gradio) (1.0.7)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->gradio) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->gradio) (4.39.3)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->gradio) (1.4.4)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->gradio) (3.0.9)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->gradio) (1.26.15)
Requirement already satisfied: anyio<5.0,>=3.0 in /usr/local/lib/python3.10/dist-packages (from httpcore<0.18.0,>=0.15.0->httpx->gradio) (3.6.2)
Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair>=4.2.0->gradio) (0.19.3)
Collecting uc-micro-py (from linkify-it-py<3,>=1->markdown-it-py[linkify]>=2.0.0->gradio)
  Downloading uc_micro_py-1.0.2-py3-none-any.whl (6.2 kB)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas->gradio) (1.16.0)
Building wheels for collected packages: ffmpy
  Building wheel for ffmpy (setup.py) ... [?25l[?25hdone
  Created wheel for ffmpy: filename=ffmpy-0.3.0-py3-none-any.whl size=4694 sha256=b101c769ccfdd63cd71c36b3e638437835c6ab69e0f2b45f74dd06fd4d731f55
  Stored in directory: /root/.cache/pip/wheels/0c/c2/0e/3b9c6845c6a4e35beb90910cc70d9ac9ab5d47402bd62af0df
Successfully built ffmpy
Installing collected packages: pydub, ffmpy, uc-micro-py, semantic-version, python-multipart, orjson, aiofiles, mdit-py-plugins, linkify-it-py, huggingface-hub, httpcore, httpx, gradio-client, gradio
Successfully installed aiofiles-23.1.0 ffmpy-0.3.0 gradio-3.33.1 gradio-client-0.2.5 httpcore-0.17.2 httpx-0.24.1 huggingface-hub-0.15.1 linkify-it-py-2.0.2 mdit-py-plugins-0.3.3 orjson-3.9.0 pydub-0.25.1 python-multipart-0.0.6 semantic-version-2.10.0 uc-micro-py-1.0.2

```

```python
import gradio as gr
```

```python
def respond(message, chat_history):  # 채팅봇의 응답을 처리하는 함수를 정의합니다.

    result = chain(message)

    bot_message = result['answer']

    for i, doc in enumerate(result['source_documents']):
        bot_message += '[' + str(i+1) + '] ' + doc.metadata['source'] + '(' + str(doc.metadata['page']) + ') '

    chat_history.append((message, bot_message))  # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가합니다.

    return "", chat_history  # 수정된 채팅 기록을 반환합니다.

with gr.Blocks() as demo:  # gr.Blocks()를 사용하여 인터페이스를 생성합니다.
    chatbot = gr.Chatbot(label="채팅창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
    msg = gr.Textbox(label="입력")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
    clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.

    msg.submit(respond, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
    clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.

demo.launch(debug=True)  # 인터페이스를 실행합니다. 실행하면 사용자는 '입력' 텍스트박스에 메시지를 작성하고 제출할 수 있으며, '초기화' 버튼을 통해 채팅 기록을 초기화 할 수 있습니다.
```

```
Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().
Note: opening Chrome Inspector may crash demo inside Colab notebooks.

To create a public link, set `share=True` in `launch()`.

```

```
<IPython.core.display.Javascript object>
```

```python

```

