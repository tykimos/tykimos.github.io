---
layout: post
title: "챗GPT API와 외부도구 연계"
author: 김태영
date: 2023-8-21 00:00:00
categories: ai
comments: true
---

이 노트북은 GPT 모델의 능력을 확장하기 위해 외부 함수와 함께 Chat Completions API를 사용하는 방법에 대해 다룹니다.

`functions`는 Chat Completion API에서 선택적으로 사용할 수 있는 매개변수로, 함수 사양을 제공하는 데 사용됩니다. 이의 목적은 제공된 사양을 준수하는 함수 인수를 생성하도록 모델을 활성화하는 것입니다. API는 실제로 어떤 함수 호출도 실행하지 않습니다. 개발자는 모델 출력을 사용하여 함수 호출을 실행해야 합니다.

`functions` 매개변수가 제공되면 모델은 기본적으로 어떤 함수를 사용할지 결정하게 됩니다. `function_call` 매개변수를 `{"name": "<함수-이름-삽입>"}`로 설정함으로써 API는 특정 함수를 사용하도록 강제될 수 있습니다. 또한 `function_call` 매개변수를 `"none"`으로 설정함으로써 API는 어떤 함수도 사용하지 않도록 강제될 수 있습니다. 함수가 사용되면, 출력에는 응답에서 `"finish_reason": "function_call"`이 포함되며, 함수의 이름과 생성된 함수 인수를 가진 `function_call` 객체도 포함됩니다.

### 개요

이 노트북에는 다음의 2개 섹션이 포함되어 있습니다:

- **함수 인수 생성하는 방법:** 일련의 함수를 지정하고 API를 사용하여 함수 인수를 생성합니다.
- **모델이 생성한 인수로 함수 호출하는 방법:** 실제로 모델이 생성한 인수로 함수를 실행함으로써 루프를 닫습니다.


```python
import os

os.environ["OPENAI_API_KEY"] = "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" # 환경변수에 OPENAI_API_KEY를 설정합니다.
```

## How to generate function arguments

```python
!pip install scipy
!pip install tenacity
!pip install tiktoken
!pip install termcolor
!pip install openai
!pip install requests
```

```
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (1.10.1)
Requirement already satisfied: numpy<1.27.0,>=1.19.5 in /usr/local/lib/python3.10/dist-packages (from scipy) (1.22.4)
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: tenacity in /usr/local/lib/python3.10/dist-packages (8.2.2)
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting tiktoken
  Downloading tiktoken-0.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m25.8 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2022.10.31)
Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2.27.1)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2022.12.7)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2.0.12)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.4)
Installing collected packages: tiktoken
Successfully installed tiktoken-0.4.0
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: termcolor in /usr/local/lib/python3.10/dist-packages (2.3.0)
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting openai
  Downloading openai-0.27.8-py3-none-any.whl (73 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m73.6/73.6 kB[0m [31m4.5 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: requests>=2.20 in /usr/local/lib/python3.10/dist-packages (from openai) (2.27.1)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from openai) (4.65.0)
Collecting aiohttp (from openai)
  Downloading aiohttp-3.8.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.0 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m29.9 MB/s[0m eta [36m0:00:00[0m
[?25hRequirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2022.12.7)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2.0.12)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (3.4)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (23.1.0)
Collecting multidict<7.0,>=4.5 (from aiohttp->openai)
  Downloading multidict-6.0.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (114 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m114.5/114.5 kB[0m [31m14.4 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting async-timeout<5.0,>=4.0.0a3 (from aiohttp->openai)
  Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
Collecting yarl<2.0,>=1.0 (from aiohttp->openai)
  Downloading yarl-1.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (268 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m268.8/268.8 kB[0m [31m29.2 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting frozenlist>=1.1.1 (from aiohttp->openai)
  Downloading frozenlist-1.3.3-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (149 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m149.6/149.6 kB[0m [31m18.0 MB/s[0m eta [36m0:00:00[0m
[?25hCollecting aiosignal>=1.1.2 (from aiohttp->openai)
  Downloading aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
Installing collected packages: multidict, frozenlist, async-timeout, yarl, aiosignal, aiohttp, openai
Successfully installed aiohttp-3.8.4 aiosignal-1.3.1 async-timeout-4.0.2 frozenlist-1.3.3 multidict-6.0.4 openai-0.27.8 yarl-1.9.2
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (2.27.1)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests) (2022.12.7)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests) (2.0.12)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests) (3.4)

```

```python
import json
import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

GPT_MODEL = "gpt-3.5-turbo-0613"
openai.api_key = "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" # 환경변수에 OPENAI_API_KEY를 설정합니다.
```

### 유틸리티

먼저 Chat Completions API에 호출을 하기 위한 몇 가지 유틸리티와 대화 상태를 유지하고 추적하기 위한 유틸리티를 정의해봅시다.

```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

```

```python
def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(f"function ({message['name']}): {message['content']}\n")
    for formatted_message in formatted_messages:
        print(
            colored(
                formatted_message,
                role_to_color[messages[formatted_messages.index(formatted_message)]["role"]],
            )
        )
```

### 기본 개념

가상의 날씨 API와 인터페이스하기 위한 몇몇 함수 사양을 생성해봅시다. 이러한 함수 사양을 Chat Completions API에 전달하여 사양에 부합하는 함수 인수를 생성합니다.

```python
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        },
    },
    {
        "name": "get_n_day_weather_forecast",
        "description": "Get an N-day weather forecast",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
                "num_days": {
                    "type": "integer",
                    "description": "The number of days to forecast",
                }
            },
            "required": ["location", "format", "num_days"]
        },
    },
]
```

만약 우리가 모델에게 현재 날씨에 대해 물어본다면, 모델은 몇 가지 구체적인 질문으로 응답할 것입니다.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "What's the weather like today"})
chat_response = chat_completion_request(
    messages, functions=functions
)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
assistant_message

```

우리가 누락된 정보를 제공하면, 모델은 우리를 위해 적절한 함수 인수를 생성해줄 것입니다.

```python
messages.append({"role": "user", "content": "I'm in Glasgow, Scotland."})
chat_response = chat_completion_request(
    messages, functions=functions
)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
assistant_message

```

다르게 프롬프트하면, 우리가 알려준 다른 함수를 대상으로 하게 만들 수 있습니다.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in Glasgow, Scotland over the next x days"})
chat_response = chat_completion_request(
    messages, functions=functions
)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
assistant_message

```

다시 한번, 모델은 아직 충분한 정보를 갖고 있지 않기 때문에 우리에게 구체화를 요청하고 있습니다. 이 경우에는 이미 예보를 위한 위치를 알고 있지만, 예보에 필요한 일수가 얼마나 되는지 알아야 합니다.

```python
messages.append({"role": "user", "content": "5 days"})
chat_response = chat_completion_request(
    messages, functions=functions
)
chat_response.json()["choices"][0]

```

#### 특정 함수의 사용 강제 또는 함수 사용 안 함

우리는 `function_call` 인수를 사용하여 특정 함수, 예를 들면 `get_n_day_weather_forecast`를 사용하도록 모델에게 강제할 수 있습니다. 이렇게 하면, 모델은 그것을 어떻게 사용해야 하는지에 대한 가정을 하게 됩니다.

```python
# in this cell we force the model to use get_n_day_weather_forecast
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me a weather report for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, functions=functions, function_call={"name": "get_n_day_weather_forecast"}
)
chat_response.json()["choices"][0]["message"]

```

```python
# if we don't force the model to use get_n_day_weather_forecast it may not
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me a weather report for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, functions=functions
)
chat_response.json()["choices"][0]["message"]

```

모델에게 아예 함수를 사용하지 않도록 강제할 수도 있습니다. 이렇게 하면 모델이 적절한 함수 호출을 생성하는 것을 방지합니다.

```python
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me the current weather (use Celcius) for Toronto, Canada."})
chat_response = chat_completion_request(
    messages, functions=functions, function_call="none"
)
chat_response.json()["choices"][0]["message"]

```

## 모델이 생성한 인수로 함수 호출하는 방법

다음 예제에서는 모델이 생성한 입력을 가진 함수를 어떻게 실행하는지 보여줄 것이며, 이를 사용하여 데이터베이스에 관한 질문에 대답할 수 있는 에이전트를 구현하는 방법을 소개합니다. 단순화를 위해 [Chinook 샘플 데이터베이스](https://www.sqlitetutorial.net/sqlite-sample-database/)를 사용하겠습니다.

*주의:* SQL 생성은 생산 환경에서 높은 위험을 수반할 수 있습니다. 모델들은 올바른 SQL을 생성하는 데 완벽하게 믿을 수 없기 때문입니다.

### SQL 쿼리를 실행하기 위한 함수 지정

먼저 SQLite 데이터베이스에서 데이터를 추출하는 데 도움이 되는 유틸리티 함수를 정의해보겠습니다.

```python
import sqlite3

conn = sqlite3.connect("Chinook.db")
print("Opened database successfully")
```

```
데이터베이스를 성공적으로 열었습니다.

```

```python
def get_table_names(conn):
    """Return a list of table names."""
    table_names = []
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names


def get_column_names(conn, table_name):
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names


def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts

```

이제 이 유틸리티 함수들을 사용하여 데이터베이스 스키마의 표현을 추출할 수 있습니다.

```python
database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)
```

이전과 마찬가지로, API에게 인수를 생성하도록 원하는 함수에 대한 함수 사양을 정의하겠습니다. 데이터베이스 스키마를 함수 사양에 삽입한다는 점에 주목하세요. 이것은 모델이 알아야 할 중요한 정보입니다.

```python
functions = [
    {
        "name": "ask_database",
        "description": "Use this function to answer user questions about music. Output should be a fully formed SQL query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The query should be returned in plain text, not in JSON.
                            """,
                }
            },
            "required": ["query"],
        },
    }
]
```

### SQL 쿼리 실행

이제 데이터베이스에 쿼리를 실제로 실행할 함수를 구현해봅시다.

```python
def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results

def execute_function_call(message):
    if message["function_call"]["name"] == "ask_database":
        query = json.loads(message["function_call"]["arguments"])["query"]
        results = ask_database(conn, query)
    else:
        results = f"Error: function {message['function_call']['name']} does not exist"
    return results
```

```python
messages = []
messages.append({"role": "system", "content": "Answer user questions by generating SQL queries against the Chinook Music Database."})
messages.append({"role": "user", "content": "Hi, who are the top 5 artists by number of tracks?"})
chat_response = chat_completion_request(messages, functions)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
if assistant_message.get("function_call"):
    results = execute_function_call(assistant_message)
    messages.append({"role": "function", "name": assistant_message["function_call"]["name"], "content": results})
pretty_print_conversation(messages)
```

```
system: Answer user questions by generating SQL queries against the Chinook Music Database.

user: Hi, who are the top 5 artists by number of tracks?

assistant: {'name': 'ask_database', 'arguments': '{\n  "query": "SELECT Artist.Name, COUNT(*) AS TrackCount FROM Artist INNER JOIN Album ON Artist.ArtistId = Album.ArtistId INNER JOIN Track ON Album.AlbumId = Track.AlbumId GROUP BY Artist.ArtistId ORDER BY TrackCount DESC LIMIT 5"\n}'}

function (ask_database): [('Iron Maiden', 213), ('U2', 135), ('Led Zeppelin', 114), ('Metallica', 112), ('Lost', 92)]


```

```python
messages.append({"role": "user", "content": "What is the name of the album with the most tracks?"})
chat_response = chat_completion_request(messages, functions)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
if assistant_message.get("function_call"):
    results = execute_function_call(assistant_message)
    messages.append({"role": "function", "content": results, "name": assistant_message["function_call"]["name"]})
pretty_print_conversation(messages)
```

### 출처

* https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
