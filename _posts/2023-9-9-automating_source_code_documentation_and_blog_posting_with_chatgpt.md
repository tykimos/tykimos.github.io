---
layout: post
title: 챗GPT로 소스코드 문서화부터 블로그 포스팅까지 자동화!
author: Taeyoung Kim
date: 2023-9-9 00:00:00
categories: bic
comments: true
image: https://tykimos.github.io/warehouse/2023/2023-/2023-9-9-automating_source_code_documentation_and_blog_posting_with_chatgpt_title.png
---

![img](https://tykimos.github.io/warehouse/2023/2023-/2023-9-9-automating_source_code_documentation_and_blog_posting_with_chatgpt_title.png)

## 전체 흐름
The ASCII visualization of the input source code:

```
{ Input: Source Code }
        |
        v
[ extract_functions_from_file() ]
        |
        v
{Functions} {Source Code}
        |
        v
[ generate_flow() ]
        |
        v
{Flow Description}
        |
        v
[ generate_description() ]
        |
        v
{Function Descriptions}
        |
        v
[ create_or_update_github_file() ]
        |
        v
{GitHub File Creation/Update}
```

아래는 작업 흐름에 대한 한국어 설명입니다:

1. 입력으로 소스 코드를 받습니다.
2. `extract_functions_from_file()` 함수는 파일 경로를 인자로 받아 해당 파일에서 모든 함수를 추출하고, 함수 리스트와 소스 코드를 반환합니다.
3. `generate_flow()` 함수는 추출된 소스 코드를 사용하여 각 함수와 그 출력 사이의 관계를 시각화한 ASCII 표현을 생성합니다.
4. `generate_description()` 함수는 각 함수의 소스 코드를 이용하여 한국어로 함수를 설명하고 반환합니다.
5. 마지막으로, `create_or_update_github_file()` 함수는 생성된 흐름 설명과 함수 설명을 이용하여 GitHub 파일을 생성하거나 업데이트합니다.

## 함수 목록
- get_source_of_function
- get_response_from_llm
- generate_flow
- generate_description
- extract_functions_from_file
- gen_content_body
- get_file_sha
- create_or_update_github_file

## 함수 설명
### get_source_of_function

'get_source_of_function' 함수는 파이썬 코드에서 특정 함수의 소스 코드를 추출하는 기능을 합니다. 먼저 주어진 'source_code'를 파싱하여 추상 구문 트리(AST)를 생성합니다. 그 다음, AST를 순회하면서 각 노드가 함수 정의인지, 그리고 그 함수의 이름이 'function_name'과 일치하는지 확인합니다. 이 조건에 맞는 함수를 찾으면, 그 함수의 시작 라인과 끝 라인을 이용하여 원본 코드에서 해당 부분을 추출하고 반환합니다. 만약 일치하는 함수가 없다면, 빈 문자열을 반환합니다. 이 함수는 파이썬 코드 분석이나 디버깅에 유용하게 사용될 수 있습니다.

```python
def get_source_of_function(function_name, source_code):
    node = ast.parse(source_code)
    for n in ast.walk(node):
        if isinstance(n, ast.FunctionDef) and n.name == function_name:
            return source_code.splitlines()[n.lineno-1:n.end_lineno]
    return ""
```

### get_response_from_llm

'get_response_from_llm' 함수는 두 개의 매개변수, 즉 시스템 프롬프트와 사용자 프롬프트를 사용합니다. 이 함수는 openai의 ChatCompletion을 사용하여 'gpt-4' 모델을 기반으로 채팅 대화를 생성합니다. 이때 생성된 대화의 '온도'는 0.7로 설정되며, 이는 생성된 텍스트의 무작위성을 제어하는 요소입니다. 메시지는 두 부분으로 구성되며, 하나는 '시스템' 역할을 가진 시스템 프롬프트이고, 다른 하나는 '사용자' 역할을 가진 사용자 프롬프트입니다. 이 함수는 생성된 응답에서 첫 번째 선택사항의 메시지 내용을 반환합니다.

```python
def get_response_from_llm(system_prompt, user_prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    return response['choices'][0]['message']['content']
```

### generate_flow

'generate_flow' 함수는 주어진 소스 코드를 분석하고, 그 코드에 포함된 함수들과 그 결과를 추출하는 역할을 합니다. 이 함수는 ASCII 아트 표현을 사용하여 이러한 함수와 그 출력 사이의 관계를 시각화합니다. ASCII 아트는 본래 소스 코드의 주 입력부터 시작하여, "|"와 "v"를 사용하여 입력에서 첫 번째 함수로의 흐름을 나타냅니다. 각 함수는 대괄호로 표현되며, 각 함수의 출력은 중괄호 안에 표시되고, 하나의 함수에서 여러 출력이 나오는 경우 공백으로 구분합니다. 한 함수의 여러 출력이 다른 함수의 입력으로 사용될 경우 분기 화살표를 사용하여 흐름을 나타냅니다. 이 표현법을 계속 사용하여 최종 출력까지의 전체 프로세스를 매핑합니다. 이 흐름은 일반 코드 블록에 작성되어야 합니다. 이 함수는 입력 소스 코드를 기반으로 함수와 그들의 순서를 결정하고, 시작부터 끝까지의 흐름을 명확하게 시각화하는 ASCII 표현을 구성합니다.

```python
def generate_flow(source_code):
    
    system_prompt = (
        "Given a source code as input, your task is to analyze and extract the functions and their resulting outputs. "
        "Then, visualize the relationship between these functions and their outputs using an ASCII art representation as following [example]. Then, explain the your flow in Korean. You should write in Korean except for visualization.\n"
        "\n"
        "Steps for visualization:\n"
        "1. Begin with the primary input of the source code, for example, \"{ Input }\".\n"
        "2. Use downward arrows (\"| and v\") to indicate the flow from the input to the first function.\n"
        "3. Represent each function with square brackets, like \"[ function_name() ]\".\n"
        "4. Show the outputs of each function within curly braces. If a function has multiple outputs, separate them with spaces, "
        "such as \"{Output1} {Output2} ... {OutputN}\".\n"
        "5. If multiple outputs from one function are inputs to another function, use branching arrows (\"/\", \"|\", and \"\\\") "
        "to show the flow.\n"
        "6. Continue this representation to map the entire process, ending with the final output.\n"
        "\n"
        "Based on the input source code, determine the functions and their sequences, then construct the ASCII representation "
        "that clearly visualizes the flow from start to finish. The flow shall be written in plain code blocks."
        "\n[example]"
        "```"
        "{ Input Video }"
        "        |"
        "        v"
        "[ split_video() ]"
        "    / | \\"
        "{Part1} {Part2} {Part3} ... {PartN}"
        "    \ | /"
        "        v"
        "[ video_files2text_files() ]"
        "    / | \\"
        "{Text1} {Text2} {Text3} ... {TextN}"
        "    \ | /"
        "        v"
        "[ combine_text_files() ]"
        "```"
        "\n"
    )
    
    user_prompt = source_code
    return get_response_from_llm(system_prompt, user_prompt)
```

### generate_description

'generate_description' 함수는 주어진 파이썬 코드의 설명을 생성하는 함수입니다. 이 함수는 'function_name'과 'function_source' 두 개의 매개변수를 입력으로 받습니다. 'function_name'은 설명을 생성하려는 함수의 이름이고, 'function_source'는 해당 함수의 소스 코드입니다. 

함수 내부에서는, 먼저 'function_source'를 줄바꿈 문자('\n')로 연결하여 'source_code'를 생성합니다. 그 다음, 시스템 프롬프트인 'system_prompt'를 생성하는데, 이는 'function_name'과 'source_code'를 포함하는 설명 요청 문장입니다. 

또한, 사용자 프롬프트인 'user_prompt'도 생성하는데, 이는 함수의 소스 코드를 작성하지 말고 한국어로 함수를 설명하라는 요청 문장입니다. 

마지막으로, 'get_response_from_llm' 함수를 호출하여 'system_prompt'와 'user_prompt'를 입력으로 하여 생성된 설명을 반환합니다. 이 때, 설명은 한 단락으로 작성되어야 하며, 순서를 나타내야 하는 경우 번호 목록을, 특수 문자를 사용해야 하는 경우 평문 코드 블록을, 항목을 나열해야 하는 경우 불릿 포인트를 사용해야 합니다.

```python
def generate_description(function_name, function_source):
    source_code = '\n'.join(function_source)
    system_prompt = (f"Explain the function '{function_name}' in Korean from the Python code:\n\n{source_code}"
                     "Write the explanation as one paragraph. If you need to show the flow of order, use numbered lists."
                     "If you need to use special charactors, use plain code block"
                     "If you need to list out items, use bullet points.")
    user_prompt = "Describe this function for me in Korean. Don't write the function source code"
    return get_response_from_llm(system_prompt, user_prompt)
```

### extract_functions_from_file

'extract_functions_from_file' 함수는 주어진 파일 경로에서 함수를 추출하는 역할을 합니다. 먼저, 파일을 읽기 모드('r')로 열어 소스 코드를 읽어옵니다. 그 다음, 읽어온 소스 코드를 파이썬의 추상 구문 트리(ast)로 변환하여 파싱합니다. ast를 사용하면 코드를 트리 형태로 분석할 수 있습니다. 그 후 'ast.walk(node)'를 통해 추상 구문 트리를 순회하면서 각 노드가 함수 정의(ast.FunctionDef)인지 확인합니다. 만약 함수 정의 노드라면, 해당 함수의 이름과 소스 코드를 가져와서 리스트에 튜플 형태로 추가합니다. 이렇게 추출된 모든 함수와 원본 소스 코드를 반환합니다. 이 함수를 사용하면 특정 파이썬 파일에서 함수들만을 추출하여 분석하거나 다른 코드로 재사용하는 등의 작업이 가능해집니다.

```python
def extract_functions_from_file(file_path):
    with open(file_path, 'r') as f:
        source_code = f.read()
        node = ast.parse(source_code)
        functions = [(n.name, get_source_of_function(n.name, source_code)) for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
    return functions, source_code
```

### gen_content_body

'gen_content_body' 함수는 주어진 파일 경로에서 파이썬 소스 코드를 추출하고, 해당 코드의 전체 흐름, 함수 목록, 각 함수의 설명 및 소스코드를 가지고 있는 'content_body'를 생성하는 역할을 합니다. 먼저, 'extract_functions_from_file' 함수를 이용해 파일에서 함수와 소스코드를 추출합니다. 그 다음, 'generate_flow' 함수를 사용하여 소스코드의 전체 흐름을 생성하고, 이를 'content_body'에 추가합니다. 그 후, 추출한 각 함수에 대해 함수명을 'content_body'에 추가하고, 'generate_description' 함수를 이용해 각 함수의 설명을 생성합니다. 생성된 설명을 출력하고, 이를 'content_body'에 추가합니다. 추가적으로, 각 함수의 소스 코드를 파이썬 코드 블록 형식으로 만들어 'content_body'에 추가합니다. 마지막으로, 전체 소스코드를 'content_body'에 추가하고, 이를 반환합니다. 이 함수는 소스코드의 구조와 흐름을 이해하는 데 도움을 주는 문서를 생성하는데 사용될 수 있습니다.

```python
def gen_content_body(file_path):
    
    content_body = ""
    
    functions, source_code = extract_functions_from_file(file_path)
    
    content_body += "## 전체 흐름\n"
        
    content_body += generate_flow(source_code)
    
    content_body += "\n\n## 함수 목록\n"
    for func, _ in functions:
        content_body += f"- {func}\n"
        
    print(content_body)
    
    content_body += "\n## 함수 설명\n"
    for func, source in functions:
        
        print(f"Generating description for {func}...")

        description = generate_description(func, source)
        
        print(description)
        
        # Formatting the source code as a Python code block.
        formatted_source = '\n'.join(source)
    
        content_body += f"### {func}\n\n{description}\n\n```python\n{formatted_source}\n```\n\n"
    
    content_body += "\n## 전체 소스코드\n"
    content_body += f"```python\n{source_code}\n```\n"
    
    return content_body
```

### get_file_sha

'get_file_sha' 함수는 Python 코드에서 GitHub API를 사용하여 특정 저장소의 특정 파일의 SHA 값을 가져오는 기능을 수행합니다. 이 함수는 token, repo, path라는 세 개의 매개변수를 받습니다. token은 사용자 인증 정보를 포함한 토큰이며, repo는 저장소의 이름, path는 파일의 경로를 나타냅니다. 함수는 먼저 GitHub API의 URL을 만든 다음, 이 URL에 대한 요청을 보내고 응답을 받습니다. 이때 인증 토큰을 헤더에 포함하여 요청을 보냅니다. 이후 서버로부터 받은 응답의 상태 코드가 200, 즉 요청이 성공적으로 이루어졌을 경우, 응답 내용 중 'sha' 항목의 값을 반환합니다. 만약 응답 상태 코드가 200이 아닐 경우, 함수는 None을 반환합니다.

```python
def get_file_sha(token, repo, path):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("sha")
    return None
```

### create_or_update_github_file

'create_or_update_github_file' 함수는 주어진 정보를 바탕으로 Github에 파일을 생성하거나 업데이트하는 기능을 합니다. 인자로는 토큰, 리포지토리 이름, 날짜, 주제, 본문 내용이 필요합니다. 먼저, 날짜를 문자열로 바꾸고 주제를 파일 이름으로 사용합니다. 그리고 이 정보를 바탕으로 파일 경로, 제목, 이미지 URL을 생성합니다. 다음으로 Github API에 전송할 요청 URL과 헤더를 설정합니다. 헤더에는 인증 정보와 Github API 버전 정보가 포함됩니다. 그 다음으로 파일 내용을 설정하고 base64로 인코딩합니다. 만약 이미 해당 파일이 존재한다면, 해당 파일의 SHA 값을 가져와 요청 데이터에 추가합니다. 이후 이 요청을 Github API에 전송하여 파일을 생성하거나 업데이트합니다. 요청 결과에 따라 성공 메시지 또는 실패 메시지를 출력합니다. 마지막으로 작성한 내용을 로컬 파일 'saved_post.md'에 저장합니다.

```python
def create_or_update_github_file(token, repo, date, topic, content_body):
    
    print(token)
    date_str = date.strftime('%Y-%-m-%-d')
    
    # 파일 이름 및 메타데이터 파싱
    filename = f"{date_str}-{topic.replace(' ', '_').lower()}.md"
    path = f"_posts/{filename}"
    title = topic
    image_url = f"http://tykimos.github.io/warehouse/{filename.replace('.md', '_title.jpeg')}"
    
    # 기본 URL 및 인증 설정
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 파일 내용 설정
    content = f"""---
layout: post
title: "{title}"
author: Taeyoung Kim
date: {date_str} 00:00:00
categories: bic
comments: true
image: {image_url}
---

{content_body}
"""

    # 파일의 내용을 base64로 인코딩
    encoded_content = base64.b64encode(content.encode()).decode()

    # 파일의 SHA 값을 가져오기
    sha = get_file_sha(token, repo, path)
    
    # 요청 데이터 구성
    data = {
        "message": f"Add {filename}",
        "content": encoded_content
    }

    # 만약 sha 값이 있다면, data에 추가 (기존 파일 수정을 위함)
    if sha:
        data["sha"] = sha

    print("github api request")
    print(url)
    print(headers)
    print(data)
    
    response = requests.put(url, headers=headers, json=data)

    if response.status_code == 201:
        print(f"Successfully created or updated file at {url}")
    elif response.status_code == 200:
        print(f"Successfully updated file at {url}")
    else:
        print(f"Failed to create or update file. Response: {response.text}")

    with open("saved_post.md", 'wt') as f:
        f.write(content)
```


## 전체 소스코드
```python
import requests
import base64
from datetime import datetime  # datetime 모듈을 import합니다.
import sys
import ast
import openai
import os

# openai.api_key = sk-???

def get_source_of_function(function_name, source_code):
    node = ast.parse(source_code)
    for n in ast.walk(node):
        if isinstance(n, ast.FunctionDef) and n.name == function_name:
            return source_code.splitlines()[n.lineno-1:n.end_lineno]
    return ""

def get_response_from_llm(system_prompt, user_prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    return response['choices'][0]['message']['content']

def generate_flow(source_code):
    
    system_prompt = (
        "Given a source code as input, your task is to analyze and extract the functions and their resulting outputs. "
        "Then, visualize the relationship between these functions and their outputs using an ASCII art representation as following [example]. Then, explain the your flow in Korean. You should write in Korean except for visualization.\n"
        "\n"
        "Steps for visualization:\n"
        "1. Begin with the primary input of the source code, for example, \"{ Input }\".\n"
        "2. Use downward arrows (\"| and v\") to indicate the flow from the input to the first function.\n"
        "3. Represent each function with square brackets, like \"[ function_name() ]\".\n"
        "4. Show the outputs of each function within curly braces. If a function has multiple outputs, separate them with spaces, "
        "such as \"{Output1} {Output2} ... {OutputN}\".\n"
        "5. If multiple outputs from one function are inputs to another function, use branching arrows (\"/\", \"|\", and \"\\\") "
        "to show the flow.\n"
        "6. Continue this representation to map the entire process, ending with the final output.\n"
        "\n"
        "Based on the input source code, determine the functions and their sequences, then construct the ASCII representation "
        "that clearly visualizes the flow from start to finish. The flow shall be written in plain code blocks."
        "\n[example]"
        "```"
        "{ Input Video }"
        "        |"
        "        v"
        "[ split_video() ]"
        "    / | \\"
        "{Part1} {Part2} {Part3} ... {PartN}"
        "    \ | /"
        "        v"
        "[ video_files2text_files() ]"
        "    / | \\"
        "{Text1} {Text2} {Text3} ... {TextN}"
        "    \ | /"
        "        v"
        "[ combine_text_files() ]"
        "```"
        "\n"
    )
    
    user_prompt = source_code
    return get_response_from_llm(system_prompt, user_prompt)

# Generate a description for a given function
def generate_description(function_name, function_source):
    source_code = '\n'.join(function_source)
    system_prompt = (f"Explain the function '{function_name}' in Korean from the Python code:\n\n{source_code}"
                     "Write the explanation as one paragraph. If you need to show the flow of order, use numbered lists."
                     "If you need to use special charactors, use plain code block"
                     "If you need to list out items, use bullet points.")
    user_prompt = "Describe this function for me in Korean. Don't write the function source code"
    return get_response_from_llm(system_prompt, user_prompt)

# Extract all functions from a Python file
def extract_functions_from_file(file_path):
    with open(file_path, 'r') as f:
        source_code = f.read()
        node = ast.parse(source_code)
        functions = [(n.name, get_source_of_function(n.name, source_code)) for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
    return functions, source_code

def gen_content_body(file_path):
    
    content_body = ""
    
    functions, source_code = extract_functions_from_file(file_path)
    
    content_body += "## 전체 흐름\n"
        
    content_body += generate_flow(source_code)
    
    content_body += "\n\n## 함수 목록\n"
    for func, _ in functions:
        content_body += f"- {func}\n"
        
    print(content_body)
    
    content_body += "\n## 함수 설명\n"
    for func, source in functions:
        
        print(f"Generating description for {func}...")

        description = generate_description(func, source)
        
        print(description)
        
        # Formatting the source code as a Python code block.
        formatted_source = '\n'.join(source)
    
        content_body += f"### {func}\n\n{description}\n\n```python\n{formatted_source}\n```\n\n"
    
    content_body += "\n## 전체 소스코드\n"
    content_body += f"```python\n{source_code}\n```\n"
    
    return content_body

def get_file_sha(token, repo, path):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("sha")
    return None

def create_or_update_github_file(token, repo, date, topic, content_body):
    
    print(token)
    date_str = date.strftime('%Y-%-m-%-d')
    
    # 파일 이름 및 메타데이터 파싱
    filename = f"{date_str}-{topic.replace(' ', '_').lower()}.md"
    path = f"_posts/{filename}"
    title = topic
    image_url = f"http://tykimos.github.io/warehouse/{filename.replace('.md', '_title.jpeg')}"
    
    # 기본 URL 및 인증 설정
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # 파일 내용 설정
    content = f"""---
layout: post
title: "{title}"
author: Taeyoung Kim
date: {date_str} 00:00:00
categories: bic
comments: true
image: {image_url}
---

{content_body}
"""

    # 파일의 내용을 base64로 인코딩
    encoded_content = base64.b64encode(content.encode()).decode()

    # 파일의 SHA 값을 가져오기
    sha = get_file_sha(token, repo, path)
    
    # 요청 데이터 구성
    data = {
        "message": f"Add {filename}",
        "content": encoded_content
    }

    # 만약 sha 값이 있다면, data에 추가 (기존 파일 수정을 위함)
    if sha:
        data["sha"] = sha

    print("github api request")
    print(url)
    print(headers)
    print(data)
    
    response = requests.put(url, headers=headers, json=data)

    if response.status_code == 201:
        print(f"Successfully created or updated file at {url}")
    elif response.status_code == 200:
        print(f"Successfully updated file at {url}")
    else:
        print(f"Failed to create or update file. Response: {response.text}")

    with open("saved_post.md", 'wt') as f:
        f.write(content)

if __name__ == "__main__":
    if len(sys.argv) < 3:  # 필요한 인자가 충분하지 않을 경우
        print("Error: Missing topic argument.")
        print("Usage: python gen_post_source_code.py <TOPIC> <SOURCECODE>")
        sys.exit(1)  # 프로그램 종료, 1은 일반적인 에러 코드입니다.
        
    # 예제로 실행
    token = os.environ["GITHUB_TOKEN_KEY"]  # GitHub에서 발급받은 토큰을 여기에 넣습니다.
    
    repo = "tykimos/tykimos.github.io"

    topic = sys.argv[1]

    content_body = gen_content_body(sys.argv[2])
    
    create_or_update_github_file(token, repo, datetime.today(), topic, content_body)
```

