---
layout: post
title: "오늘의어시 - 블로그 작성 어시"
author: Taeyoung Kim
date: 2024-4-4 00:55:18
categories: llm, rpa, blog, chatgpt
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_title.jpeg
---

이번 포스트에서는 구글폼을 활용해 기본적인 블로그 내용을 입력하면, 그 내용을 토대로 GPT4가 상세한 블로그를 작성하고, 그 결과를 깃헙 페이지에 자동으로 업로드하는 어시체인을 소개하려고 합니다. 이 어시를 통해서 블로그 작성의 효율성을 향상시킬 수 있습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_title.jpeg)

### 동작 과정

어시체인의 동작 흐름은 다음과 같습니다.

1. 사용자가 구글폼을 통해 블로그의 제목과 기본 내용을 입력합니다.
2. 입력한 내용을 바탕으로 GPT4가 상세한 블로그 내용을 생성합니다.
3. GPT4가 생성한 블로그 내용을 깃헙 페이지에 자동으로 업로드합니다.

이 어시체인을 통해 손쉽게 자신의 아이디어를 블로그에 공유할 수 있게 됩니다. 

### 구글폼과 구글시트

블로그 작성은 구글폼을 통해서 쉽게 작성을 할 수 있습니다. 제목과 기초내용을 입력할 수 있으며, 타이틀 이미지도 파일 업로드 기능을 통해서 등록할 수 있습니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_1.jpg)

구글폼의 응답은 구글시트로 관리할 수 있습니다. 응답 구글시트에는 사용자가 별도로 열을 추가할 수 있는데요. 어시를 통해서 관리할 수 있도록 처리흐름의 열을 추가합니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_2.jpg)

따라서 준비해야할 파일은 아래와 같습니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_3.jpg)

- Tykimos Blog : 구글폼
- Tykimos Blog(응답) : 구글시트 (구글폼에 의해서 생성)
- Tykimos Blog (File responses) : 파일업로드 폴더 (구글폼에 의해서 생성)
- Blog Writing Assi : 어시를 구동시키기 위한 코랩

## 코랩과 상태 흐름

구글폼을 입력했을 때 자동으로 포스팅이 되기 위해서는 별도의 서버가 동작하면서 모니터링을 하면 되는데요. 이 경우 별도의 서버가 필요합니다. 서버 운영에는 비용 및 관리 부담이 있기 때문에 서버없이 일시적으로 실행시키면 그동안 등록된 요청을 처리하도록 코랩을 만들었습니다.

아래 코드는 구글 시트를 데이터프레임으로 읽었다가 처리해야할 요청이 있는 지 확인하고, 있다면 처리를 수행합니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_4.jpg)

현재 블로그 작성 어시에서는 다음 두 가지의 처리를 수행하고 있습니다.

- blog_content_generation : 기초 내용으로 상세한 블로그 내용을 작성합니다.
- github_upload : 작성한 블로그 내용을 깃헙에 업로드 합니다.

어시는 시트를 데이터베이스처럼 활용하여 아래 3가지 상태열로 관리합니다.

- fetching : 어떤 처리를 접수하였다면, 이 열에 "1"을 기입합니다.
- running : 접수한 처리를 시작하였다면 이 열에 "1"을 기입합니다.
- done : 처리를 완료하였다면 이 열에 리포트 파일명을 기입합니다.

리포트 파일명은 JSON 형태로 되어 있으며, 파일명은 처리 이름과 UUID 4자리로 구성되어 있습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_5.jpg)

리포트 파일은 임시적으로 사용되며, 다음 처리에 필요한 정보를 제공합니다.

## 시트 모니터링

모든 요청에 대한 처리를 완료할 때까지 구글 시트 정보를 가지고 오며, 처리 진행 단계에 따라 구글 시트를 업로드 합니다. 이에 필요한 기능 함수는 아래와 같습니다.

```python

import gspread

client = gspread.authorize(creds)

# 구글 시트 열기
worksheet = client.open(SHEET_NAME).sheet1

# 첫 번째 행에서 열 이름을 찾아 해당 열 인덱스를 얻기
def get_column_index(column_name):
    first_row = worksheet.row_values(1)  # 첫 번째 행의 값 가져오기
    return first_row.index(column_name) + 1  # 열 번호는 1부터 시작하므로 1을 더해야 합니다.

def update_sheet_status(row_index, column_name, value):
    """시트의 상태를 업데이트하는 함수"""
    column_index = get_column_index(column_name)
    worksheet.update_cell(row_index + 2, column_index, value)
```

## Blog Content Generation Chain

GPT4가 생성한 블로그 내용은 깃헙 페이지에 자동으로 업로드됩니다. 깃헙 페이지는 개발자들에게 많이 사용되는 무료 웹 호스팅 서비스로, 이를 통해 블로그를 손쉽게 공개할 수 있습니다. 

```python
system_prompt = """Given the initial draft content provided by the user, generate a comprehensive and engaging blog post body that expands upon the draft with additional insights, explanations, and related content.
Make sure to structure the post in a way that is informative and engaging to the reader.
Incorporate the provided draft content seamlessly into the narrative, enhancing it with creative elements and factual information where appropriate.
The goal is to create a cohesive and compelling narrative that captures the reader's interest and provides valuable information on the topic.
All content must be written in Korean.
Please write the blog content in markdown format, tailored for a technical blog, including code snippets, bullet points, and headers to ensure clarity and enhance readability. Do not invent or create arbitrary source code or information; only expand upon and provide detailed and friendly explanations based on the information provided by the user.

[예시] == 시작 ==

입력: 구글폼을 통해서 제목과 기초 내용을 간단히 입력하면, 기초 내용을 바탕으로 GPT4가 상세 블로그를 작성 한 후, 깃헙 페이지에 자동으로 업로드를 수행합니다. 
출력:


[예시] == 시작 ==

"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_prompt),
    ("human", "{draft_content}")]
)

chain = (prompt_template
        | chat_model
        | StrOutputParser())

gen_content = chain.invoke({"draft_content" : draft_content})
```

## 마무리

이상으로 GPT4와 함께하는 자동 블로그 포스팅 시스템에 대한 소개를 마치겠습니다. 이 시스템을 통해 블로그 작성의 부담을 덜고, 더욱 풍부하고 다양한 내용을 공유할 수 있게 되었으면 좋겠습니다. 감사합니다.
