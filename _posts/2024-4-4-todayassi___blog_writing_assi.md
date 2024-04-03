---
layout: post
title: "오늘의어시 - 블로그 작성 어시"
author: Taeyoung Kim
date: 2024-4-4 00:55:18
categories: llm, rpa, blog, chatgpt
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_title.jpeg
---

본 내용은 (어시+랭체인)에 의해 자동으로 작성된 글입니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_title.jpeg)
# GPT4와 함께하는 자동 블로그 포스팅 시스템 

안녕하세요, 오늘은 구글폼을 활용해 기본적인 블로그 내용을 입력하면, 그 내용을 토대로 GPT4가 상세한 블로그를 작성하고, 그 결과를 깃헙 페이지에 자동으로 업로드하는 시스템을 소개하려고 합니다. 이 기능은 AI를 활용해 블로그 작성의 효율성을 크게 향상시킬 수 있는 매우 흥미로운 시스템입니다. 

## 동작 과정

시스템의 동작 과정을 간단하게 설명하면 다음과 같습니다.

1. 사용자가 구글폼을 통해 블로그의 제목과 기본 내용을 입력합니다.
2. 입력한 내용을 바탕으로 GPT4가 상세한 블로그 내용을 생성합니다.
3. GPT4가 생성한 블로그 내용을 깃헙 페이지에 자동으로 업로드합니다.

이 시스템을 통해 사용자는 복잡한 블로그 작성 과정을 거치지 않고도, 손쉽게 자신의 아이디어를 공유할 수 있게 됩니다. 

## GPT4의 활용

GPT4는 OpenAI에서 개발한 최신 인공지능 언어 모델로, 자연어 처리 능력이 매우 뛰어나다는 것으로 알려져 있습니다. 이러한 특성을 활용해, 사용자가 입력한 기본 내용을 토대로 상세하고 풍부한 블로그 내용을 생성하는 것입니다.

```python
from transformers import GPT4Tokenizer, GPT4Model
tokenizer = GPT4Tokenizer.from_pretrained('gpt4')
model = GPT4Model.from_pretrained('gpt4')

def generate_blog(title, draft):
    inputs = tokenizer.encode(title + draft, return_tensors='pt')
    outputs = model.generate(inputs, max_length=1000, temperature=0.7)
    return tokenizer.decode(outputs[0])
```

## 깃헙 페이지 업로드

GPT4가 생성한 블로그 내용은 깃헙 페이지에 자동으로 업로드됩니다. 깃헙 페이지는 개발자들에게 많이 사용되는 무료 웹 호스팅 서비스로, 이를 통해 블로그를 손쉽게 공개할 수 있습니다. 

```python
import git
repo = git.Repo('path/to/repo')
file = open("new_post.md", "w")
file.write(generate_blog(title, draft))
file.close()
repo.git.add('new_post.md')
repo.git.commit('-m', 'Add new post')
repo.git.push()
```

## 마무리

이상으로 GPT4와 함께하는 자동 블로그 포스팅 시스템에 대한 소개를 마치겠습니다. 이 시스템을 통해 블로그 작성의 부담을 덜고, 더욱 풍부하고 다양한 내용을 공유할 수 있게 되었으면 좋겠습니다. 감사합니다.
