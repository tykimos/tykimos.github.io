---
layout: post
title: "OpenAI API을 처음으로 사용해보기"
author: 김태영
date: 2023-8-29 00:00:30
categories: ai
comments: true
---

$OPENAI_API_KEY에 발급 받은 키 정보를 삽입합니다.

```
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

![img](http://tykimos.github.io/warehouse/2023-8-29-openai_api_usage_1.png)

https://reqbin.com/curl 에 접속하여, 위 코드를 복사 후 실행시킵니다. 정상적으로 처리된다면, 그 결과 아래와 같이 응답을 받을 수 있습니다. 

```
{
    "id": "chatcmpl-7sk3WWQ6Kb6dcDGgomtCIHMX20crd",
    "object": "chat.completion",
    "created": 1693281106,
    "model": "gpt-3.5-turbo-0613",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hi there! How can I assist you today?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 19,
        "completion_tokens": 10,
        "total_tokens": 29
    }
}

![img](http://tykimos.github.io/warehouse/2023-8-29-openai_api_usage_2.png)
```
