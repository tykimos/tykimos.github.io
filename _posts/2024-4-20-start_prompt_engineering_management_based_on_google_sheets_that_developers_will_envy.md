---
layout: post
title: "개발자 부럽지 않은 챗GPT 프롬프트 엔지니어링 구글시트 기반으로 관리 시작하기"
author: Taeyoung Kim
date: 2024-4-20 19:04:31
categories: llm
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_title_001.jpg
---

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_title_001.jpg)

프롬프트 엔지니어링은 GPT-3, GPT-4 등 다양한 모델들이 사용자의 명령을 이해하고 적절한 응답을 내놓도록 하는 핵심적인 과정입니다. 프롬프트 엔지니어링을 잘 활용하는 사용자들은 자신만의 독특한 업무를 수행시키거나, 독창적인 콘텐츠를 생성할 수 있습니다.

프롬프트를 관리하면 효율적으로 프롬프트 엔지니어링을 수행할 수 있습니다. 이를 위해 워드나 노션 등 다양한 도구를 활용하는 경우가 많습니다. 하지만 이번 포스트에서는 노션이나 워드 대신 구글 시트나 엑셀을 활용하는 방법에 대해 알아보겠습니다.

## 구글 시트의 열 구분자

구글 시트를 활용하면 프롬프트 내용을 셀로 구분하여 체계적으로 기록할 수 있습니다. 

구글 시트를 활용하려면 먼저 열 구분자의 개념을 이해해야 합니다. 열 구분자는 각 열이 어떤 정보를 담고 있는지 구분해주는 역할을 합니다. 예를 들어, "이름", "나이", "성별" 등의 헤더가 있는 열이 있다면, 각각의 열은 해당 헤더의 정보를 담고 있는 것입니다.

구글 시트에서는 이러한 열 구분자를 쉽게 설정할 수 있습니다. 먼저, 헤더를 정의하고, 각 행에 해당하는 정보를 입력하면 됩니다. 이렇게 하면, 시트의 각 열은 특정 정보를 담고 있는 열로 정의되어, 챗GPT 등의 모델에게 명확한 정보를 전달할 수 있습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_001.jpg)


![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_002.jpg)


## SNS 게시물 태깅 태스트 예시

SNS 게시물을 태깅하는 작업을 예로 들어보겠습니다. SNS 게시물을 분류하거나, 특정 주제에 대한 게시물을 찾는 작업에는 태깅이 큰 도움이 됩니다.

구글 시트를 이용하면, 각 게시물에 대한 정보와 그에 해당하는 태그를 쉽게 관리할 수 있습니다. 예를 들어, "게시물 내용"과 "태그" 두 열을 만들고, 각 게시물에 대한 정보와 태그를 입력하면 됩니다. 이렇게 하면, 챗GPT가 각 게시물을 적절하게 분류하거나, 사용자가 원하는 주제의 게시물을 찾는 데 도움을 줄 수 있습니다.

```markdown
| 게시물 내용                           | 태그              |
|-------------------------------------|-------------------|
| 오늘은 맛있는 피자를 먹었어요.        | #피자 #맛집 #음식  |
| 새로운 책을 읽기 시작했습니다.        | #독서 #책 #문화생활 |
| 주말에 등산을 다녀왔습니다.          | #등산 #여행 #운동  |
```
![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_003.jpg)



## 챗GPT 결과 확인

구글 시트에 프롬프트를 입력한 후, 이를 챗GPT에게 전달하면, 챗GPT는 이에 대한 적절한 응답을 생성합니다. 챗GPT의 응답은 사용자가 입력한 프롬프트에 대한 모델의 이해도와 관련이 있습니다. 따라서, 프롬프트 엔지니어링이 잘 되어 있다면, 챗GPT는 사용자의 의도에 맞는 응답을 내놓을 것입니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_004.jpg)


이렇게 하면, 사용자는 복잡한 코드를 작성하지 않고도, 간단히 구글 시트를 이용하여 챗GPT 등의 모델을 활용할 수 있습니다. 또한, 구글 시트는 클라우드 기반으로 동작하기 때문에 언제 어디서나 접근이 가능하며, 여러 사람이 동시에 작업을 할 수 있어 협업에도 효과적입니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_005.jpg)


프롬프트 엔지니어링은 챗봇 모델을 활용하는 데 있어 중요한 요소입니다. 이를 잘 활용하면, 복잡한 알고리즘을 직접 구현하지 않아도, 사용자의 의도에 맞는 챗봇 응답을 생성할 수 있습니다. 이번 포스트를 통해 구글 시트를 활용한 프롬프트 엔지니어링 방법에 대해 알아보았습니다. 이 방법을 활용하여, 여러분만의 독특한 챗봇을 만들어 보세요.

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_006.jpg)
