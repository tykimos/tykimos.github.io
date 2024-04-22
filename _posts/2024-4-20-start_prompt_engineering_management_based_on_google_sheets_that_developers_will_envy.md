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

프롬프트 엔지니어링은 GPT-3, GPT-4 등 다양한 모델들이 사용자의 명령을 이해하고 적절한 응답을 내놓도록 하는 핵심적인 과정입니다. 프롬프트 엔지니어링을 잘 활용하는 사용자들은 자신만의 방법으로 프롬프트을 관리합니다. 워드나 노션 그리고 전용 툴 등 다양한 도구들이 사용되지만 이번 포스트에서는 엑셀이나 구글 시트를 활용하는 방법에 대해서 알아보겠습니다.

### 프롬프트 엔지니어링

LLM(Large Language Model)은 인컨텍스트 러닝을 통해 사용자의 지시에 따른 응답을 생성할 수 있습니다. 제한된 인컨텍스트 내에서 효율적인 프롬프트를 작성하기 위해서 여러 기법을 적용하는 것을 "프롬프트 엔지니어링"이라고 합니다. 프롬프트 엔지니어링 기법은 나름 정형화되어 있기 때문에 아래 처럼 셀로 나누어 작성을 할 수 있습니다. 파란색 셀은 롤 프롬프팅과 지시문이 작성되어 있고, 형광색 셀에는 예시인 퓨샷 프롬프팅이 기록되어 있습니다. 

* 롤 프롬프팅(Role Prompting): 이 기법은 모델에 특정 역할을 부여하여 응답을 유도하는 방식입니다. 예를 들어, 모델에게 비평가, 조언자, 교수 등의 역할을 부여하여 그에 맞는 답변을 유도할 수 있습니다.
* 퓨샷 프롬프팅(Few-shot Prompting): 이 기법은 모델에게 몇 가지 예제를 보여줌으로써 특정 작업을 수행하는 방법을 "학습"시키는 방식입니다. 예제를 통해 모델은 작업의 맥락과 요구 사항을 인식하고 비슷한 문제에 적용할 수 있습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_000.jpg)

### 구글 시트에서의 열구분

구글 시트 내용을 그대로 챗GPT에 복사할 경우 제대로 인식될 수 있는 지 체크해보겠습니다. 챗GPT는 텍스트로만 복사붙이기가 되기 때문에 표가 유지되지는 않지만 "탭" 기호로 구분되어 있습니다. 아래 그림은 제대로 구분이 되는 지 테스트한 결과입니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_001.jpg)

구글 시트의 셀을 복사하여 챗GPT 입력창에 붙인 후 확인한 결과 행은 개행으로 구분되고, 열은 탭 기호로 구분되어 있습니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_002.jpg)

그럼 다시 마크다운 테이블로 복원해보겠습니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_003.jpg)

정상적으로 표가 표시되는 것을 확인할 수 있습니다. 

## SNS 게시물 해시태그 생성 태스크

인스타그램, 페이스북, 트위터 등 SNS 게시물을 작성할 때 적절한 해시태그를 붙이는 것이 중요합니다. 기존에 태그를 붙인 패턴 및 스타일을 기반으로 새로운 게시물에 대한 해시태그를 생성하는 간단한 예시를 만들어봤습니다. 먼저 구글 시트로 만들겠습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_004.jpg)

이 구글시트에는 시스템 프롬프트와 예시 그리고 마지막에 (작성)이라고 표기하여 이 부분에 새로운 해시태그를 작성하도록 하였습니다. 

## 챗GPT 결과 확인

구글 시트에서 복사붙이기할 때 행과 열이 구분된다는 것을 앞서 확인하였습니다. 그럼 셀 내용을 그대로 채팅 입력창에 붙여보겠습니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_005.jpg)

그 결과 아래처럼 해시태그가 작성됩니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-20-start_prompt_engineering_management_based_on_google_sheets_that_developers_will_envy_006.jpg)

이런 방식을 활용하면 프롬프트를 체계적으로 관리할 수 있을 뿐만 아니라 복잡한 코드를 작성하지 않고도, 간단히 구글 시트를 이용하여 챗GPT 등의 모델을 쉽게 활용할 수 있습니다. 