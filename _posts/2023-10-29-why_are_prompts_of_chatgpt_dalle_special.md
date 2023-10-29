---
layout: post
title:  "챗GPT가 만드는 DALL·E 프롬프트는 왜 특별한가?"
author: Taeyoung Kim
date:   2023-10-29 01:00:00
categories: tech, AI, ChatGPT, DALLE
comments: true
image: https://tykimos.github.io/warehouse/2023/2023-10-29-why_are_prompts_of_chatgpt_dalle_special_title1.jpg
---

# 1. 챗GPT가 만드는 DALL·E 프롬프트는 왜 특별한가?

본 글에서는 챗GPT가 만드는 DALL·E 프롬프트의 특별함과 그 과정에서 사용되는 기술과 접근 방법에 대해 다뤄보겠습니다. 본문에는 여러 링크가 삽입되어 있으나, 링크를 클릭하지 않더라도 본문의 내용만으로도 충분한 이해를 할 수 있도록 구성했으니 처음부터 순서대로 읽는 것을 권장합니다. 

1. 챗GPT가 만드는 DALL·E 프롬프트는 왜 특별한가? - 현재 페이지
2. [챗GPT는 DALL·E에 어떤 명령을 내리는가?](https://tykimos.github.io/2023/10/29/what_commands_does_chatgpt_give_to_dalle/)
3. [챗GPT가 시드를 알려줄 수 있는 이유는?](https://tykimos.github.io/2023/10/29/why_can_chatgpt_provide_the_seed/)
4. [DALL·E 이미지 재현과 튜닝을 위해 챗GPT로 시드 관리하기](https://tykimos.github.io/2023/10/29/managing_seeds_with_chatgpt_for_dalle_image_reproduction_and_tuning/)
5. [챗GPT가 GPT-4와 DALL·E랑 주고받는 비밀 메시지들](https://tykimos.github.io/2023/10/29/secret_messages_exchanged_between_chatgpt_gpt-4_and_dalle/)
6. [DALL·E 연동을 위한 챗GPT의 시스템 프롬프트 전문](https://tykimos.github.io/2023/10/29/system_prompts_specifically_for_dalle-integrated_chatgpt)

![img](https://tykimos.github.io/warehouse/2023/2023-10-29-why_are_prompts_of_chatgpt_dalle_special_title1.jpg)

## 1.1. 이미지 생성형 AI의 작업 흐름

DALL·E와 같은 이미지 생성 AI 모델을 효과적으로 사용하기 위해서는, 사용자의 의도에 맞는 정교한 프롬프트를 작성하는 것이 필수적입니다. 이러한 프롬프트 작성 과정을 ‘프롬프트 엔지니어링’이라고 합니다. 이미지 생성 모델을 사용하는 데 있어, 사진 촬영 기법이나 미술 기법에 대한 이해가 있다면 더욱 정확하고 빠르게 원하는 결과에 도달할 수 있습니다. 하지만 전문 지식이 없는 사용자라도 아래와 같은 일련의 과정을 통해 이미지 생성 모델을 활용할 수 있습니다.

* 프롬프트 생성: 다양한 기법을 사용하여 프롬프트를 구체적이고 상세하게 작성합니다.
* 프롬프트 튜닝: 생성된 이미지를 확인하고 프롬프트를 재조정하여 더 나은 결과를 얻습니다.

이 두 과정은 원하는 결과를 얻을 때까지 반복적으로 이루어집니다. 하지만 이 과정은 쉽지 않으며, 상당한 시간과 노력이 소요될 수 있습니다. 프롬프트를 조금만 수정해도 완전히 다른 이미지가 생성될 수 있기 때문에, 미세한 조정이 어렵습니다. 이를 해결하기 위해서는 ‘시드’값을 고정하는 것이 필요합니다. 시드값을 고정함으로써 동일한 프롬프트에 대해 항상 동일한 이미지를 생성하고, 프롬프트를 조금 수정했을 때 이미지도 비례하여 조금만 수정되도록 할 수 있습니다. 시드에 대해서는 "챗GPT가 시드를 알려줄 수 있는 이유는?"에서 설명 드리겠습니다.

## 1.2. 챗GPT-DALL·E

챗GPT를 활용하면 프롬프트 생성 및 튜닝 과정을 보다 쉽게 진행할 수 있습니다. 이를 가능하게 하는 것이 바로 챗GPT-DALL·E 서비스 입니다. 이 서비스에는 사용자의 요청을 효과적으로 처리하기 위한 미리 정의된 지시문이 존재합니다. 이 지시문을 바탕으로 어떻게 프롬프트를 생성하는 지 알아보겠습니다.

[![](https://mermaid.ink/img/pako:eNqrVkrOT0lVslJKy8kvT85ILCpR8AmKyVMAAkeNN01r3sxa-WbeBE0FXV27mlcbWl53r3g7dY7Cm5Ylr1e11ig4abzZON09IEQTosMJpEyh5s3cLa_X73izvEHhTfPcNy0bFd5OaXm9cA2QfNu1o8ZZw8XRx-fQdleoJmc0TTVO2A2rcYzJU9JRyk0tyk3MTAG6uBqkLkapJCM1NzVGyQrIzEstLSlKzIlRismrBSpNLC3JD67MS1ayKikqTdVRKi1ISSxJdclMTC9KzFWySkvMKQaKpqZkluQX-UJCARwYtQBB8XIX?type=png)](https://mermaid.live/edit#pako:eNqrVkrOT0lVslJKy8kvT85ILCpR8AmKyVMAAkeNN01r3sxa-WbeBE0FXV27mlcbWl53r3g7dY7Cm5Ylr1e11ig4abzZON09IEQTosMJpEyh5s3cLa_X73izvEHhTfPcNy0bFd5OaXm9cA2QfNu1o8ZZw8XRx-fQdleoJmc0TTVO2A2rcYzJU9JRyk0tyk3MTAG6uBqkLkapJCM1NzVGyQrIzEstLSlKzIlRismrBSpNLC3JD67MS1ayKikqTdVRKi1ISSxJdclMTC9KzFWySkvMKQaKpqZkluQX-UJCARwYtQBB8XIX)

### 1.2.1. 프롬프트 생성

사용자는 자신의 요구사항을 챗GPT에게 간단히 전달합니다. 챗GPT는 이 요구사항을 바탕으로 구체적이고 상세한 프롬프트를 생성하여 DALL·E에 전달합니다. 그럼 어떤 기준으로 이렇게 바꿔질까요? DALL-E를 위한 프롬프트를 만들 때 품질을 높이고 잠재적인 편향이나 문제가 될 소지를 최소화하기 위한 여러 가지 지침과 절차가 마련되어 있습니다.

#### 1.2.1.1. 품질 향상을 위한 세부 가이드라인

##### 세부적인 설명

사용자가 작성한 요청을 가능한 한 상세하게 재작성합니다. 이를 통해 생성된 이미지가 사용자의 의도와 밀접하게 일치하게 됩니다. 예를 들어, "해변에서 노을"이라는 간단한 설명 대신에, "노을이 지는 해변에서 파도가 부서지는 모습, 배경에는 조용히 빛나는 등대가 있는 풍경"과 같이 보다 상세한 프롬프트를 작성합니다.

##### 다양한 캡션 생성

사용자가 특정 이미지 수를 요청하지 않은 경우, 가능한 한 다양하게 작성된 2개의 캡션을 생성합니다. 이를 통해 이미지의 다양성을 높일 수 있습니다.

##### 이미지 유형 명시

캡션의 시작 부분에서 이미지의 유형을 명시적으로 언급합니다. 예를 들어, "사진을 촬영한 노을이 지는 해변" 또는 "유화 스타일의 해변에서의 노을"과 같이 특정합니다. 이를 통해 생성될 이미지에 대한 명확한 스타일을 설정할 수 있습니다.

##### 해상도 명세

요청된 이미지의 해상도를 명확히 명시하여, 사용자의 요청한 사이즈에 부합되도록 합니다.

#### 1.2.1.2. 편향 및 잠재적 문제 최소화

##### 이미지 생성 제한

사용자가 많은 수의 이미지를 요청하더라도, 우리는 2개 이상의 이미지를 생성하지 않습니다. 이는 잠재적으로 문제가 될 수 있는 내용의 과도한 생성을 방지하기 위함입니다.

##### 특정 인물 피하기

정치인이나 다른 공공 인물의 이미지는 생성하지 않습니다. 대신에, 사용자에게 다른 창의적인 아이디어를 제공하고 추천합니다.

##### 예술가 스타일 제한

최근 100년 이내에 창작된 작품을 가진 예술가의 스타일로는 이미지를 생성하지 않습니다. 이는 현대 예술가들의 저작권 및 지적 재산권을 존중하기 위함입니다.

##### 다양한 대표

우리는 이미지 속 인물을 다양하게 묘사하며, 각 인물의 출신과 성별을 명확하게 지정합니다. 이는 포괄성을 증진하고 고정관념을 방지하는 데 도움이 됩니다.

##### 특정 참조 수정

특정인이나 유명인을 언급하는 설명이 포함된 경우, 우리는 이러한 참조를 일반적인 설명으로 치환하여 신중하게 수정합니다. 이는 의도하지 않은 신원 공개를 방지하기 위함입니다.

##### 모욕적인 내용 피하기

모욕적일 수 있는 이미지는 생성하지 않습니다. 우리는 생성된 콘텐츠가 모든 사용자에게 존중되고 배려되도록 최선을 다합니다.

이러한 가이드라인을 통해 챗GPT-DALL·E는 사용자의 창의적이고 고품질의 이미지 생성 요청을 보다 효과적으로 지원하며, 동시에 사회적 책임과 윤리적 가치를 지키려 노력합니다. 위에 언급한 가이드라인은 모두 챗GPT-DALL·E에 설정된 시스템 프롬프트에 의해 동작이 됩니다. 대게 시스템 프롬프트는 공개되지 않고, 숨겨져 있습니다. 이러한 시스템 프롬프트를 알 수 있는 방법이 있는데요. 이를 프롬프트 리킹이라고 합니다. 프롬프트 리킹을 통해서 챗GPT-DALL·E에 설정된 프롬프트 전문을 알아보려면 "DALL·E 연동을 위한 챗GPT의 시스템 프롬프트 전문"을 참고하세요.

### 1.2.2. 프롬프트 튜닝

이미지 프롬프트가 DALL·E에 전달되면 이미지가 생성됩니다. 생성된 이미지는 챗GPT를 통해 사용자에게 제공됩니다. 사용자는 이 이미지를 바탕으로 추가적인 조정이 필요한지 판단할 수 있습니다. 만약 더 정밀한 조정이 필요하다고 판단된다면, 사용자는 챗GPT에게 구체적인 피드백을 제공할 수 있습니다. 챗GPT는 이 피드백을 분석하여 프롬프트를 어떻게 조정해야 할지 판단합니다. 

* 다른 이미지 생성을 위해 새로운 프롬프트 생성
* 사용자의 수정사항을 반영하기 위해 기존 이미지의 시드와 프롬프트 활용

여기서 “수정”이라는 의미는 기존에 생성된 이미지의 프롬프트를 챗GPT가 알고 있어야 가능한 일인데요. 챗GPT는 대화 히스토리를 알고 있기 때문에 가능한 일입니다. 이를 통해 앞서 언급한 시드도 고정할 수 있으니, 이미지를 조금씩 수정하면서 사용자와 함께 원하는 작품을 생성합니다.

## 이어서

챗GPT를 통해 DALL·E를 위한 이미지 생성 프롬프트를 생성했다면, DALL·E에게 어떤 명령을 내릴까요? 좀 더 깊게 알아보겠습니다.

* [챗GPT는 DALL·E에 어떤 명령을 내리는가?](https://tykimos.github.io/2023/10/29/what_commands_does_chatgpt_give_to_dalle/)