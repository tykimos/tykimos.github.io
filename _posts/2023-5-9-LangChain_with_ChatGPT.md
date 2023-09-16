---
layout: post
title:  "Langchain, Giving Wings to ChatGPT"
author: 김태영
date:   2023-5-9 00:00:00
categories: tech
comments: true
image: https://tykimos.github.io/warehouse/2023/2023-5-9-LangChain_with_ChatGPT.gif
---
일본 IoT ALGYAN 커뮤니티의 챗GPT 세미나에 초청되어 랭체인을 주제로 발표하였습니다. 

![](https://tykimos.github.io/warehouse/2023/2023-5-9-LangChain_with_ChatGPT_02.png)

랭체인의 기본과 활용 예제, VisualChatGPT 내용을 중심으로 개념 및 코드리뷰를 했었습니다. 랭체인의 이해를 돕고자 대규모언어모델(LLM)과 랭체인(LangChain)를 에반게리온 초호기에 비유해봤습니다.

![](https://tykimos.github.io/warehouse/2023/2023-5-9-LangChain_with_ChatGPT.gif)

* 대규모언어모델는 에반게리온 초호기의 구속구를 제거한 본체 모습입니다. 굉장히 파워풀하나 통제 및 소통하기 힘듭니다.
* Output Parsers는 구속구에 해당합니다. 우리가 원하는 형태를 나올 수 있도록 제어합니다.
* 에반게리온이 미션을 수행하기 위해서 선택되는 무기 또는 도구는 LangChain의 Tools에 해당합니다.
* 프롬프트, 언어모델, Output Parser 등을 모두 연동하여 하나로 묶어주는 역할인 Chain은 엔트리 플러그에 해당합니다.
* 실제 당신이 제어하는 것은 엔트리 플러그 내에 조종석입니다. 이를 통해 명령을 내리는 데, 이 명령을 효율적으로 전달할 수 있도록 프롬프트 템플릿이 그 역할을 수행합니다.
* 에반게리온과 조종사외에도 기체를 운용하기 위해 네르프에서 여러 에이전트가 돕습니다. 이는 LangChain의 Agent와 비슷한 역할을 수행합니다.
