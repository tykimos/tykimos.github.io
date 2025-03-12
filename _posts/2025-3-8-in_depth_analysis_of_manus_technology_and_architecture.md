---
layout: post
title: "매너스 기술 및 아키텍처 심층 분석"
author: 김태영
date: 2025-03-08 03:00:00
categories: [Manus, AI, Agent, Assistant, AssiWorks]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-3-8-in_depth_analysis_of_manus_technology_and_architecture_title_1.jpg
---

매너스(Manus)는 **멀티 에이전트 기반 AI 시스템**으로, 다양한 인공지능 모델이 협업하여 복잡한 문제를 해결하는 **범용 AI 에이전트**입니다. 기존의 단일 AI 모델이 모든 작업을 수행하는 방식과 달리, 매너스는 여러 개의 AI 모델이 각자의 역할을 수행한 후 이를 종합하여 최적의 결론을 도출하는 방식을 사용합니다.  

<iframe width="100%" height="400" src="https://youtube.com/embed/..." title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>

이러한 구조 덕분에 매너스는 단순한 질문 응답을 넘어 **검색, 데이터 분석, 문서 작성, 프로그래밍 실행** 등 다양한 기능을 수행할 수 있으며, 이를 지원하는 **도구(tool)**도 제공합니다.  

## 매너스의 핵심 개념  

매너스는 AI 시스템이 효율적으로 작동할 수 있도록 **여섯 가지 주요 개념**을 중심으로 설계되었습니다. 이 개념들은 AI가 사용자 요청을 분석하고, 계획을 세우며, 작업을 실행하고, 결과를 제공하는 과정을 정의합니다.  

| 개념 | 설명 |
|------|------|
| **chat** | 사용자의 요청을 입력받고 AI가 응답을 제공하는 개념 |
| **plan** | AI가 작업을 수행하기 위한 단계별 계획을 수립하고 실행하는 개념 |
| **sandbox** | AI가 코드 실행 및 데이터 분석을 수행하는 격리된 환경 |
| **tool** | AI가 웹 검색, 브라우저 탐색, 문서 편집 등을 수행하는 도구 |
| **data_api** | 외부 데이터를 실시간으로 가져오기 위한 내장 기능 |
| **knowledge** | 사용자의 대화에 따라 지식을 저장하고 접근할 수 있는 공간 |

## 에이전트 동작 흐름

기초적인 동작 흐름을 아래 다이어그램으로 표시해봤습니다. 

![매너스 시스템 동작 흐름](http://tykimos.github.io/warehouse/2025/2025-3-8-in_depth_analysis_of_manus_technology_and_architecture_0.jpg)

1. **사용자 요청 분석 (chat)** : 사용자가 메시지를 입력합니다.
2. **실행 환경 설정 (sandbox)** : 매너스 오케스트레이터가 실행 환경을 준비합니다.
3. **작업 계획 생성 (plan)** : 매너스 오케스트레이터가 계획을 수립합니다. 이때 todo.md라는 파일을 사용합니다.
4. **적절한 도구 사용 (tool)** : 계획에 필요한 도구를 적절하게 사용하여 사용자 요구사항을 만족시킵니다.
5. **작업 상태 업데이트 (status)** : 각 단계별로 상황을 모니터링하면서 계획을 진행시킵니다.
6. **최종 결과 반환 및 정리 (chat)** : 최종 결과를 사용자에게 보고하고, 계획 종료 및 실행 환경도 종료시킵니다.

## 주요 개념별 상세 설명  

### **chat: 사용자와 AI 간의 상호작용**  

chat은 매너스가 사용자와 소통하는 가장 기본적인 개념입니다. 사용자는 매너스에게 질문을 하거나 작업을 요청할 수 있으며, 매너스는 이에 대한 답변을 제공합니다. 매너스의 chat은 단순한 문장 응답을 넘어 **사용자의 요청을 분석하고, 필요한 경우 추가 작업을 수행하는 기능**을 합니다. 예를 들어, 사용자가 "스위스 알프스에서 가장 좋은 하이킹 코스를 찾아줘"라고 요청하면, 매너스는 이 요청을 분석한 뒤 **검색을 수행하고, 정보를 정리한 후 사용자에게 제공하는 방식**으로 작동합니다.  

### **plan: AI의 작업 계획 및 실행 흐름**  

plan은 AI가 사용자의 요청을 수행하기 위해 필요한 **단계별 실행 계획**을 수립하는 개념입니다. 매너스는 사용자의 요청을 받은 후, 단순한 응답을 제공하는 것이 아니라 **작업을 자동화할 계획을 생성**합니다.  

예를 들어, 사용자가 "비즈니스 여행을 위한 최적의 항공권을 찾아줘"라고 요청하면, 매너스는 다음과 같은 단계별 계획을 수립할 수 있습니다.  

1. **검색 단계**: 사용자의 요구사항(출발지, 목적지, 예산 등)에 맞는 항공권 검색  
2. **필터링 단계**: 환불 가능 여부, 비즈니스 클래스 옵션 등 추가 조건 적용  
3. **정리 단계**: 최적의 10개 항공편을 리스트로 정리  
4. **결과 제공 단계**: 사용자에게 최종 리스트 제공  

이처럼 plan을 활용하면 **작업을 체계적으로 나누고, 효율적으로 수행할 수 있습니다.** 아래는 실제 사용사례에서 todo.md 파일을 캡처한 것입니다. 마크다운 형식으로 되어 있으며, 진행 완료가 된 것은 X표시로 되어 있는 것을 확인할 수 있습니다. 이후 과정에서는 이 todo.md 파일을 참고하여 각 단계별로 작업을 수행합니다.

![매너스 계획 생성](http://tykimos.github.io/warehouse/2025/2025-3-8-in_depth_analysis_of_manus_technology_and_architecture_1.jpg)

### **sandbox: AI의 실행 환경**  

sandbox는 AI가 데이터를 처리하거나 프로그램을 실행하는 독립적인 환경입니다. 매너스는 특정 작업을 수행할 때, 실제 코드 실행이 필요한 경우가 많습니다. 이때 안전하고 격리된 환경에서 작업을 수행하기 위해 sandbox를 활용합니다. sandbox를 사용하면 AI가 실제 컴퓨터 환경을 구성하고, 파일을 생성하거나 코드를 실행할 수 있습니다. 아래는 실제 사용사례에서 sandbox 위에서 터미널이 실행되는 모습을 캡처한 것입니다. 

![매너스 실행 환경](http://tykimos.github.io/warehouse/2025/2025-3-8-in_depth_analysis_of_manus_technology_and_architecture_2.jpg)

### **tool: AI가 활용하는 도구**  

매너스는 다양한 작업을 수행하기 위해 여러 개의 도구(tool)를 제공합니다. 각 도구는 특정한 기능을 수행하며, AI가 작업을 보다 정교하게 실행할 수 있도록 돕습니다.  

| 도구 | 설명 | 상세보기 |
|------|------|----------|
| **search** | 웹 검색을 통해 정보를 수집하는 도구 | [링크](https://tykimos.github.io/2025/03/08/manus_tools_websearch) |
| **browser** | 특정 웹사이트를 열람하고 데이터를 가져오는 도구 | [링크](https://tykimos.github.io/2025/03/08/manus_tools_browser) |
| **text_editor** | 문서를 생성하고 편집하는 도구 | [링크](https://tykimos.github.io/2025/03/08/manus_tools_text_editor) |
| **terminal** | 터미널 명령어를 실행하는 도구 | [링크](https://tykimos.github.io/2025/03/08/manus_tools_terminal) |

각 도구는 AI가 필요할 때 자동으로 호출됩니다. 각각에 대해서는 상세보기에서 심도있게 다루도록 하겠습니다.

## 매너스의 데이터 API (data_api)  

data_api는 매너스가 **외부 데이터에 접근하고 활용하기 위한 내장 기능**입니다. 이 API를 통해 금융 정보, 경제 지표, 소셜 미디어 데이터 등 다양한 실시간 정보를 조회하고 분석할 수 있습니다.특히 중요한 점은 data_api가 **코드 실행이나 웹 검색 없이도** 신뢰할 수 있는 최신 데이터에 접근할 수 있게 해준다는 것입니다. 이는 매너스가 빠르고 정확한 정보를 제공하는 데 큰 도움이 됩니다. 제가 확인한 데이터 API는 아래와 같습니다.:

| API ID | API 이름 | 설명 |
|--------|----------|------|
| api_16 | Get stock profile | 주식 프로필 조회 |
| api_19 | Get stock chart | 주식 차트 조회 |
| api_20 | Get stock holders | 주식 보유자 조회 |
| api_21 | Get stock insights | 주식 인사이트 조회 |
| api_22 | Get stock SEC filing | 주식 SEC 제출 조회 |
| api_23 | Get what analysts are saying of a stock | 주식 분석가 의견 조회 |
| api_25 | Get worldbank indicator data | 세계은행 지표 데이터 조회 |
| api_26 | Lookup worldbank indicator detail | 세계은행 지표 상세 조회 |
| api_27 | Get worldbank indicators list | 세계은행 지표 목록 조회 |
| api_28 | Search Twitter | 트위터 검색 |
| api_29 | Get Twitter profile by username | 트위터 프로필 조회 |
| api_30 | Get user tweets | 트위터 사용자 트윗 조회 |
| api_31 | Get LinkedIn profile by username | 링크드인 프로필 조회 |
| api_32 | Search people on LinkedIn | 링크드인 사람 검색 |
| api_33 | Get company's LinkedIn details | 회사 링크드인 상세 조회 |

이 API들은 매너스가 **빠르게 정보를 검색하고, 데이터를 분석하며, 실시간으로 활용할 수 있도록 지원합니다.** 아래는 주식관련된 정보를 내부 data_api를 통해서 가지고 오는 모습을 캡처한 것입니다. 

![매너스 데이터 API](http://tykimos.github.io/warehouse/2025/2025-3-8-in_depth_analysis_of_manus_technology_and_architecture_3.jpg)

### **knowledge: AI의 지식 저장소**

knowledge는 사용자와의 대화 내용을 기반으로 **중요한 정보와 지식을 저장하는 공간**입니다. 매너스는 사용자와 대화하는 과정에서 얻은 정보를 이 저장소에 기록하고, 필요할 때 이 정보에 접근하여 활용합니다. 이 지식 저장소는 **대화의 연속성을 유지**하고 **사용자의 선호도와 요구사항을 학습**하는 데 중요한 역할을 합니다. 예를 들어, 사용자가 "나는 해산물 알레르기가 있어"라고 언급하면, 매너스는 이 정보를 knowledge에 저장하고, 추후 식당 추천이나 요리법 제안 시 해산물을 제외한 옵션을 제공합니다. 매너스는 작업을 수행하면서 knowledge를 지속적으로 업데이트하고, 새로운 정보를 추가하며, 필요한 경우 기존 정보를 수정하거나 삭제합니다. 이를 통해 **개인화된 경험**을 제공하고 **맥락에 맞는 정확한 응답**을 할 수 있습니다.

## 마무리  

매너스는 **멀티 에이전트 시스템**을 활용하여 사용자의 요청을 분석하고, 작업을 계획하며, 다양한 도구를 활용하여 최적의 결과를 제공하는 강력한 AI 시스템입니다. 이 시스템은 chat, sandbox, tool, plan, knowledge, data_api라는 여섯 가지 핵심 개념을 기반으로 작동하며, 이 개념들을 활용하여 사용자의 요청을 처리하고 최적의 결과를 제공합니다. 

## 참고자료

- [매너스 공식 사이트](https://manus.im)

## 함께 읽기

1. [지금 중국은 매너스 열풍! 범용 AI 에이전트](https://tykimos.github.io/2025/03/08/manus_the_general_ai_agent)
2. [매너스 UI 사용법 및 리플레이 살펴보기](https://tykimos.github.io/2025/03/08/exploring_manus_ui_usage_and_replay)
3. [매너스 기술 및 아키텍처 심층 분석](https://tykimos.github.io/2025/03/08/in_depth_analysis_of_manus_technology_and_architecture)
4. [매너스 도구들 - 웹검색](https://tykimos.github.io/2025/03/08/manus_tools_websearch)
5. [매너스 도구들 - 브라우저](https://tykimos.github.io/2025/03/08/manus_tools_browser)
6. [매너스 도구들 - 문서편집기](https://tykimos.github.io/2025/03/08/manus_tools_text_editor)
7. [매너스 도구들 - 터미널](https://tykimos.github.io/2025/03/08/manus_tools_terminal)
8. [매너스 1등한 범용 AI 평가 GAIA 소개](https://tykimos.github.io/2025/03/08/gaia_manus_evaluation)
9. [매너스 사례들](https://tykimos.github.io/2025/03/08/manus_usecases)

## (광고) 한국의 노코드 에이전틱AI 플랫폼

AIFactory에서도 에이전틱AI 플랫폼을 서비스 및 고도화하고 있습니다. 어시웍스(AssiWorks)는 "도구(Tools)", "워크플로우(Flows)", "에이전트(Agents)", "팀(Teams)"이라는 네 가지 주요 개념을 중심으로, 노코드(No-Code) 환경에서 AI 기반 업무 자동화와 협업형 에이전트 구성을 손쉽게 구현할 수 있도록 지원하는 종합 플랫폼입니다. 

자세히 보기 >> [어시웍스](https://aifactory.space/guide/8/14)

![어시웍스](http://tykimos.github.io/warehouse/2025/2025-3-8-assiworks.png)

## 퍼가는 법

이 글은 자유롭게 퍼가셔도 좋아요! 다만 출처는 아래 링크로 꼭 남겨주세요 😊

[https://tykimos.github.io/2025/03/08/in_depth_analysis_of_manus_technology_and_architecture/](https://tykimos.github.io/2025/03/08/in_depth_analysis_of_manus_technology_and_architecture/)