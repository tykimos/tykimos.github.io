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

이러한 구조 덕분에 매너스는 단순한 질문 응답을 넘어 **검색, 데이터 분석, 문서 작성, 프로그래밍 실행** 등 다양한 기능을 수행할 수 있으며, 이를 지원하는 **도구(tool)**도 제공합니다.  

## 매너스의 핵심 개념  

매너스는 AI 시스템이 효율적으로 작동할 수 있도록 **네 가지 주요 개념**을 중심으로 설계되었습니다. 이 개념들은 AI가 사용자 요청을 분석하고, 계획을 세우며, 작업을 실행하고, 결과를 제공하는 과정을 정의합니다.  

| 개념 | 설명 |
|------|------|
| **`chat`** | 사용자의 요청을 입력받고 AI가 응답을 제공하는 개념 |
| **`sandbox`** | AI가 코드 실행 및 데이터 분석을 수행하는 격리된 환경 |
| **`tool`** | AI가 웹 검색, 브라우저 탐색, 문서 편집 등을 수행하는 도구 |
| **`plan`** | AI가 작업을 수행하기 위한 단계별 계획을 수립하고 실행하는 개념 |

이제 각 개념을 자세히 살펴보겠습니다.  

---

## 주요 개념별 상세 설명  

### **`chat`: 사용자와 AI 간의 상호작용**  

`chat`은 매너스가 사용자와 소통하는 가장 기본적인 개념입니다.  
사용자는 매너스에게 질문을 하거나 작업을 요청할 수 있으며, 매너스는 이에 대한 답변을 제공합니다.  

매너스의 `chat`은 단순한 문장 응답을 넘어 **사용자의 요청을 분석하고, 필요한 경우 추가 작업을 수행하는 기능**을 합니다.  
예를 들어, 사용자가 "스위스 알프스에서 가장 좋은 하이킹 코스를 찾아줘"라고 요청하면,  
매너스는 이 요청을 분석한 뒤 **검색을 수행하고, 정보를 정리한 후 사용자에게 제공하는 방식**으로 작동합니다.  

### **`sandbox`: AI의 실행 환경**  

`sandbox`는 AI가 데이터를 처리하거나 프로그램을 실행하는 독립적인 환경입니다. 매너스는 특정 작업을 수행할 때, 실제 코드 실행이 필요한 경우가 많습니다. 이때 안전하고 격리된 환경에서 작업을 수행하기 위해 `sandbox`를 활용합니다. `sandbox`를 사용하면 AI가 실제 컴퓨터 환경을 구성하고, 파일을 생성하거나 코드를 실행할 수 있습니다. 예를 들어, 사용자가 "TSMC의 1000억 달러 투자 분석 리포트를 만들어줘"라고 요청하면, 매너스는 `sandbox`를 생성한 후 필요한 데이터를 수집하고, 이를 분석하여 보고서를 작성합니다.  

### **`tool`: AI가 활용하는 도구**  

매너스는 다양한 작업을 수행하기 위해 여러 개의 도구(`tool`)를 제공합니다. 각 도구는 특정한 기능을 수행하며, AI가 작업을 보다 정교하게 실행할 수 있도록 돕습니다.  

| 도구 | 설명 |
|------|------|
| **`search`** | 웹 검색을 통해 정보를 수집하는 도구 |
| **`browser`** | 특정 웹사이트를 열람하고 데이터를 가져오는 도구 |
| **`text_editor`** | 문서를 생성하고 편집하는 도구 |
| **`terminal`** | 터미널 명령어를 실행하는 도구 |

각 도구는 AI가 필요할 때 자동으로 호출되며, 사용자가 직접 제어할 수도 있습니다.  

#### **도구별 기능**  

- **`search`**: 웹에서 특정 키워드를 검색하여 관련 정보를 수집합니다.  
- **`browser`**: 특정 웹사이트를 방문하여 내용을 분석하거나 데이터를 저장합니다.  
- **`text_editor`**: 문서를 생성하거나 내용을 수정하여 보고서를 작성합니다.  
- **`terminal`**: 파일을 생성하거나, 코드를 실행하는 등의 작업을 수행합니다.  

### **3.4 `plan`: AI의 작업 계획 및 실행 흐름**  

`plan`은 AI가 사용자의 요청을 수행하기 위해 필요한 **단계별 실행 계획**을 수립하는 개념입니다.  
매너스는 사용자의 요청을 받은 후, 단순한 응답을 제공하는 것이 아니라 **작업을 자동화할 계획을 생성**합니다.  

예를 들어, 사용자가 "비즈니스 여행을 위한 최적의 항공권을 찾아줘"라고 요청하면,  
매너스는 다음과 같은 단계별 계획을 수립할 수 있습니다.  

1. **검색 단계**: 사용자의 요구사항(출발지, 목적지, 예산 등)에 맞는 항공권 검색  
2. **필터링 단계**: 환불 가능 여부, 비즈니스 클래스 옵션 등 추가 조건 적용  
3. **정리 단계**: 최적의 10개 항공편을 리스트로 정리  
4. **결과 제공 단계**: 사용자에게 최종 리스트 제공  

이처럼 `plan`을 활용하면 **작업을 체계적으로 나누고, 효율적으로 수행할 수 있습니다.**  

## 매너스의 데이터 API (`data_api`)  

매너스는 외부 데이터를 실시간으로 가져오기 위해 **데이터 API (`data_api`)**를 제공합니다.  
이를 통해 금융 정보, 경제 지표, 소셜 미디어 데이터 등을 조회할 수 있습니다.  

| API ID | API 이름 |
|--------|----------|
| `api_16` | Get stock profile |
| `api_19` | Get stock chart |
| `api_20` | Get stock holders |
| `api_21` | Get stock insights |
| `api_22` | Get stock SEC filing |
| `api_23` | Get what analysts are saying of a stock |
| `api_25` | Get worldbank indicator data |
| `api_26` | Lookup worldbank indicator detail |
| `api_27` | Get worldbank indicators list |
| `api_28` | Search Twitter |
| `api_29` | Get Twitter profile by username |
| `api_30` | Get user tweets |
| `api_31` | Get LinkedIn profile by username |
| `api_32` | Search people on LinkedIn |
| `api_33` | Get company's LinkedIn details |

이 API들은 매너스가 **빠르게 정보를 검색하고, 데이터를 분석하며, 실시간으로 활용할 수 있도록 지원합니다.**  

## 마무리  

매너스는 **멀티 에이전트 시스템**을 활용하여 사용자의 요청을 분석하고, 작업을 계획하며, 다양한 도구를 활용하여 최적의 결과를 제공하는 강력한 AI 시스템입니다. 이 시스템은 `chat`, `sandbox`, `tool`, `plan`이라는 네 가지 핵심 개념을 기반으로 작동하며, **다양한 데이터 API를 통해 실시간 정보를 활용할 수 있습니다.** 🚀

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

AIFactory에서도 에이전틱AI 플랫폼을 서비스 및 고도화하고 있습니다. 어시웍스(AssiWorks)는 “도구(Tools)”, “워크플로우(Flows)”, “에이전트(Agents)”, “팀(Teams)”이라는 네 가지 주요 개념을 중심으로, 노코드(No-Code) 환경에서 AI 기반 업무 자동화와 협업형 에이전트 구성을 손쉽게 구현할 수 있도록 지원하는 종합 플랫폼입니다. 

자세히 보기 >> [어시웍스](https://aifactory.space/guide/8/14)

![어시웍스](http://tykimos.github.io/warehouse/2025/2025-3-8-assiworks.png)

## 퍼가는 법

이 글은 자유롭게 퍼가셔도 좋아요! 다만 출처는 아래 링크로 꼭 남겨주세요 😊

[https://tykimos.github.io/2025/03/08/manus_the_general_ai_agent/](https://tykimos.github.io/2025/03/08/manus_the_general_ai_agent/)