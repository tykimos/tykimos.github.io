---
layout: post
title: "매너스 기술 및 아키텍처 심층 분석"
author: 김태영
date: 2025-03-08 03:00:00
categories: [Manus, AI, Agent, Assistant, AssiWorks]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-3-8-in_depth_analysis_of_manus_technology_and_architecture_title_1.jpg
---

 Manus는 여러 AI 모델을 결합한 독특한 아키텍처를 채택하고 있습니다. 개발사 Monica.im에 따르면 Manus는 다수의 독립적인 AI 모델의 협업, 즉 멀티시그<sup>multisignature</sup> 방식으로 동작하여 복잡한 의사결정을 수행합니다​
TRIBUNE.COM.PK

<iframe width="100%" height="400" src="https://youtube.com/embed/..." title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>

. 이러한 구조 하에서 각 모델이 부분적으로 판단을 내리고 상호 검증함으로써, 단일 모델 대비 더 높은 신뢰성과 정확성을 얻고 있다고 합니다. Manus의 핵심 엔진은 **대형 언어 모델(LLM)**들을 기반으로 구성되어 있습니다. 정확히 어떤 언어 모델들을 사용하는지는 공개되지 않았지만, 개발팀의 이전 제품인 Monica가 OpenAI의 Claude와 중국산 DeepSeek 등 최신 LLM 여러 개를 통합했던 전례가 있습니다​
TRIBUNE.COM.PK


​
TECHNODE.COM
. 이를 미루어볼 때 Manus 역시 여러 거대 언어 모델의 능력을 조합하여 활용하는 것으로 추정됩니다. 또한 Manus는 “Less structure, more intelligence”, 즉 규칙화된 모듈보다는 데이터 품질과 모델 자체의 성능, 유연한 구조에 중점을 둔 설계를 지향하고 있습니다​
TRIBUNE.COM.PK
. 이는 사전에 정해진 시나리오나 템플릿보다, 강력한 AI 모델이 자체적으로 최적 해법을 찾아내도록 하는 철학으로 볼 수 있습니다. Manus의 동작 원리는 고도화된 AI 모델들을 계획(planning)-추론(reasoning)-실행(action) 단계로 연계시키는 에이전트 아키텍처로 이해할 수 있습니다. 사용자가 목표를 입력하면 Manus는 이를 달성하기 위한 세부 계획을 세우고, 필요한 경우 인터넷 검색이나 코드 실행 등의 **도구 사용(tool use)**을 병행하면서 단계별로 작업을 진행합니다​
OPENREVIEW.NET
. 이러한 프로세스는 사람이 문제 해결에 접근하는 방식과 유사하며, Manus는 각 단계에서 결과를 검증하거나 필요시 경로를 수정할 수 있는 자기 피드백 및 반성(self-reflectiveness) 능력도 갖춘 것으로 알려졌습니다. 요약하면, Manus Agent는 복수의 AI 모델로 하여금 협동적으로 사고하고 행동하게 함으로써, 사용자의 “생각을 실행으로 옮기는” 기술적 기반을 구축한 것입니다​
MANUS.IM
.


## 마무리하며 💌

(작성)

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