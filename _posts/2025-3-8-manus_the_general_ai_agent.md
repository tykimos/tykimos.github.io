---
layout: post
title: "지금 중국은 매너스 열풍! 범용 AI 에이전트"
author: 김태영
date: 2025-03-07 10:00:00
categories: [Manus, AI, Agent, Assistant, AssiWorks]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-3-8-manus_the_general_ai_agent_title.jpg
---

지금 중국에서는 Manus(매너스 혹은 마너스) 라는 범용 AI 에이전트가 큰 화제인데요. 매너스가 무엇인지, 왜 화제인지, 사용사례, 커뮤니티 소식등을 살펴보겠습니다. 

<iframe width="100%" height="400" src="https://youtube.com/embed/..." title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>

## 매너스(Manus)란?

매너스(Manus)라는 이름은 유명한 개념인 "Mens et Manus (라틴어: 정신과 손)"에서 따왔습니다. 이는 "지식은 반드시 실천되어야 한다"는 신념을 의미합니다. 매너스의 목표는 단순한 정보 제공이 아니라, 사용자의 역량을 확장하고, 영향력을 증대시키며, 당신의 아이디어를 현실로 만들어 주는 손이 되는 것입니다. 웹서비스를 통해 챗봇 형식으로 손쉽게 사용할 수 있으며, 다른 사람이 공유한 결과를 리플레이 기능을 통해서 그 과정을 살펴볼 수 있습니다. (참고 : [매너스 UI 사용법 및 리플레이 살펴보기](https://tykimos.github.io/2025/03/08/exploring_manus_ui_usage_and_replay))

매너스의 에이전트는 기존의 챗봇이나 AI 어시스턴트와 달리 사용자의 목표를 이해하여 실제 행동과 결과를 만들어내는 완전 자율 AI 에이전트입니다. 매너스는 여러 AI 에이전트가 협력하여 복잡한 의사결정을 수행하는 구조를 가지고 있습니다. (참고 : [매너스 기술 및 아키텍처 심층 분석](https://tykimos.github.io/2025/03/08/in_depth_analysis_of_manus_technology_and_architecture))각 모델은 부분적인 판단을 내리고, 이를 상호 검증하여 신뢰성과 정확성을 높입니다. 

- **계획(planning)**: 사용자의 목표를 이해하고 이를 달성하기 위한 세부 계획을 수립합니다. 할 일 목록을 내부적으로 관리하면서 하나하나 수행합니다.
- **추론(reasoning)**: 계획을 달성하기 위한 최적의 방법을 추론하고, 필요한 작업을 결정합니다.
- **실행(action)**: 계획된 작업을 실제로 수행하고 결과를 제공합니다. 이 때 매너스는 여러 도구들을 활용하여 작업을 수행합니다. 또한 공식 데이터 소스에 접근하여 신뢰할 수 있는 데이터를 가져옵니다.

또한 매너스는 자기 피드백 및 반성(self-reflectiveness) 능력을 갖추고 있어, 작업 수행 중 문제가 발생하면 스스로 경로를 수정할 수 있습니다. 

## 주요 사용 사례

업무, 생활, 교육 등 다양한 분야에서 실제 작업을 수행할 수 있습니다. 매너스 공식사이트(manus.im)에 다양한 예시가 있습니다. (참고 : [매너스 사례집](https://tykimos.github.io/2025/03/08/manus_usecases))

- **채용 후보자 선별**: 지원자의 이력서를 분석하여 적합한 후보자를 자동으로 선별합니다.
- **부동산 투자 분석**: 부동산 시장 데이터를 분석하여 투자 가능성을 평가하고 보고서를 작성합니다.
- **IT 업무 자동화**: 복잡한 IT 작업을 수행하고, 웹 서비스 배포와 같은 기술적 업무를 처리합니다.
- **맞춤형 여행 일정 수립**: 사용자의 취향과 요구사항에 맞는 여행 일정을 자동으로 계획합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_the_general_ai_agent_1.jpg)

이러한 작업들은 클라우드 상에서 비동기적으로 실행되기 때문에 사용자가 PC나 앱을 종료해도 매너스는 백그라운드에서 작업을 계속 진행하고 완료 시 결과를 제공합니다.

## 성능 평가 및 경쟁력

매너스는 GAIA(General AI Assistants) 벤치마크 테스트에서 기존 최고 성능을 경신하며 뛰어난 성능을 입증했습니다. 구체적인 결과는 다음과 같습니다.

- Level 1: Manus 86.5% (OpenAI 74.3%)
- Level 2: Manus 70.1% (OpenAI 69.1%)
- Level 3: Manus 57.7% (OpenAI 47.6%)

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_the_general_ai_agent_2.jpg)

이러한 결과는 매너스가 복잡한 작업을 처리하는 데 있어 기존 AI 모델 대비 우수한 성능을 보인다는 것을 의미합니다. 전문가들은 매너스를 OpenAI가 준비 중인 자율 에이전트의 강력한 경쟁 상대로 평가하고 있습니다. (참고: [매너스 1등한 범용 AI 평가 GAIA 소개](https://tykimos.github.io/2025/03/08/gaia_manus_evaluation))

## 커뮤니티 반응과 매너스의 인기

매너스는 공개 직후 중국 웨이보(Weibo) 등 소셜 미디어에서 큰 화제를 모으며 실시간 트렌드에 올랐습니다. 특히 베타 초대 코드에 대한 수요가 폭발적으로 증가하여 일부 중고 거래 시장에서 초대 코드가 거래될 정도로 인기를 끌었습니다. 개발자 커뮤니티에서도 매너스의 접근 방식과 잠재력에 대한 긍정적인 평가가 주를 이루고 있습니다.

## 오픈소스 계획과 향후 전망

현재 매너스는 초대 사용자들을 대상으로 한 베타 테스트 단계이며, 일반 사용자들이 직접 사용할 수 있는 상태는 아닙니만, 개발팀은 향후 일부 AI 모델과 추론 컴포넌트를 오픈소스로 공개할 계획을 밝혔습니다. 이를 통해 연구자들과의 협업을 유도하고, 단계적으로 개방 범위를 확대할 예정입니다.

오픈소스 진영에서도 [OpenManus](https://github.com/mannaandpoem/OpenManus)가 공개되었는데요. Manus의 초대코드를 구하기 힘들기 때문에 개발자들이 매너스를 본 따서 만든 오픈소스 프로젝트도 있으니 살펴보시기 바랍니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_the_general_ai_agent_0.jpg)

## 마무리

매너스는 사용자의 목표를 이해하고 실제 행동과 결과를 만들어내는 완전 자율 AI 에이전트로, 여러 AI 모델의 협력적 구조를 통해 뛰어난 성능과 신뢰성을 제공합니다. 앞으로 매너스의 공개 범위가 확대되면 더 많은 사용자와 개발자들이 참여하여 다양한 분야에서 혁신적인 활용 사례가 등장할 것으로 기대됩니다.

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
9. [매너스 사례집](https://tykimos.github.io/2025/03/08/manus_usecases)