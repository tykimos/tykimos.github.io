---
layout: post
title: "매너스 기술 및 아키텍처 심층 분석"
author: 김태영
date: 2025-03-08 03:00:00
categories: [Manus, AI, Agent, Assistant, AssiWorks]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-3-8-in_depth_analysis_of_manus_technology_and_architecture_title.jpg
---

 Manus는 여러 AI 모델을 결합한 독특한 아키텍처를 채택하고 있습니다. 개발사 Monica.im에 따르면 Manus는 다수의 독립적인 AI 모델의 협업, 즉 멀티시그<sup>multisignature</sup> 방식으로 동작하여 복잡한 의사결정을 수행합니다​
TRIBUNE.COM.PK
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
2. 성능 분석
Manus는 발표 직후부터 다양한 실제 과제들에서 인상적인 성능을 시연했습니다. 일반적인 챗봇과 달리, Manus는 단순 질의응답을 넘어 완결된 작업 결과를 제공하는 것이 특징입니다​
TRIBUNE.COM.PK
. 예를 들어, 시연 영상에서 Manus는 지원자의 이력서를 일일이 검토해 채용 후보자를 선별하고​
TRIBUNE.COM.PK
, 부동산 투자 분석 리포트를 작성하며, 주식 데이터의 상관관계를 계산해 금융 분석 결과를 도출하는 모습을 보여주었습니다​
INVESTING.COM
. 또한 Python 코드를 자동으로 생성하여 데이터를 시각화하고, 나아가 해당 코드를 이용해 인터랙티브 웹사이트를 실제로 배포하는 등 복잡한 IT 업무도 수행해냈습니다​
INVESTING.COM
. 이처럼 Manus는 업무, 생활, 교육 등 다방면의 작업을 스스로 처리하는데, 공식 웹사이트의 활용 사례만 보더라도 맞춤형 여행 일정 수립​
MANUS.IM
, 주식 데이터 심층 분석 및 대시보드 생성​
MANUS.IM
, 교육용 프레젠테이션 자료 제작, 보험 상품 비교 및 최적안 추천​
MANUS.IM
 등 광범위한 영역을 포괄하고 있습니다. 특히 Manus는 작업을 클라우드 상에서 비동기적으로 실행하기 때문에 사용자가 PC나 앱을 꺼두어도 알아서 일을 계속 진행하고, 완료 시 결과를 제공하는 백그라운드 작업능력을 갖추고 있습니다​
INVESTING.COM
. 또한 사용과정에서 축적한 데이터를 통해 장기 메모리와 학습 기능을 발휘, 사용자 성향에 맞춰 진화하는 퍼스널 에이전트로 설계되었습니다​
INVESTING.COM
.

안녕하세요, 여러분! 👋 요즘 중국 개발자 커뮤니티에서 정말 핫한 이슈가 있는데요. 바로 'Manus(매너스 혹은 마너스)'라는 범용 AI 에이전트입니다. 이게 대체 왜 그렇게 주목받는 걸까요? 기술적으로는 또 어떤 특별한 점이 있는지 함께 살펴봐요!

<iframe width="100%" height="400" src="https://youtube.com/embed/..." title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>

## 매너스(Manus)가 뭐길래 이렇게 핫한거죠? 🤔

'매너스(Manus)'라는 이름, 들어보셨나요? 이 이름은 "Mens et Manus(라틴어: 정신과 손)"에서 따왔다고 해요. "지식은 반드시 실천되어야 한다"는 의미를 담고 있죠. 매너스는 단순히 정보만 알려주는 게 아니라, 여러분의 손이 되어 아이디어를 현실로 만들어준답니다! 웹에서 챗봇처럼 간단하게 사용할 수 있고, 다른 사람들이 공유한 결과도 리플레이 기능으로 쉽게 볼 수 있어요. (궁금하시다면 [매너스 UI 사용법 및 리플레이 살펴보기](https://tykimos.github.io/2025/03/08/exploring_manus_ui_usage_and_replay)를 참고해보세요!)

매너스는 기존 AI 어시스턴트와는 좀 다른데요, 여러분의 목표를 진짜로 이해하고 실제 행동으로 옮겨 결과를 만들어내는 완전 자율 AI 에이전트랍니다. 여러 AI 에이전트들이 협력해서 복잡한 일을 처리하는 구조예요. ([매너스 기술 및 아키텍처 심층 분석](https://tykimos.github.io/2025/03/08/in_depth_analysis_of_manus_technology_and_architecture)에서 더 자세히 알아보세요!) 각 모델이 판단을 내리고, 서로 검증하면서 정확도를 높인답니다.

매너스의 핵심 능력 세 가지를 소개할게요:
- **계획(planning)**: 여러분의 목표를 이해하고 할 일 목록을 만들어 하나씩 해결해요.
- **추론(reasoning)**: 계획을 달성하기 위한 최적의 방법을 찾아내죠.
- **실행(action)**: 실제로 작업을 수행하고 결과를 가져옵니다. 다양한 도구들을 활용하고 공식 데이터 소스에서 신뢰할 수 있는 정보만 가져와요!

게다가 무엇보다 대단한 건, 자기 피드백 능력이 있어서 스스로 문제를 발견하면 다른 방법을 찾아 해결한다는 거예요. 거의 인간 같지 않나요? 😮

## 매너스로 뭘 할 수 있을까요? 🚀

정말 다양한 분야에서 활용할 수 있어요! 매너스 공식사이트(manus.im)에 가보면 정말 많은 예시들이 있는데요, 몇 가지만 살펴볼까요? ([매너스 사례들](https://tykimos.github.io/2025/03/08/manus_usecases)에서 더 많은 사례를 확인하세요!)

- **채용 후보자 선별**: 지원자들의 이력서를 쫙~ 분석해서 적합한 사람을 찾아줍니다.
- **부동산 투자 분석**: 부동산 시장 데이터를 분석해서 "여기 투자하면 어떨까요?" 하는 보고서를 작성해줘요.
- **IT 업무 자동화**: 복잡한 IT 작업을 알아서 척척 해결합니다.
- **맞춤형 여행 일정**: "난 맛집 투어를 좋아하고 미술관도 가고 싶어" 이런 요구사항에 맞는 여행 일정을 자동으로 짜줘요!

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_the_general_ai_agent_1.jpg)

더 좋은 건, 이 모든 작업이 클라우드에서 비동기적으로 실행되기 때문에 컴퓨터를 끄거나 앱을 종료해도 매너스는 계속 일을 하고 있어요. 작업이 끝나면 결과를 알려준답니다. 마치 개인 비서가 있는 것 같지 않나요? 👍

## 진짜로 잘하는 걸까요? 성능은 어떨까요? 🏆

"말은 그럴듯한데, 실제로는 어떨까?" 하실 수 있겠죠? 매너스는 GAIA(General AI Assistants)라는 벤치마크 테스트에서 기존 최고 성능을 경신했답니다! 결과를 한번 볼까요?

- Level 1: Manus 86.5% (OpenAI 74.3%)
- Level 2: Manus 70.1% (OpenAI 69.1%)
- Level 3: Manus 57.7% (OpenAI 47.6%)

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_the_general_ai_agent_2.jpg)

이 결과만 봐도 매너스가 얼마나 뛰어난지 알 수 있죠? 전문가들도 매너스를 OpenAI의 강력한 경쟁자로 평가하고 있어요. ([매너스 1등한 범용 AI 평가 GAIA 소개](https://tykimos.github.io/2025/03/08/gaia_manus_evaluation)에서 더 자세히 알아보세요!)

## 사람들의 반응은 어떨까요? 🔥

매너스는 공개 직후 중국의 웨이보(Weibo) 등 소셜 미디어에서 폭발적인 반응을 얻었어요! 실시간 트렌드에 오르고, 베타 초대 코드를 구하기 위해 사람들이 난리였다고 해요. 심지어 중고 거래 시장에서 초대 코드가 거래될 정도였으니, 그 인기가 어느 정도인지 짐작이 가시죠? 개발자 커뮤니티에서도 매너스에 대한 평가가 정말 좋다고 합니다.

## 앞으로의 계획은? 사용해볼 수 있을까요? 🔮

현재 매너스는 아직 초대 사용자들만 사용할 수 있는 베타 테스트 단계에요. 아쉽게도 아직 일반 사용자들이 바로 사용할 수는 없지만, 개발팀에서 일부 AI 모델과 추론 컴포넌트를 오픈소스로 공개할 계획이라고 해요! 연구자들과 협업하면서 점차 개방 범위를 넓혀갈 예정이랍니다.

그리고 반가운 소식! 오픈소스 진영에서 [OpenManus](https://github.com/mannaandpoem/OpenManus)가 이미 공개되었어요. 매너스 초대코드 구하기가 하늘의 별 따기처럼 어렵다 보니, 개발자들이 매너스를 본떠서 만든 오픈소스 프로젝트도 있으니 한번 살펴보는 것도 좋을 것 같아요!

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_the_general_ai_agent_0.jpg)

## 마무리하며 💌

매너스는 정말 사용자의 목표를 이해하고 실제 행동으로 옮기는 완전 자율 AI 에이전트로, 여러 AI 모델이 협력해서 놀라운 성능을 보여주고 있어요. 앞으로 매너스가 더 많은 사람들에게 공개되면 정말 다양한 분야에서 혁신적인 활용 사례가 나올 것 같아 너무 기대됩니다! 여러분은 어떤 용도로 매너스를 사용해보고 싶으신가요? 댓글로 알려주세요! 😊 자 그럼 이제 [매너스 UI 사용법 및 리플레이 살펴보기](https://tykimos.github.io/2025/03/08/exploring_manus_ui_usage_and_replay) 포스팅을 읽어볼까요?

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