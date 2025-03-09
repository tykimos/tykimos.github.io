---
layout: post
title: "매너스 UI 사용법 및 리플레이 살펴보기"
author: 김태영
date: 2025-03-08 11:00:00
categories: [Manus, AI, Agent, Assistant, AssiWorks]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_title.jpg
---

오늘은 기대를 모으고 범용 AI 에이전트 '매너스(Manus)'의 사용자 인터페이스와 신기한 리플레이 기능에 대해 살펴보려 합니다.

<iframe width="100%" height="400" src="https://youtube.com/embed/..." title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>

## 초대코드 시스템

현재 매너스는 폭발적인 인기로 인해 서버 부하 관리를 위해 초대 코드 시스템을 운영 중입니다. 이는 성장 단계의 AI 서비스가 안정성을 유지하면서 확장하는 전형적인 전략이죠.

![초대코드 입력 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_0.jpg)

초대코드가 없으신가요? 걱정마세요! [접근 권한 신청하기] 버튼을 통해 대기자 명단에 등록할 수 있습니다. 물론 기다림의 긴 시간이 필요합니다.

![접근 권한 신청 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_0.jpg)

## 개발자 커뮤니티와의 실시간 소통

매너스 팀은 사용자 및 개발자 커뮤니티와의 원활한 소통을 위해 디스코드 채널을 활용하고 있습니다.

[https://discord.com/invite/gjuXBWaU](https://discord.com/invite/gjuXBWaU)

💡 **꿀팁**: 디스코드 채널에는 가끔 한정 수량의 초대코드가 공유됩니다. 알림 설정을 켜두고 빠르게 대응하면 운 좋게 코드를 얻을 수 있을지도 모릅니다!

![디스코드 채널 모습](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_1.jpg)

## 인터페이스 심층 분석

로그인에 성공하면 직관적이고 미니멀한 채팅 인터페이스가 여러분을 맞이합니다.

![매너스 메인 인터페이스](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_2.jpg)

### 사용자 맞춤 설정

우측 하단 설정 버튼을 클릭하면 설정창이 띄워집니다.

![설정 메뉴 접근](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_4.jpg)

설정 메뉴에서는 언어, 테마, 프로필 편집 등의 옵션이 제공됩니다.

![설정 메뉴 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_5.jpg)

우리 개발자들이 사랑하는 다크 테마도 물론 지원됩니다! 눈의 피로를 줄이면서 장시간 작업할 때 특히 유용하죠.

![다크 테마 적용 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_6.jpg)

다국어 지원으로 글로벌 사용자들도 편리하게 사용할 수 있습니다.

![언어 설정 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_7.jpg)

프로필 설정을 통해 나만의 아이덴티티를 표현할 수도 있습니다.

![프로필 편집 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_8.jpg)

프로필이 변경된 모습입니다.

![변경된 프로필 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_9.jpg)

### 지식 데이터베이스 구축

매너스의 주요 기능 중 하나는 바로 지식 관리 시스템입니다. [지식] 설정을 통해 에이전트가 참고할 개인 데이터를 관리할 수 있습니다.

![지식 메뉴 접근](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_10.jpg)

이 기능은 ChatGPT의 메모리와 유사합니다.

![지식 관리 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_11.jpg)

[지식추가] 버튼을 클릭하면 에이전트에게 학습시킬 새로운 지식을 추가할 수 있습니다. 이는 RAG(Retrieval-Augmented Generation) 기술의 실용적 구현으로 볼 수 있어요.

![지식 추가 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_12.jpg)

개인 정보를 추가한 후의 모습입니다. 이렇게 추가된 정보는 벡터 데이터베이스에 저장되어 관련 질문이 있을 때 컨텍스트로 활용됩니다.

![추가된 지식 목록](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_13.jpg)

### 고급 모델 선택 옵션

매너스는 사용 목적에 따라 다른 모델을 선택할 수 있는 유연성을 제공합니다. [표준]과 [고강도] 두 가지 모드가 있으며, 고강도 모드는 더 복잡한 작업에 적합하지만 처리 시간이 길고 사용 제한이 있습니다.

![모델 선택 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_14.jpg)

이런 모델 선택 옵션은 컴퓨팅 리소스의 효율적 사용과 사용자 경험 사이의 균형을 맞추기 위한 설계입니다. 일상적인 질문은 표준 모드로, 복잡한 코딩이나 심층 분석이 필요할 때는 고강도 모드를 선택하는 전략적 접근이 필요하죠.

## 리플레이 기능

매너스의 멋진 기능 중 하나는 바로 '리플레이' 시스템입니다. 초대코드가 없어도 다른 사용자들이 공유한 세션을 관찰할 수 있어, 간접적으로 매너스의 능력을 경험하고 배울 수 있습니다.

![리플레이 목록 화면](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_16.jpg)

리플레이의 진정한 가치는 단순히 결과를 보는 것이 아니라, AI 에이전트가 문제를 해결해 나가는 전체 사고 과정을 단계별로 관찰할 수 있다는 점입니다. 이는 AI의 블랙박스를 열어보는 듯한 경험을 제공합니다.

![리플레이 과정 시연](http://tykimos.github.io/warehouse/2025/2025-3-8-exploring_manus_ui_usage_and_replay_15.gif)

이 기능은 단순한 구경거리를 넘어 실제 개발자들에게 새로운 문제 해결 접근법과 프롬프트 엔지니어링 기법을 배울 수 있는 교육적 가치가 있습니다. 다양한 사례를 통해 매너스의 활용 가능성을 탐색해보세요!

## 마무리: 더 깊은 이해를 위한 여정

이제 매너스의 UI와 리플레이 기능에 대해 기본적인 이해를 갖추셨을 겁니다. 하지만 진정한 잠재력을 이해하기 위해서는 그 기술적 아키텍처와 동작 원리를 살펴볼 필요가 있습니다.

다음 글 [매너스 기술 및 아키텍처 심층 분석](https://tykimos.github.io/2025/03/08/in_depth_analysis_of_manus_technology_and_architecture)에서는 매너스의 작동 메커니즘에 대해 더 깊이 파헤쳐 보겠습니다. 개발자로서 이 부분에 더 흥미를 느끼실 거예요!

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

## 퍼가는 법
 
이 글은 자유롭게 퍼가셔도 좋아요! 다만 출처는 아래 링크로 꼭 남겨주세요 😊

[https://tykimos.github.io/2025/03/08/exploring_manus_ui_usage_and_replay/](https://tykimos.github.io/2025/03/08/exploring_manus_ui_usage_and_replay/)