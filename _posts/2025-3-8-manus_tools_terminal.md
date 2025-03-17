---
layout: post
title: "매너스 도구들 - 터미널"
author: 김태영
date: 2025-03-08 07:00:00
categories: [Manus, AI, Agent, Assistant, AssiWorks]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_title.jpg
---

매너스(Manus)는 여러 종류의 **도구(tools)**를 통해 작업을 자동화하는 **멀티 에이전트 기반** AI 에이전트입니다. 그중 **터미널(Terminal)** 도구는 프로젝트 디렉터리 생성, 패키지 설치, 스크립트 실행 등 **시스템 명령어를 직접 실행**하는 역할을 담당합니다. 이 글에서는 매너스에서 터미널 도구가 어떻게 사용되는지 함께 살펴보겠습니다.

<iframe width="100%" height="400" src="https://youtube.com/embed/..." title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_3.jpg)

---

## 개요

이 도구는 리눅스/유닉스 계열 명령어를 호출하여 **폴더 생성**, **파일 다운로드**, **소프트웨어 패키지 설치**, **파이썬 스크립트 실행** 등 다양한 작업을 수행합니다. 주요 장점은 아래와 같습니다.

- **CI/CD 파이프라인 연계**: 코드를 빌드하고, 테스트를 자동화
- **데이터 처리**: 외부 데이터 다운로드, CSV/Excel 변환, 파이썬 스크립트 실행
- **프로젝트 초기화**: `mkdir -p ...`, `git clone ...` 등의 명령어를 통한 자동 프로젝트 구성

---

## 메시지 액션 분류

아래는 터미널이 호출될 때, **사용자가 의도한 명령**을 기준으로 분류한 결과입니다. 사용횟수는 현재까지 나온 사용 사례를 기반으로 분석한 것입니다.

| Action                 | 사용 횟수 |
|------------------------|----------|
| **Executing command**  | 601      |
| **Waiting for terminal** | 14      |
| **Terminating process** | 6       |
| **Handling terminal error** | 4    |
| **Viewing terminal**   | 4       |

1. **Executing command**: 가장 많이 등장하는 유형으로, 특정 셸 명령어를 실행하는 작업  
2. **Waiting for terminal**: 터미널 응답이나 프로세스 완료를 대기하는 동작  
3. **Terminating process**: 실행 중인 프로세스를 종료  
4. **Handling terminal error**: 터미널 사용 중 발생한 에러를 처리  
5. **Viewing terminal**: 현재 터미널 상태(출력 결과 등)를 확인

---

## 터미널 내부 동작 분류

메시지 액션과 달리, **터미널 내부에서 실제로 수행된 작업**은 다음과 같습니다.

| Action        | 사용 횟수 |
|---------------|----------|
| **execute**   | 601      |
| **wait**      | 14       |
| **kill_process** | 6     |
| **view**      | 4        |

- **execute**: `mkdir`, `python3`, `pip3 install` 등 특정 명령어를 실행  
- **wait**: 명령어 실행 후 일정 시간 대기  
- **kill_process**: 시간 초과, 사용자 요청 등에 의해 프로세스를 강제 종료  
- **view**: 현재 터미널 상태나 출력 내용을 보는 동작

---

## 메시지 액션과 내부 동작의 조합

실제로 사용자 또는 AI가 의도한 명령(Message Action)과 실제 내부 동작(Terminal Action) 간의 매핑을 보면 다음과 같습니다:

| Message Action         | Terminal Action | 사용 횟수 |
|------------------------|----------------|----------|
| **Executing command**  | execute        | 601      |
| **Waiting for terminal** | wait          | 14       |
| **Terminating process** | kill_process  | 6        |
| **Handling terminal error** | 지정되지 않음 | 4      |
| **Viewing terminal**   | view          | 4        |

주요 포인트:
- **Executing command** → **execute** 조합이 대부분을 차지 (601회)  
- **Waiting for terminal** → **wait**, **Terminating process** → **kill_process**가 자연스럽게 매칭  
- 에러 발생 시 `Handling terminal error`가 기록되지만, 내부 동작은 지정되지 않은 경우가 많음  

---

## 터미널 사용 샘플 분석

매너스가 터미널을 어떻게 활용하여 작업을 자동화하는지 예시를 통해 확인해보겠습니다.

### 디렉터리 생성

- **Message Action:** Executing command  
- **Terminal Action:** execute  
- **명령어:** `mkdir -p openai_org_chart`

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_4.jpg)

mkdir -p를 통해 폴더를 생성하여 향후 문서파일을 저장할 구조를 만듭니다. 프로젝트 초기화나 환경 셋업 스크립트에서 자주 등장합니다.

### 패키지 설치

- **Message Action:** Executing command  
- **Terminal Action:** execute  
- **명령어:** `pip3 install pandas openpyxl matplotlib`

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_5.jpg)

파이썬 라이브러리 설치 명령으로, 데이터 처리와 시각화를 위한 기본 라이브러리 사용 준비를 마칩니다. 매너스는 이 과정을 자동으로 진행해, 분석 환경을 빠르게 구축할 수 있습니다.

### Python 스크립트 단발 실행

- **Message Action:** Executing command  
- **Terminal Action:** execute  
- **명령어:** `python3 -c "import pandas as pd; print(pd.read_excel('/home/ubuntu/upload/interview_survey_final.xlsx').head())"`

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_6.jpg)

터미널에서 파이썬 단발성 스크립트(-c 옵션)를 통해 엑셀 파일의 상위 5행을 출력합니다. 빠른 데이터 확인 및 디버깅에 매우 유용합니다.

위 예시에서는 사용자가 파일을 첨부하는 것부터 시작하는 데요. 이 과정을 참고로 살펴보도록 하겠습니다. 사용자가 파일을 첨부하면 샌드박스를 초기화한다음 파일을 업로드 합니다. 그 다음 터미널이 해당 파일을 읽기 위한 패키지를 샌드박스에 설치합니다. 그 후 파일을 읽은 다음 그 내용을 오케스트레이터에게 전달합니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_7.jpg)

### 다이어그램 생성 (Python 기반)

터미널을 통해서 다이어그램을 생성할 수 있습니다. 다이어그램은 파이썬의 matplotlib 라이브러리를 통해 생성합니다. 아래 화면은 다이어그램을 그리기 위해 파이썬 스크립트를 생성하는 모습입니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_8.jpg)

아래 그림을 통해 생성한 다이어그램입니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_9.jpg)

### 마누스 샌드박스 용량 체크

프로젝트를 구성하거나 패키지를 설치할 가용한 용량을 사전에 체크합니다. 현재 샌드박스는 13G의 용량으로 제공되고 있습니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_10.jpg)

### 음성 인식 및 텍스트 추출

음성 인식 및 텍스트 추출을 위해 터미널을 호출합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_11.jpg)

### 히트맵 시각화

데이터를 분석하여 다양한 시각화 처리를 수행합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_12.jpg)

### 아이콘 셋 생성

웹사이트를 분석해서 다양한 아이콘셋을 생성합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_13.jpg)

### 웹사이트 배포

웹사이트를 생성하여 배포까지 진행합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_14.jpg)

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_terminal_15.jpg)

위 예시에서 실제로 배포된 웹사이트는 아래 링크에서 확인할 수 있습니다.
* [Quantum Computing Learning Hub](https://evvqfqoz.manus.space/)
* [Rubber Mats Price Comparison Dashboard](https://zvgzsafz.manus.space/)

---

## 마무리

매너스의 터미널(Terminal) 도구는 시스템 명령어 실행을 자동화해 개발, 배포, 데이터 분석 등 다양한 업무의 생산성을 높여줍니다.

- 프로젝트 초기화
-- 디렉터리 생성(mkdir -p), git clone 등을 통해 코드 저장소를 초기화
-- pip3 install 명령어로 필요한 라이브러리를 자동 설치

- 데이터 분석 자동화
-- Python 스크립트를 실행하여 CSV/Excel 데이터를 전처리, 통계 분석, 시각화
-- 일회성 스크립트(python3 -c "...")를 통해 결과를 빠르게 확인

- 장시간 작업 관리
-- 대기(wait) 및 프로세스 종료(kill_process) 기능을 통해 장기 분석이나 모델 트레이닝을 모니터링
-- 중간 점검이 필요할 때 결과를 부분 출력하거나, 일정 시간이 지나면 자동 중단

- 오류 처리
-- Handling terminal error 로그를 통해 오류 발생 시점과 내용을 추적
-- 필요한 경우 재시도, 대체 명령 실행, 또는 사용자 알림

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

## 한국의 온프레미스 노코드 에이전틱AI 플랫폼

AIFactory에서도 에이전틱AI 플랫폼을 서비스 및 고도화하고 있습니다. 어시웍스(AssiWorks)는 “도구(Tools)”, “워크플로우(Flows)”, “에이전트(Agents)”, “팀(Teams)”이라는 네 가지 주요 개념을 중심으로, 노코드(No-Code) 환경에서 AI 기반 업무 자동화와 협업형 에이전트 구성을 손쉽게 구현할 수 있도록 지원하는 종합 플랫폼입니다. 

자세히 보기 >> [어시웍스](https://aifactory.space/guide/8/14)

![어시웍스](http://tykimos.github.io/warehouse/2025/2025-3-8-assiworks.png)

## 퍼가는 법
 
이 글은 자유롭게 퍼가셔도 좋아요! 다만 출처는 아래 링크로 꼭 남겨주세요 😊

[https://tykimos.github.io/2025/03/08/manus_tools_terminal](https://tykimos.github.io/2025/03/08/manus_tools_terminal)