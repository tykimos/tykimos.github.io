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

### 예시 1. 디렉터리 생성

- **파일명:** gEB0v1elxoKSESkfREyXZ7.json  
- **Message Action:** Executing command  
- **Terminal Action:** execute  
- **명령어:** `mkdir -p openai_org_chart`

```json
"description": "Executed command `mkdir -p openai_org_chart`",
"message": {
  "action": "Executing command",
  "param": "mkdir -p openai_org_chart"
}

mkdir -p를 통해 폴더를 생성하여 향후 문서파일을 저장할 구조를 만듭니다.
프로젝트 초기화나 환경 셋업 스크립트에서 자주 등장합니다.

### 예시 2. 패키지 설치
파일명: AN3pVVcq4qHnWbC55XhAYB.json
Message Action: Executing command
Terminal Action: execute
명령어: pip3 install pandas openpyxl matplotlib
json
복사
"description": "Executed command `pip3 install pandas openpyxl matplotlib`"
파이썬 라이브러리 설치 명령으로, 데이터 처리와 시각화를 위한 기본 라이브러리 사용 준비를 마칩니다.
매너스는 이 과정을 자동으로 진행해, 분석 환경을 빠르게 구축할 수 있습니다.

### 예시 3. Python 스크립트 단발 실행
파일명: AN3pVVcq4qHnWbC55XhAYB.json
Message Action: Executing command
Terminal Action: execute
명령어:
bash
복사
python3 -c "import pandas as pd; print(pd.read_excel('/home/ubuntu/upload/interview_survey_final.xlsx').head())"
json
복사
"param": "python3 -c \"import pandas as pd; print(pd.read_excel('/home/ubuntu/upload/interview_survey_final.xlsx').head())\""
터미널에서 파이썬 단발성 스크립트(-c 옵션)를 통해 엑셀 파일의 상위 5행을 출력합니다.
빠른 데이터 확인 및 디버깅에 매우 유용합니다.

### 예시 4. 스크립트 파일 실행 + 대기
파일명: AN3pVVcq4qHnWbC55XhAYB.json
명령어: python3 /home/ubuntu/analyze_survey.py
설명: "Executing command python3 /home/ubuntu/analyze_survey.py and waited 30 seconds"
json
복사
"detail": {
  "terminal": {
    "action": "execute",
    "finished": false,
    "shellId": "shell1",
    "command": "python3 /home/ubuntu/analyze_survey.py"
  }
}
장시간 걸리는 스크립트(분석, 모델 학습)를 실행 후 대기 상태에 들어갑니다.
일정 시간(30초)이 지났는데도 결과가 없으면 진행 상황을 재점검하거나 중단할 수 있습니다.

### 예시 5. Python 스크립트 재실행
파일명: AN3pVVcq4qHnWbC55XhAYB.json
Message Action: Executing command
Terminal Action: execute
명령어:
bash
복사
python3 -c "import pandas as pd; df = pd.read_excel('/home/ubuntu/upload/interview_survey_final.xlsx'); print(df)"
이전 단계에서 설치한 pandas 및 openpyxl 라이브러리를 활용해 엑셀 내용을 전체 출력합니다.
이처럼 매너스는 프로젝트 환경을 자동 구성한 뒤, 데이터 처리→결과 확인 과정을 쉼 없이 이어갈 수 있습니다.

---

## 터미널 도구 활용 전략

프로젝트 초기화

디렉터리 생성(mkdir -p), git clone 등을 통해 코드 저장소를 초기화
pip3 install 명령어로 필요한 라이브러리를 자동 설치
데이터 분석 자동화

Python 스크립트를 실행하여 CSV/Excel 데이터를 전처리, 통계 분석, 시각화
일회성 스크립트(python3 -c "...")를 통해 결과를 빠르게 확인
장시간 작업 관리

대기(wait) 및 프로세스 종료(kill_process) 기능을 통해 장기 분석이나 모델 트레이닝을 모니터링
중간 점검이 필요할 때 결과를 부분 출력하거나, 일정 시간이 지나면 자동 중단
CI/CD 파이프라인 연계

터미널을 이용해 테스트 실행, 도커 이미지 빌드, 서버 배포 등 DevOps 작업을 자동화
AI가 빌드 실패 원인을 분석하고, 수정할 부분을 찾아주는 방식과도 연계 가능
오류 처리

Handling terminal error 로그를 통해 오류 발생 시점과 내용을 추적
필요한 경우 재시도, 대체 명령 실행, 또는 사용자 알림

---

## 마무리

매너스의 터미널(Terminal) 도구는 시스템 명령어 실행을 자동화해 개발, 배포, 데이터 분석 등 다양한 업무의 생산성을 높여줍니다. 특히 다음과 같은 장점이 돋보입니다:

다양한 환경에 즉시 적용 가능: 리눅스 기반 서버, 도커 컨테이너, 클라우드 인스턴스 등에서 동일하게 작동
유연한 명령 체인 구성: 라이브러리 설치 → 스크립트 실행 → 결과 파일 처리 등 순차적 자동화
장시간 작업 대응: 명령어 대기와 프로세스 강제 종료를 통해 안정적으로 작업을 제어
CI/CD 및 DevOps 파이프라인과 연동: 테스트, 빌드, 배포 과정을 AI가 자동화
앞으로도 매너스는 터미널 도구를 더욱 고도화해, 서버 및 클라우드 환경 전반에 걸쳐 원격 조작 및 관리를 지원하도록 발전해 나갈 것으로 기대합니다. 🚀

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

[https://tykimos.github.io/2025/03/08/manus_tools_terminal](https://tykimos.github.io/2025/03/08/manus_tools_terminal)