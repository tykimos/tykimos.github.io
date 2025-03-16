---
layout: post
title: "매너스 도구들 - 문서편집기"
author: 김태영
date: 2025-03-08 06:00:00
categories: [Manus, AI, Agent, Assistant, AssiWorks]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_title.jpg
---

매너스(Manus)는 멀티 에이전트 기반으로 동작하며, 다양한 **도구(tools)**를 활용해 작업을 수행하는 범용 AI 에이전트입니다. 그중 **문서편집기(Text Editor)**는 문서 생성과 수정, 내용을 읽어들이는 기능을 담당하는 핵심 도구입니다. 매너스가 연구 보고서 작성, 기획 문서 작성, 프로젝트 TODO 정리 등 **텍스트를 활용한 작업**을 자동화할 때 자주 사용됩니다.

<iframe width="100%" height="400" src="https://youtube.com/embed/..." title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_1.jpg)

이 도구는 사용자나 AI가 직접 문서를 작성·편집·확인하는 과정을 자동화할 수 있게 해주며, 특히 긴 형태의 텍스트를 다룰 때 효율적으로 활용됩니다.사용자는 문서 작업에 들어가는 시간을 크게 절약할 수 있습니다.

---

## 주요 사용 유형

문서편집기 도구가 호출될 때, **어떤 의도(action)**로 사용되었는지를 기록한 항목입니다.  

| Action                     | 사용 횟수 |
|---------------------------|----------|
| **Creating file**         | 642      |
| **Editing file**          | 570      |
| **Reading file**          | 77       |
| **Adding content to file**| 21       |
| **Handling text editor error** | 9 |

1. **Creating file**: 새 문서를 생성하는 작업(예: 프로젝트 초기 설정, README나 TODO 파일 생성)이 가장 많았습니다.  
2. **Editing file**: 이미 존재하는 문서를 수정하는 작업이 그다음으로 빈도가 높았습니다.  
3. **Reading file**: 문서 내용을 확인하기 위한 조회 작업입니다.  
4. **Adding content to file**: 기존 문서에 텍스트를 추가 삽입하는 것으로, 전체 내용을 다시 쓰는 것이 아니라 일부만 덧붙이는 작업으로 파악됩니다.  
5. **Handling text editor error**: 에디터 사용 중 발생하는 오류를 처리하는 동작입니다.

텍스트 에디터 세부 동작 유형(action)을 기준으로 보면 다음과 같이 구분됩니다.

- **write**: 파일에 **새로운 텍스트**를 작성하거나, 파일 생성 시 최초 내용을 쓰는 행위  
- **update**: **기존 텍스트**를 수정·갱신하는 행위  
- **read**: 파일의 **내용을 읽는** 행위  

실제로 `의도`(사용자가 의도한 명령)과 `행동`(내부적으로 어떤 작업이 수행되었는지) 간의 매핑을 보면 다음과 같습니다.

| Message Action            | Editor Action |
|---------------------------|--------------|
| **Creating file**         | write        |
| **Editing file**          | update       |
| **Reading file**          | read         |
| **Editing file**          | write        |
| **Adding content to file**| write        |
| **Handling text editor error** | 지정되지 않음 |

주요 포인트:

- **Creating file** → **write** 패턴이 가장 많이 등장하며, 새 문서를 만들 때 처음 내용을 쓴다고 볼 수 있습니다.  
- **Editing file** → **update**가 기본적이나, 실제로는 **write**도 많이 사용되었음을 알 수 있습니다. (파일 전체를 재작성했거나, 부분 추가가 동작 상 `write`로 처리된 경우)  
- **Reading file** → **read**로 일대일 대응됩니다.

---

## 3. 텍스트 에디터 사용 샘플 분석

아래는 실제 로그 파일을 요약한 **5가지 사용 예시**로, 매너스가 문서를 어떻게 작성·편집·활용하는지 알 수 있습니다.

### 새 파일 생성 후 내용 작성
- **Message Action:** Creating file  
- **Detail Action:** write  
- **파일 경로:** `/home/ubuntu/openai_org_chart/todo.md`  
- **용도:** OpenAI 조직도 리서치용 TODO 리스트

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_2.jpg)

여기서는 새로운 Markdown 문서를 만들고 연구 계획과 작업 목록 등을 자세히 적어둔 모습입니다. 연구 단계별 진행 상황을 체크해 실시간으로 TODO 문서에 반영합니다. 이를 통해 AI와 협업하며, 관리자가 문서 상태를 언제든지 확인할 수 있게 됩니다. 프로젝트의 디렉터리 구조를 생성(terminal)한 뒤, 즉시 문서를 생성(text_editor)하고, TODO 리스트를 채우는 작업을 한 번에 자동화할 수 있습니다.

### 기존 파일 수정
- **Message Action:** Editing file  
- **Detail Action:** update  
- **파일 경로:** `/home/ubuntu/openai_org_chart/todo.md`  
- **특이사항:** 작성된 TODO 리스트 중 일부 완료 체크박스를 업데이트

이미 생성된 Markdown 파일에서 특정 작업 항목을 완료로 표시하는 등 내용 업데이트를 수행합니다. 이처럼 매너스는 문서 편집 과정을 자동화하고, 작성된 내용 중 완료된 항목을 체크하거나, 새 항목을 추가하는 식으로 쉽게 수정합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_3.jpg)

### 리포트 데이터 문서 생성
- **Message Action:** Creating file  
- **Detail Action:** write  
- **파일 경로:** `/home/ubuntu/openai_org_chart/leadership_data.md`  
- **내용:** OpenAI의 주요 임원, 최근 퇴사자 정보 등을 구조적으로 정리

여기서는 별도의 파일에 리포트를 Markdown 형식으로 정리합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_4.jpg)

### 차트 문서 생성
- **Message Action:** Creating file  
- **Detail Action:** write  
- **파일 경로:** `/home/ubuntu/openai_org_chart/departments_data.md`  
- **내용:** 부서별 팀 구조, 운영 원칙, 사무 공간 배치 등 포괄적 자료

조직을 세분화해 파악하는 작업으로, 부서별 운영 방식, 인력 구조를 문서에 담고 있습니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_5.jpg)

### 차트 문서 생성
- **Message Action:** Creating file  
- **Detail Action:** write  
- **파일 경로:** `/home/ubuntu/openai_org_chart/departments_data.md`  
- **내용:** 부서별 팀 구조, 운영 원칙, 사무 공간 배치 등 포괄적 자료

조직을 세분화해 파악하는 작업으로, 부서별 운영 방식, 인력 구조를 문서에 담고 있습니다.

### 코딩 작성 및 결과 확인

코딩이 필요할 경우 파이썬 코드를 작성하고, 터미널에 전달하여 파이썬을 실행시킵니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_6.jpg)

파이썬 결과 파일을 읽을 때는 텍스트 에디터를 활용해서 문서를 읽습니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_7.jpg)


### Markdown Extended (mdx) 문서 작성

MDX는 Markdown Extended의 약자로, 일반 Markdown(`.md`)에 JSX(JavaScript XML) 문법을 추가로 지원하는 확장 포맷입니다. 매너스는 이러한 MDX 문서 작성과 편집도 원활하게 지원합니다.

**MDX의 주요 특징:**
- Markdown 기본 문법과 JSX 구문을 함께 사용 가능
- React 컴포넌트를 문서 내에 직접 삽입 가능
- 동적 콘텐츠를 포함한 기술 문서 작성에 최적화

**활용 사례:**
- Next.js, Gatsby 등 React 기반 프레임워크의 문서화
- 대화형 기술 문서 작성
- 컴포넌트 라이브러리 예제 구현 (Storybook)
- 프레젠테이션 슬라이드 제작 (MDX Deck)

다음은 매너스가 작성한 MDX 문서 예시입니다. 차트인 경우 마우스를 올리면 해당 영역의 값을 확인할 수 있습니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_8.jpg)

### PDF 읽기

사용자가 PDF 파일을 업로드했을 경우, 터미널에서 pdftotext 명령어를 통해 PDF 파일을 텍스트로 변환합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_9.jpg)

### 환경 설정

매너스는 분석을 위한 환경을 설정할 때 디스크 용량을 고려하여 최적의 라이브러리를 사용합니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_10.jpg)

### 음성 인식 및 타입스탬프 추출

매너스는 음성 인식 및 타입스탬프 추출할 때 터미널을 사용합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_11.jpg)

### 다양한 데이터 시각화 생성

매너스는 matplotlib 라이브러리를 통해 다양한 데이터 시각화 이미지를 생성합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_12.jpg)

### 아이콘 생성

매너스는 웹사이트를 분석해서 다양한 아이콘셋을 생성합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_13.jpg)

### 웹사이트 배포

매너스는 웹사이트를 생성하여 배포까지 진행합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_text_editor_14.jpg)

---

## 효과적인 활용 전략

- **자동 문서 생성 및 관리**: 프로젝트 초기 단계에서 README, TODO, 설계 문서 등을 자동으로 생성하고, 작업 진행 상황에 따라 문서를 즉시 업데이트합니다. 이는 팀원 간 협업 효율성을 극대화하고 프로젝트 추적을 용이하게 합니다.
- **Markdown 기반 문서화 시스템**: Markdown 형식은 가볍고 표현력이 뛰어나며 웹 환경으로의 전환이 용이합니다. 표, 차트, 이미지, 체크박스, 링크 등 다양한 요소를 손쉽게 배치할 수 있어 복잡한 정보도 명확하게 전달할 수 있습니다.
- **터미널 도구와의 원활한 연계**: 터미널 도구와 함께 사용하면 파일 구조 생성 후 즉시 문서를 작성하거나, 코드 실행 결과를 문서에 자동 삽입하는 등 개발 환경과 문서화 작업의 통합이 가능합니다. 이를 통해 개발-문서화 워크플로우가 끊김 없이 진행됩니다.
- **실시간 문서 업데이트 및 버전 관리**: 매너스는 문서를 작성하거나 수정할 때마다 자동으로 버전을 관리합니다. 수정 시점, 변경 내용 등을 체계적으로 기록하여 회의록, TODO 리스트, 참고 자료 등의 변경 이력을 손쉽게 추적할 수 있습니다.

## 결론

매너스의 문서편집기(Text Editor) 도구는 문서 생성·수정·읽기를 모두 자동화함으로써, 기술 문서, 보고서, TODO 리스트, 설계 문서 등을 효율적으로 관리할 수 있도록 지원합니다. 특히 다음과 같은 장점이 두드러집니다:

* 자동화된 워크플로우: 프로젝트 디렉터리를 생성 → 문서 생성 → 내용 수정 과정을 한 번에 처리
* 다채로운 문서 형식 지원: Markdown, 텍스트, 코드 파일 등 다양한 형식
* 협업 시 편리성: TODO 체크박스, 일정 관리, 회의록 작성 등 실시간 협업이 가능
* 결과적으로 팀 단위 문서 작업, 장기 프로젝트 관리, 데이터 기반 리포트 작성 등 분야에서 뛰어난 효율성을 보입니다. 앞으로도 매너스는 텍스트 기반 업무에 필요한 기능을 계속 진화시켜, 한층 더 강력한 문서 작업 자동화를 제공할 것입니다. 🚀

---


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

[https://tykimos.github.io/2025/03/08/manus_tools_text_editor](https://tykimos.github.io/2025/03/08/manus_tools_text_editor)