---
layout: post
title: "매너스 도구들 - 브라우저"
author: 김태영
date: 2025-03-08 05:00:00
categories: [Manus, AI, Agent, Assistant, AssiWorks]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_title.jpg
---

매너스(Manus)는 복잡한 작업을 자동화하고, 다양한 데이터를 분석하기 위해 **여러 종류의 도구(Tool)**를 제공합니다. 그 중 **브라우저 도구(Browser)**는 실제 웹 브라우저와 유사한 방식으로 웹페이지에 접근하고, 화면을 스크롤하거나 엘리먼트를 클릭하는 등 **인터랙션**을 수행하는 핵심 기능입니다. 일반적인 **웹검색(Web Search)**과 달리, 브라우저 도구는 **검색 결과로 나온 웹사이트**나 **특정 URL**에 직접 진입해, 스크린샷을 캡처하거나, 자바스크립트를 실행하는 등 **심층적인 작업**을 할 수 있습니다.

<iframe width="100%" height="400" src="https://youtube.com/embed/..." title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>


![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_1.jpg)

---

## 브라우저 도구의 주요 액션(Action)

매너스의 브라우저 도구는 실제 사용자가 웹 브라우저에서 하는 행위를 에이전트가 자동으로 수행할 수 있게 합니다. 아래 표는 매너스 시스템에서 관찰된 브라우저 액션과 사용 횟수(지금까지 사용 사례를 기반으로 통계)를 예시로 정리한 것입니다:

| Action                | 사용 횟수 |
|-----------------------|----------:|
| **Scrolling down**    |       413 |
| **Browsing**          |       397 |
| **Clicking element**  |       146 |
| **Handling browser error** |    39 |
| **Running JavaScript**|         7 |
| **Typing**            |         7 |
| **Scrolling to bottom** |       4 |
| **Viewing the page**  |         1 |
| **Scrolling to top**  |         1 |

### Browsing
- **기본 액션**: 특정 웹사이트를 로딩하거나, 링크를 통해 새로운 페이지로 이동할 때 발생  
- 이 과정을 통해 **해당 페이지의 HTML**, **스크린샷**, **메타데이터** 등을 매너스 내부로 가져올 수 있습니다.

### Scrolling
- **Scrolling down / Scrolling to bottom / Scrolling to top**:  
  - 웹페이지가 길 때, 자동으로 스크롤을 내려(또는 올려) 원하는 위치까지 내용을 확인  
  - 동적 로딩되는 웹페이지에서 추가 데이터를 불러오거나, 광고·컨텐츠가 페이지 끝까지 로딩되는지 점검할 때 사용  

아래 화면은 스크롤 다운 예시입니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_2.jpg)

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_3.jpg)

### Clicking element
- **사용자 인터페이스 조작**: 버튼, 링크, 입력창 등 특정 요소를 클릭  
- 매너스는 엘리먼트 인덱스를 통해 클릭 대상을 지정함
- 이를 통해 폼 제출, 페이지 전환 등 실제 사용자 액션을 자동화할 수 있습니다.

아래 화면은 클릭 예시입니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_4.jpg)

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_5.jpg)

브라우저를 통해서 정보를 얻었다면 다시 웹검색을 통해서 정보를 습득합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_6.jpg)


### Typing
- **텍스트 입력**: 검색창이나 폼 필드에 특정 텍스트를 자동으로 타이핑  
- 검색, 폼 작성, 로그인 정보 입력 등 다양한 작업을 자동화할 때 활용됩니다.

아래 화면은 특정 요소에 타이핑하는 예시입니다.

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_7.jpg)

### Handling browser error
- **오류 처리**: 404, 500 등 웹페이지 오류 또는 스크립트 에러 발생 시 대응  
- 작업을 중단하거나 재시도하는 등의 로직이 들어갈 수 있습니다.

### Running JavaScript
- **직접 자바스크립트를 실행**: DOM을 제어하거나, SEO 요소(메타태그, canonical, schema 등)를 확인하는 스크립트를 수행  
- 이를 통해 웹페이지의 구조나 내부 데이터를 **프로그램적으로** 접근하고 분석 가능

지금까지 파악된 자바스크립트 실행 예시는 아래와 같습니다. 

- DOM 구조/메타 정보/SEO 요소를 분석

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_8.jpg)

- 페이지 성능(Navigation/Resource Timing) 정보 획득

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_11.jpg)

- 반응형/모바일 친화성 체크

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_12.jpg)

- 링크·이미지 깨짐 확인 및 품질 검사

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_9.jpg)

- 지도 API 활용 좌표 추출 등 위치 분석 : 위경도를 얻기 위해서 구글맵을 이용하여 자바스크립트를 실행시켜 정보를 가지고 옴

![img](http://tykimos.github.io/warehouse/2025/2025-3-8-manus_tools_browser_10.jpg)


### Viewing the page
- **페이지를 보는 행동**으로 추정됩니다.

---

## 브라우저 도구 vs. 웹검색 도구

- **웹검색(Web Search)**: 특정 키워드로 검색 엔진에서 **검색 결과 목록**을 가져옴
  - 예) `Searching for "best hiking trails Swiss Alps popular routes"`
- **브라우저(Browser)**: 이미 알고 있는 URL을 열어보거나, 검색 결과로 얻은 링크에 **직접 진입**하여 페이지를 열람
  - 예) `"Browsing https://www.some-website.com"` → 내부 컨텐츠 확인, 자바스크립트 실행, 폼 입력 등

두 도구는 상호보완적입니다.  
**1)** 먼저 웹검색으로 필요한 사이트나 자료를 찾고,  
**2)** 브라우저 도구로 해당 사이트에 들어가 **실제 작업**(버튼 클릭, 상세 데이터 수집)을 수행합니다.

---

## 브라우저 도구의 활용 예시

1. **데이터 크롤링**:  
   - 특정 사이트에 접속해, 자동 스크롤이나 자바스크립트 실행을 통해 **동적 로딩 데이터**까지 로딩 후, HTML이나 JSON 데이터를 추출  
2. **SEO 분석**:  
   - 자바스크립트 실행(`Running JavaScript`)으로 **canonical 태그**, **schema markup**, **메타 태그** 등을 확인  
3. **자동 양식 제출**:  
   - **Clicking element**와 **Typing** 액션을 조합해, 웹 폼에 정보를 입력하고 제출 버튼 클릭  
5. **UI 테스트**:  
   - 웹 애플리케이션의 기능 테스트(버튼 클릭 후 페이지 전환 확인 등)에 활용 가능

---

## 마무리

`브라우저(Browser)` 도구는 매너스에서 **가장 직관적인 웹 자동화**를 제공하는 기능입니다. 사용자가 직접 웹 브라우저에서 할 법한 **스크롤, 클릭, 입력** 등의 작업을 AI 에이전트가 대신 수행하므로,

- **반복 업무**(예: 크롤링, 폼 작성, 데이터 수집)에 큰 효율성  
- **사용자 개입 없이** 다양한 웹 시나리오 자동화 가능  

앞으로도 매너스는 **브라우저 도구**를 통해 더욱 다양한 웹 상호작용을 지원하고, 동시에 사용자 인증·보안·대규모 트래픽 처리 등에 대한 개선을 진행할 것으로 기대됩니다.

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

[https://tykimos.github.io/2025/03/08/manus_tools_browser](https://tykimos.github.io/2025/03/08/manus_tools_browser)