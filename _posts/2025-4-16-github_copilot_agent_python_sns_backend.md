---
layout: post
title: "GitHub Copilot 에이전트로 SNS 파이썬 백엔드 만들기"
author: 김태영
date: 2025-04-16 08:00:00
categories: [GitHub, Copilot, Python, Backend, SNS]
comments: true
image: http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_title.jpg
---

GitHub Copilot 에이전트는 개발자의 생산성을 크게 향상시켜주는 AI 코드 어시스턴트입니다. 이 글에서는 GitHub Copilot 에이전트를 활용하여 파이썬 기반 SNS 백엔드 애플리케이션을 구축하는 과정을 단계별로 살펴보겠습니다.

<iframe width="100%" height="400" src="https://youtube.com/embed/..." title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>

1. 빈 폴더를 하나 만듭니다. 본 예제에서는 "sns_python_backend"이란 이름으로 만들었습니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_0.jpg)

2. "Visual Studio Code - Insiders" 어플리케이션을 실행시킨 후 "File > Open Folders" 메뉴에서 선택 후에 1번에서 만든 폴더를 선택합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_1.jpg)


3. 그럼 아래와 같이 빈 프로젝트이 준비됩니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_2.png)

4. 새 파일 아이콘을 클릭한 후 "main.py" 파일을 생성합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_3.png)

5. "openapi.yaml"을 아래 링크에서 다운로드 받은 뒤 "openapi.yaml" 파일명으로 수정 후에 프로젝트에 추가합니다.

[openapi.yaml](2025-4-16-github_copilot_agent_python_sns_backend_openapi.yaml)

아래는 "openapi.yaml" 파일을 추가한 화면입니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_4.png)

openapi.yaml 파일은 우리가 만들 웹 API의 설계도라고 생각하면 됩니다. 마치 건물을 짓기 전에 건축 도면이 필요한 것처럼, API를 개발하기 전에 이 파일로 전체 구조를 미리 계획합니다.

이 파일에서는 다음과 같은 내용을 정의합니다:
- 어떤 기능들(포스트 작성, 댓글 달기, 좋아요 등)을 만들지
- 각 기능을 사용하려면 어떤 주소(URL)로 요청해야 하는지
- 요청할 때 어떤 정보가 필요한지(예: 포스트 내용, 사용자 이름)
- 요청 후 어떤 응답이 돌아오는지

초보자에게 가장 큰 장점은:
- 개발 전에 API 구조를 명확히 볼 수 있어 전체 그림을 이해하기 쉽습니다
- 여러 개발자가 같은 규칙으로 작업할 수 있습니다
- 자동으로 API 문서와 테스트 페이지가 생성됩니다
- 클라이언트(앱이나 웹) 개발자와 서버 개발자 간의 소통이 쉬워집니다

마치 레고 조립 설명서처럼, 어떤 부품이 어디에 들어가는지 미리 알려주는 역할을 합니다.

openapi.yaml은 SNS 앱을 위한 API 설계도로, 다음 기능들을 정의합니다:

* 포스트 관련 API (5개)
** 포스트 목록 조회, 작성, 상세 조회, 수정, 삭제
* 댓글 관련 API (5개)
** 댓글 목록 조회, 작성, 상세 조회, 수정, 삭제
* 좋아요 관련 API (2개)
** 좋아요 추가, 취소

각 API는 필요한 요청 정보와 응답 형식, 오류 처리 방법이 명확히 정의되어 있어 개발 시 일관된 인터페이스를 구현할 수 있습니다.

6. GitHub Copilot Agent 모드를 활성화 합니다. 상단에 있는 GitHub Copilot 아이콘을 클릭한 후 하단에 첫 옵션에서 "Agent"을 선택합니다. LLM(대규모언어모델)은 최신 모델로 선택합니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_5.png)

7. 왼쪽 프로젝트 창에서 "main.py"을 클릭합니다. 그럼 오른쪽 GitHub Copilot 채팅창 하단에 "main.py (Current file)"라고 표시됩니다. 이것은 GitHub Copilot가 현재 main.py를 보고 있다는 뜻입니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_6.png)

8. 다음은 "openapi.yaml" 파일을 GitHub Copilot 문맥에 추가합니다. 프로젝트 파일에서 "openapi.yaml"을 GitHub Copilot 채팅창으로 드래그앤드롭 합니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_7.png)

그럼 아래와 같이 GitHub Copilot가 셋팅됩니다. 현재 Agent가 보고 있는 파일은 "main.py"이고 문맥에 "openapi.yaml" 파일이 추가된 상태입니다. 사용자가 채팅으로 명령을 내리면 Agent는 "openapi.yaml" 참고하여 "main.py"에 코드를 작성합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_8.png)

9. 이제 준비는 다 되었습니다. 간단한 파이썬 백엔드를 작성해보겠습니다. 먼저 CRUD API를 만들어봅니다. 아래 프롬프트를 Agent에 입력합니다.

```text
FastAPI를 사용하여 openapi.yaml 명세를 기반으로 소셜 네트워크 서비스의 백엔드 API를 구현해. openapi.yaml의 정의에 따라 포스트, 댓글, 좋아요 기능을 포함하는 모든 엔드포인트를 구현한다. 명세에 맞게 적절한 HTTP 메소드, 상태 코드, 요청/응답 형식을 구현하낟. Pydantic 모델을 사용하여 데이터 검증을 구현하고, 예외 처리도 포함해주세요. 데이터는 데이터베이스 없이 간단한 메모리로 관리한다. main.py 파일 하나에 모두 구현한다.
```

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_9.png)

중간에 환경셋팅이나 실행 관련된 부분이 나오면 수행하지 않습니다. 코드를 먼저 작성한 뒤 따로 실행해보도록 하겠습니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_10.png)

10. agent가 코드를 작성했다면 이를 유지할 지 취소할 지 사용자에게 묻습니다. 여기서 "keep"을 선택합니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_11.png)

11. 이제 아래와 같이 터미널을 실행시킵니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_12.png)

아래 명령을 차례차례 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pydantic
uvicorn main:app --reload
```

 각 행의 설명은 다음과 같습니다.

* 가상환경 생성
* 가상환경 실행
* 가상환경 내에 필요한 패키지 설치
* 백엔드 실행

12. 실행이 정상적으로 되었다면 아래 화면처럼 출력됩니다. "http://127.0.0.1:8000" 접속 링크가 띄워지는 것이 정상입니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_13.png)

만약 8000 포트로 다른 서비스가 구동되고 있다면 에러가 발생합니다. 이 경우에는 아래처럼 다른 포트로 구동시킵니다. 

```bash
uvicorn main:app --reload --port 8889
```

13. "http://127.0.0.1:8000/docs"에 접속해서 API가 정상적으로 구동되는 지 확인합니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_14.png)

14. "GET /api/posts"을 클릭하여 실행하면 포스트 목록이 나오는 데, 현재는 빈배열이 출력됩니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_15.png)

15. "POST /api/posts"을 클릭하여 포스트를 하나 생성합니다. "Try it out" 클릭한 다음 "Request body" 부분에 아래와 같이 JSON 형태로 입력한 후 "Execute" 버튼을 클릭합니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_16.png)

정상적으로 실행되면 아래와 같이 출력이 나옵니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_17.png)

16. 다시 "GET /api/posts"을 클릭해보면 15번에 입력한 포스트 내용이 출력됨을 확인할 수 있습니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_18.png)

이와 같이 댓글이나 좋아요 API도 테스트합니다. 실제 서비스에서는 앱이나 웹페이지(프론트엔드라고 부름)의 UI에서 사용자가 액션을 하면, 이러한 프론트엔드에서 백엔드에서 제공하는 API를 호출하여 구동되는 방식입니다. 

17. 현재 이 방식은 서비스가 종료될 경우 지금까지 등록한 포스트나 댓글, 좋아요 등의 정보가 모두 사라집니다. 서비스를 재시작하더라도 이전 정보를 저장하려면 "데이터베이스"가 필요합니다. 다음 프롬프트를 통해서 이제 백엔드에 데이터베이스를 추가해봅니다.

```text
SQLite 데이터베이스를 사용하여 openapi.yaml 명세를 참고하여 현재 정의된 API를 구현한다. 애플리케이션 시작 시 자동으로 데이터베이스와 테이블을 생성하는 코드를 구현하되, 테이블이 이미 존재하는 경우 다시 생성하지 않도록 'CREATE TABLE IF NOT EXISTS' 구문을 사용하세요. 
```

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_19.png)

18. 실행결과 데이터베이스를 이용한 백엔드 코드가 작성됩니다. 적용되고 있는 화면은 아래와 같습니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_20.png)

어디에 어떻게 수정을 할 것인지 표시가 됩니다. "Keep" 버튼을 클릭하면 모두 적용됩니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_21.png)

서비스를 시작할 때 "uvicorn main:app --reload"으로 실행했다면 "--reload" 옵션에 의해 소스코드가 변경되면 자동으로 서비스가 재시작 됩니다. 

이 상태에서 15번대로 포스트를 하나 추가를 해봅니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_22.png)

그 다음 "Ctrl+C"를 눌러서 서비스를 종료 후에 다시 "uvicorn main:app --reload"으로 실행시킵니다.

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_23.png)

19. 14번 항목을 다시 수행해서 서비스가 종료 후에도 사용자가 등록한 포스트가 그대로 출력이 되는 것을 확인할 수 있습니다. 

![img](http://tykimos.github.io/warehouse/2025/2025-4-16-github_copilot_agent_python_sns_backend_24.png)

## 마무리

GitHub Copilot 에이전트를 활용하여 Python 기반 SNS 백엔드 애플리케이션을 개발해 보았습니다. 이 과정을 통해 알 수 있듯이, GitHub Copilot 에이전트는 단순한 코드 자동 완성 도구를 넘어 다음과 같은 강력한 기능을 제공합니다:

1. **OpenAPI 명세 기반 구현**: OpenAPI 명세(YAML 파일)를 이해하고 이를 기반으로 완전한 백엔드 API를 자동으로 구현할 수 있습니다. 이는 API 설계와 구현 사이의 간극을 줄여줍니다.

2. **코드 진화 지원**: 처음에는 메모리 기반의 간단한 구현을 제공하고, 요구사항이 변경됨에 따라 SQLite 데이터베이스를 활용한 영구 저장소 구현으로 코드를 진화시킬 수 있습니다.

3. **완전한 CRUD 기능 구현**: 포스트, 댓글, 좋아요 등의 기능에 대해 생성, 조회, 수정, 삭제 기능을 모두 갖춘 API를 빠르게 구현할 수 있습니다.

4. **올바른 코드 패턴 적용**: Pydantic 모델을 사용한 데이터 검증, 적절한 예외 처리, HTTP 상태 코드 등 FastAPI의 모범 사례를 자동으로 적용합니다.

이러한 장점들은 개발자에게 다음과 같은 실질적인 이점을 제공합니다:

- **개발 시간 단축**: API 명세가 준비되어 있다면, 기본적인 CRUD 기능을 구현하는 데 드는 시간이 크게 줄어듭니다.
- **품질 향상**: 자동 생성된 코드는 모범 사례를 따르므로, 일관성 있고 견고한 코드베이스를 얻을 수 있습니다.
- **학습 도구**: 특히 백엔드 개발을 배우는 초보자에게는 좋은 코드를 참고하고 배울 수 있는 기회를 제공합니다.
- **빠른 프로토타이핑**: 아이디어를 빠르게 프로토타입으로 만들어 검증할 수 있습니다.

### 확장 가능성

이 기본 구현을 바탕으로 다음과 같은 확장이 가능합니다:

1. **인증 및 권한 관리**: JWT를 활용한 사용자 인증 시스템 추가
2. **관계형 데이터베이스 활용**: SQLite에서 PostgreSQL이나 MySQL 같은 프로덕션 급 데이터베이스로 전환
3. **테스트 코드 작성**: 각 API 엔드포인트에 대한 자동화 테스트 구현
4. **프론트엔드 연동**: React, Vue, Angular 등의 프레임워크로 만든 프론트엔드와 연동
5. **배포 자동화**: Docker 컨테이너화 및 CI/CD 파이프라인 구성

### 실무 적용 시 고려사항

실제 제품 환경에서 사용할 때는 다음 사항들도 고려해야 합니다:

- **보안**: 입력 검증, SQL 인젝션 방지, CORS 설정 등 보안 강화
- **성능 최적화**: 데이터베이스 인덱싱, 캐싱 등을 통한 성능 개선
- **모니터링**: 로깅 및 모니터링 시스템 연동
- **확장성**: 서비스 성장에 따른 수평적 확장을 고려한 설계

GitHub Copilot 에이전트는 코드를 자동 생성해주지만, 개발자로서 이러한 측면들을 이해하고 적절히 조정하는 것은 여전히 중요합니다. 이 도구는 개발자의 전문성을 대체하는 것이 아니라, 반복적이고 시간 소모적인 작업을 줄여주어 더 가치 있는 영역에 집중할 수 있게 해줍니다.

이 튜토리얼을 통해 GitHub Copilot 에이전트가 어떻게 개발 프로세스를 가속화하고 효율적으로 만들 수 있는지 확인할 수 있었습니다. 앞으로도 AI 코딩 어시스턴트 도구들은 더욱 발전하여 개발자의 생산성과 코드 품질을 높이는 데 큰 역할을 할 것으로 기대됩니다.

## 퍼가는 법
 
이 글은 자유롭게 퍼가셔도 좋아요! 다만 출처는 아래 링크로 꼭 남겨주세요 😊

[https://tykimos.github.io/2025/04/16/github_copilot_agent_python_sns_backend](https://tykimos.github.io/2025/04/16/github_copilot_agent_python_sns_backend) 