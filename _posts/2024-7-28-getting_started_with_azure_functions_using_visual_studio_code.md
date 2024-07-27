---
layout: post
title: "Azure Functions 시작하기 with Visual Studio Code"
author: 김태영
date: 2024-7-28 01:00:00
categories: Azure Functions Serverless
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-7-28-getting_started_with_azure_functions_using_visual_studio_code_title.jpg
---

Visual Studio Code(VS Code)를 사용하여 Azure Functions를 쉽고 빠르게 시작하는 방법을 단계별로 설명하겠습니다. Azure Functions는 서버리스 컴퓨팅 서비스로, 개발자가 인프라 관리에 대한 걱정 없이 코드에 집중할 수 있게 해줍니다. 이번 실습은 아래 마이크로소프트의 문서를 참고하였습니다. 

- [Azure Functions for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azurefunctions)

이 가이드에서는 다음 주요 단계를 다룹니다:

1. VS Code에 Azure Functions 확장 설치
2. Azure 계정 연결
3. 기본 Function 프로젝트 생성
4. 로컬에서 Function 실행 및 테스트
5. Azure에 Function 배포

이 과정을 통해 개발자는 로컬 환경에서 Function을 개발하고 테스트한 후, 쉽게 클라우드로 배포할 수 있습니다. 파이썬을 사용하여 예제를 구성했지만, 다른 지원 언어로도 유사한 과정을 따를 수 있습니다.

## Azure Functions 설치 및 기본 프로젝트 생성

먼저 Visual Studio Code의 마켓플레이스에서 Azure Functions을 검색하여 설치합니다. 

![](https://cdn.aifactory.space/images/20240726222934_EolW.png)

Visual Studio Code 왼쪽 사이드바에서 Azure 버튼을 클릭한 후 [Sign in to Azure]을 클릭합니다. 

![](https://cdn.aifactory.space/images/20240726223155_CoDl.png)

로그인 절차를 마친 후 정상적으로 로그인을 수행했다면, 아래처럼 [RESOURCES]에 자신의 구독을 확인할 수 있습니다.

![](https://cdn.aifactory.space/images/20240726223257_jFtA.png)

아래 [WORKSPACE]에서 [Create Function Projects…]을 클릭하여 첫번째 프로젝트를 생성합니다.

![](https://cdn.aifactory.space/images/20240726224342_fXGT.png)

프로젝트의 개발언어는 파이썬으로 선택하겠습니다.

![](https://cdn.aifactory.space/images/20240726224405_shUB.png)

파이썬 모델도 추천하는 Model V2를 선택했습니다. 

![](https://cdn.aifactory.space/images/20240726224427_mOIT.png)

파이썬 버전은 python3.11으로 선택하였습니다.

![](https://cdn.aifactory.space/images/20240726224500_iaxF.png)

프로젝트템플릿으론 HTTP trigger로 선택했습니다.

![](https://cdn.aifactory.space/images/20240726224519_JHlB.png)

프로젝트에서 만들 함수의 이름을 [hello_world]로 지정했습니다.

![](https://cdn.aifactory.space/images/20240726224545_MKUV.png)

권한 수준은 [ANONYMOUS]로 지정했습니다. 

![](https://cdn.aifactory.space/images/20240726224629_SMSI.png)

지금 창이 새로 연 창이라 현재 윈도우에서 열도록 하겠습니다.

![](https://cdn.aifactory.space/images/20240726224654_vNkR.png)

기본 템플릿에 따라 파이썬 기반으로 웹 서비스 함수 프로젝트를 만들어줍니다. 

![](https://cdn.aifactory.space/images/20240727211943_YMUc.png)

## Azure Functions 로컬 실행

로컬에서 실행해보려면 "azure-functions-core-tools"을 설치해야 합니다. 맥에서 설치하기 위해서 먼저 아래 명령을 수행합니다. 

- brew tap azure/functions

![](https://cdn.aifactory.space/images/20240727211923_BUwr.png)

성공적으로 설치하면 다음 명령을 수행합니다. 

- brew install azure-functions-core-tools@4

![](https://cdn.aifactory.space/images/20240727212020_bHPk.png)

성공적으로 설치하면 다음 명령을 수행합니다. 

- brew link --overwrite azure-functions-core-tools@4

![](https://cdn.aifactory.space/images/20240727212059_pLhY.png)

로컬에서 실행하기 위한 모든 준비가 완료되었습니다. 왼쪽 사이드바에서 [실행 및 디버그] 메뉴를 선택한 후 [플레이] 버튼을 클릭합니다.

![](https://cdn.aifactory.space/images/20240727212147_IjLL.png)

정상적으로 실행되면 접근을 할 수 있는 로컬호스트 주소가 보여집니다.

![](https://cdn.aifactory.space/images/20240727212243_ZqLt.png)

해당 주소로 접속하면 아래와 같이 메시지가 띄워집니다.

![](https://cdn.aifactory.space/images/20240727212330_WwyU.png)

워크스페이스 메뉴에서 [Local Project] > [Functions] > [hello_world] 항목을 우클릭하여 [Copy Function Url]를 클릭합니다.

![](https://cdn.aifactory.space/images/20240727212516_Ellq.png)

기본 Url이 복사되며, 여기에 name 인자를 넣으면, 그에 맞게 출력됩니다.

- http://localhost:7071/api/hello_world?name=world

![](https://cdn.aifactory.space/images/20240727212652_NfQX.png)

우측 상단에 있는 [unplug] 버튼을 클릭하면 연결이 끊어집니다.

![](https://cdn.aifactory.space/images/20240727212823_JGRI.png)

## Azure Functions를 Azure에 배포

그럼 만든 Function을 Azure에 배포를 해보도록 하겠습니다. [RESOURCES]에서 [+]버튼을 클릭합니다.

![](https://cdn.aifactory.space/images/20240728000128_UJmO.png)

[Create Function App in Azure…]을 클릭합니다.

![](https://cdn.aifactory.space/images/20240728000200_fIlz.png)

이름을 입력합니다. 여기서는 [assi-works-functions]으로 정하겠습니다. 

![](https://cdn.aifactory.space/images/20240728000308_kzcZ.png)

런타임 스택을 설정합니다.

![](https://cdn.aifactory.space/images/20240728000319_LlrY.png)

리소스 위치를 지정합니다. [Korea Central]로 지정했습니다.

![](https://cdn.aifactory.space/images/20240728000339_dlnM.png)

그럼 Azure 출력 창에 진행 상태가 보여집니다.

![](https://cdn.aifactory.space/images/20240728000606_BzDQ.png)

정상적으로 배포가 되면, 클릭해서 볼 수 있는 링크가 제공됩니다.

![](https://cdn.aifactory.space/images/20240728000628_aqcY.png)

클릭하면 아래와 같이 Function App이 동작되고 있음을 할 수 있는 페이지가 보여집니다.

![](https://cdn.aifactory.space/images/20240728000656_LbzJ.png)

그럼 여기에 현재 프로젝트를 배포합니다. [Function App] > [assi-works-functions]에 우클릭해서 [Deploy to Function App…]을 클릭합니다.

![](https://cdn.aifactory.space/images/20240728001239_KEzs.png)

덮어쓴다는 경고 문구를 확인한 후 배포를 수행합니다.

![](https://cdn.aifactory.space/images/20240728001251_ZGdz.png)

그럼 Azure 출력창에 배포 진행 상황이 보여집니다.

![](https://cdn.aifactory.space/images/20240728001308_sFnd.png)

배포가 모두 수행되었다면, [Function App] > [Functions] > [hello_world] 항목이 보입니다. 이 항목에서 우클릭한 후 [Copy Function Url]를 클릭합니다.

![](https://cdn.aifactory.space/images/20240728002020_Ccix.png)

이를 웹 주소에 붙여넣고 name을 원하는 이름으로 변경하여 클릭하면 아래와 같이 배포된 Azure에서 그 결과를 확인할 수 있습니다. 

![](https://cdn.aifactory.space/images/20240728010858_GCOJ.png)

## 마무리

Visual Studio Code와 Azure Functions를 활용하여 서버리스 애플리케이션을 개발하고 배포하는 전체 과정을 살펴보았습니다. VS Code에 필요한 확장을 설치하는 것부터 시작하여, 기본적인 HTTP 트리거 Function을 생성하고, 이를 로컬에서 테스트한 후 최종적으로 Azure 클라우드에 배포하는 단계까지 수행했습니다.

이를 기반으로 다양한 트리거와 바인딩을 활용하여 마이크로서비스 아키텍처, 데이터 처리 파이프라인, 실시간 파일 처리 등 다양한 시나리오에 Azure Functions를 적용해 볼 수 있습니다. 앞으로 Azure Functions의 더 고급 기능들을 살펴보면서 효율적이고 확장 가능한 서버리스 솔루션을 구축해보시길 바랍니다.