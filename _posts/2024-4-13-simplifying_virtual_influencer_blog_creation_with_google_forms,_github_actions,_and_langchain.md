---
layout: post
title: "구글폼, 깃허브 액션 그리고 랭체인으로 버추얼 인플루언서 블로그 작성 간소화"
author: Taeyoung Kim
date: 2024-4-13 15:07:05
categories: llm langchain googleform googlesheet
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-4-13-simplifying_virtual_influencer_blog_creation_with_google_forms,_github_actions,_and_langchain_title.jpg
---

![img](http://tykimos.github.io/warehouse/2024/2024-4-13-simplifying_virtual_influencer_blog_creation_with_google_forms,_github_actions,_and_langchain_title.jpg)

이 어시 체인의 목적은 구글 폼을 통해 입력받은 데이터를 바탕으로 자동으로 블로그 내용을 생성하고, 생성된 블로그 내용을 깃허브 페이지에 자동으로 업로드하는 것입니다. 이를 통해 블로그 작성과 배포 과정을 보다 효율적으로 관리할 수 있게 됩니다.

### 사용 기술 및 도구 소개

구글폼, 구글시트, 랭체인, GPT-4, 깃허브 페이지 등의 도구와 기술을 활용합니다. 구글폼은 사용자로부터 블로그의 제목과 기본 내용을 입력받는 역할을 하며, 구글시트는 이렇게 입력받은 내용을 관리하는 역할을 합니다. 랭체인(with GPT-4)는 입력받은 기본 내용을 바탕으로 상세한 블로그 내용을 생성하는 역할을 하며, 깃허브 페이지는 생성된 블로그 내용을 웹에 게시하는 역할을 합니다.

### 전체적인 워크플로우 설명

1. 사용자가 구글폼을 통해 블로그의 제목과 기본 내용을 입력합니다.
2. 입력한 내용은 구글시트에 자동으로 저장됩니다.
3. 랭체인(with GPT-4)는 구글시트에 저장된 내용을 바탕으로 상세한 블로그 내용을 생성합니다.
4. 생성된 블로그 내용은 깃허브 페이지에 자동으로 업로드되어 웹에 게시됩니다.

### 세부 내용

블로그 내용을 손쉽게 작성하기 위한 구글폼을 생성합니다. 간단한 스토리를 입력할 sketch 내용과 이미지 파일 그리고 dalle prompt을 입력할 수 있도록 구성하였습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-13-simplifying_virtual_influencer_blog_creation_with_google_forms,_github_actions,_and_langchain_1.png)

사용자가 구글폼에 내용을 입력하면 자동으로 시트가 업데이트 됩니다. 왼쪽 녹색 박스가 사용자가 입력한 부분이고, 오른쪽 노란색 박스가 어시 체인에 의해 관리되는 처리 상태 정보입니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-13-simplifying_virtual_influencer_blog_creation_with_google_forms,_github_actions,_and_langchain_2.png)

구글폼에 의해 새로운 내용이 추가되면 아래 구글 앱 스크립트가 자동으로 동작됩니다. 이 스트립트의 주요 내용은 깃허브 액션의 워크플로우를 실행 요청을 하는 것입니다. 깃허브에서 제공하는 API를 사용합니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-13-simplifying_virtual_influencer_blog_creation_with_google_forms,_github_actions,_and_langchain_6.png)

그럼 미리 정의한 워크플로우에 의해서 파이썬 코드가 실행됩니다. 깃허브 액션 탭에서 실시간으로 과정을 확인해볼 수 있습니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-13-simplifying_virtual_influencer_blog_creation_with_google_forms,_github_actions,_and_langchain_3.png)

실행되는 파이썬 코드의 핵심인 랭체인은 아래와 같이 구성되어 있습니다. 이 체인은 사용자가 입력한 sketch로부터 미리 설정한 프롬프트에 의해 "타이리"의 페르소나에 맞게 내용이 작성됩니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-13-simplifying_virtual_influencer_blog_creation_with_google_forms,_github_actions,_and_langchain_4.png)

blog-writing 액션 실행이 완료되면, 마크다운 파일이 깃허브에 업로드가 되고, 이로 인해 pages-build-development 액션이 자동으로 연달아 실행됩니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-13-simplifying_virtual_influencer_blog_creation_with_google_forms,_github_actions,_and_langchain_5.png)

이렇게 생성된 블로그는 아래와 같습니다. 

- 구경하기 : ![https://tyritarot.github.io/](https://tyritarot.github.io/)

![img](http://tykimos.github.io/warehouse/2024/2024-4-13-simplifying_virtual_influencer_blog_creation_with_google_forms,_github_actions,_and_langchain_7.png)

### 마무리

이 어시체인을 통해 사용자는 구글폼을 통해 간단히 블로그의 제목과 기본 내용을 입력하기만 하면, 나머지 블로그 작성 및 업로드 과정이 자동으로 수행되므로 블로그 관리의 효율성을 크게 향상시킬 수 있습니다. 또한 이 시스템은 블로그뿐만 아니라 다양한 웹 콘텐츠의 생성과 배포에도 활용할 수 있습니다.
