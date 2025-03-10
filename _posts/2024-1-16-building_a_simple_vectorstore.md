---
layout: post
title: "문서로더부터 벡터스토어까지 쌩 파이썬으로 만들어보기"
author: Taeyoung Kim
date: 2024-1-16 00:00:00
categories: bic
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-1-16-building_a_simple_vectorstore_title.png
---

## 소개

문서 등의 지식을 이용한 챗봇이 널리 사용되고 있습니다. 여기에 사용되는 기법이 Retrieval-Augmented Generation (RAG)인데요. 이를 구현하기 위해서는 문서로더부터 벡터스토어 구축까지 많은 전처리가 필요합니다. 이를 랭체인으로 쉽게 해볼 수 있는데요. 랭체인은 추상화가 잘 되어 있어서, 그 안을 바로 살펴보는 것은 쉽지 않습니다. 이번 포스팅에서는 Python과 OpenAI 라이브러리를 사용하여 간단한 벡터 스토어를 구축하는 방법을 소개합니다. 이를 통해 랭체인 기반의 RAG을 실제로 구현하고, 이해하는 데 도움이 될 것입니다.

* 세미나 일시 : 2024년 1월 18일 오후 9시 ~ 9시반
* 장소 : 유튜브 라이브 (아래 링크에 지정된 시간에 접속)
* 내용 : RAG 개념 설명 및 쌩 파이썬 소스코드 설명
* 참가신청하기 : [https://aifactory.space/task/2718/discussion/817](https://aifactory.space/task/2718/discussion/817) 접속 후 우측상단 [참가하기] 버튼 클릭

<iframe width="100%" height="400" src="https://www.youtube.com/embed/631aGBftKjo?si=ZyCFQA2l9NLbxHWg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 필수 설치와 환경설정
* 먼저, 필요한 라이브러리를 설치해야 합니다. 이 예제에서는 openai와 tqdm 라이브러리를 사용합니다.
* OenAI API를 사용하기 위해서는 API 키를 환경 변수로 설정해야 합니다. (참고로 여기서 제공된 키는 예시입니다.)

## 텍스트 로딩과 처리
* 텍스트 로더: 지정된 파일 경로에서 텍스트를 로드하는 클래스입니다. 파일을 열고 내용을 읽어서 반환합니다.
* 텍스트 스플리터: 주어진 텍스트를 지정된 크기의 청크로 나누는 클래스입니다. 구분자 패턴을 기준으로 문서를 분할하고, 지정된 크기를 넘지 않도록 청크를 생성합니다.

## 벡터 스토어 구축
* 임베딩 클래스: OpenAI의 임베딩 모델을 사용하여 텍스트를 벡터로 변환하는 클래스입니다. text-embedding-ada-002 모델을 사용하여 임베딩을 생성합니다.
* 벡터 스토어: 문서와 그에 해당하는 벡터를 저장하는 클래스입니다. 각 문서를 임베딩하고, 유사도 검색을 위해 코사인 유사도를 계산합니다.

## 사용 예제
* 국가 연설문 예제: 미국 대통령의 연설문을 로드하고, 벡터 스토어를 구축하여 특정 쿼리에 대한 유사 문서를 검색합니다.
* 한국 헌법 예제: 한국 헌법의 전문을 로드하고, 작은 청크로 나누어 벡터 스토어를 구축합니다. 이를 통해 특정 질문에 대한 관련 문서를 찾습니다.

## 대화형 QA 시스템 구현
* QA 체인: 검색된 문서를 바탕으로 GPT-3.5 모델을 활용하여 질문에 대한 답변을 생성합니다. 사용자 입력에 따라 AI가 적절한 응답을 제공하는 대화 시스템으로 확장할 수 있습니다.

이 예제를 통해, 간단한 벡터 스토어의 구축부터 RAG의 개념적 이해, 그리고 실제 적용까지 전체 과정을 경험할 수 있습니다. RAG가 어떻게 동작하는지 깊이 이해하는 데 큰 도움이 될 것입니다. 복잡한 개념도 직접 구현해보면서 쉽게 이해할 수 있음을 보여주고자 합니다. 
