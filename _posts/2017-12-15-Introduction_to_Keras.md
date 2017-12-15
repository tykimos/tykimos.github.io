---
layout: post
title:  "케라스 소개"
author: François Chollet
translator : 김태영
date:   2017-12-15 01:00:00
categories: Lecture
comments: true
image: http://tykimos.github.io/warehouse/2017-12-15-Introduction_to_Keras_1.jpg
---
케라스에 대한 소개는 프랑소와 쏠레님이 집필하신 "Deep Learning with Python"에 잘 나와 있습니다. 웹에 3장까지 공개되어 있는데요, 그 중 케라스 소개 부분을 번역해봤습니다. (참고: https://www.manning.com/books/deep-learning-with-python)

3.2. 케라스 소개

본 책의 예제코드는 모두 케라스(https://keras.io)로 되어 있습니다. 케라스는 쉽게 거의 모든 종류의 딥러닝 모델을 정의하고 학습시킬 수 있는 파이썬 기반의 딥러닝 프레임워크입니다. 케라스는 원래 연구자들이 빨리 실험을 할 목적으로 개발되었습니다.

케라스의 주요 기능은 다음과 같습니다.

* CPU든 GPU든 동일한 코드가 적용됩니다.
* 빨리 딥러닝 프로토타입을 만들 수 있도록 쉬운 API를 제공합니다.
* 컴퓨터 비전을 위한 컨볼루션망과 시퀀스 처리를 위한 순환신경망 그리고 이 둘의 조합을 기본적으로 제공합니다.
* 다중 입력 또는 다중 출력 모델, 레이어 공유, 모델 공유 등 자유로운 네크워크 구조를 지원합니다. 이는 적대적 생성 모델(GAN, generative adversarial network)부터 뉴럴 튜닝 머신(neural Turing machine)까지 어떤 모델이든 구성하는 데 적합하다는 것을 의미합니다. 
* 케라스는 상용 프로젝트에서 자유로이 사용할 수 있는 MIT 라이선스 정책을 따르고 있습니다. 그리고 케라스는 파이썬 2.7부터 3.6까지의 버전과 호환됩니다.

케라스는 스타트업 및 대기업의 학술 연구자 및 엔지니어부터 대학원생 및 취미로 하시는 분까지 20만명이 사용하고 있습니다. 케라스는 구글, Netflix, Uber, CERN, Yelp, Square 및 수백의 스타트업에서 다양한 문제를 해결하기 위해 사용됩니다. 케라스는 또한 기계학습 대회 웹사이트인 케글(Kaggle)에서 널리 사용되는 프레임워크입니다. 최근 딥러닝 대회 우승 모델은 대부분 케라스를 사용하였습니다.

그림 3.2. 시간에 따른 딥러닝 프레임워크 관심도 (구글 검색)

http://tykimos.github.io/warehouse/2017-12-15-Introduction_to_Keras_1.jpg
