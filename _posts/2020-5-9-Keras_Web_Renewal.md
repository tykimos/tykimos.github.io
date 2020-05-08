---
layout: post
title:  "케라스 웹페이지 가이드 및 코드샘플 100% 리뉴얼"
author: 김태영
date:   2020-04-25 12:00:00
categories: etc
comments: true
image: http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_title_1.png
---

케라스 웹페이지(https://keras.io/)가 100% 리뉴얼 되었습니다. 엔지니어와 연구자를 구분하여 맞춤형 가이드를 제공하고 다양하고 실용적인 예제가 구글 코랩과 함께 제공되어 클릭 한번으로 실습이 가능합니다. 케라스 튜너, 오토케라스 등 흩어져 있던 케라스 에코시스템도 공식 홈페이지에서 정리되어서 소개되고 있으니 딥러닝을 사랑하는 사람들에겐 종합선물세트 같은 기분이 들 것 같습니다. 

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_title_1.png)

---
### 대문 살펴보기

가장 처음에 "간단하고 유연하고 강력하다"라고 쓰여져 있네요. 개인적으로 상투적인 표현을 좋아하진 않지만 케라스를 3단어로 표현하라면 딱 이거다 싶네요. 대문에서는 입구 버튼이 세가지만 표시되어 있지만 하위 메뉴는 더 많이 있습니다. 각각에 대해서는 조금 있다가 살펴보도록 하죠.

* About Keras (케라스란?)
* Getting started (시작하기)
* Developer guides (개발자 가이드)
* Keras API reference (케라스 API 레퍼런스)
* Code examples (코드 예제)
* Why choose Keras? (왜 케라스를 선택하는가?)
* Community & governance (커뮤니티와 거버넌스)
* Contributing to Keras (케라스에 기여하는 법)

대문에 가장 먼저 보이는 건 케라스 코드입니다. 이 몇 줄 안되는 코드가 비디오 QA 딥러닝 모델이군요. 즉 비디오와 관련된 질문을 입력하면 답을 알려주는 모델입니다. 비디오라는 것이 이미지에다 시간 차원이 더 들어가 있고, 질문이라는 것도 텍스트를 다뤄야 하는 부분이라 꽤나 복잡한 구성인데도 케라스로 구현하면 이렇게나 간단해집니다. 비디오 QA에 대해서 좀 더 보시려면 >> [케라스와 텐서플로우와의 통합](https://tykimos.github.io/Keras/2017/02/22/Integrating_Keras_and_TensorFlow/)

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_1.png)

그리고 "사람을 위한 딥러닝"이라고 의미심장한 인트로 문장이 나오네요. 케라스의 단점이 문서화가 빈약하다는 것이었는데, 지금은 광범위한 문서와 개발자 가이드를 가지고 있다고 자부하네요. 예전에 제가 작성했던 케라스의 단점에 대해서 궁금하시다면 >> [케라스 단점](https://tykimos.github.io/2017/12/20/Keras_Drawback/)

이 밖에 대문에 적힌 문장들을 요약해봤습니다.
* 캐글의 5대 우승팀을 봤을 때 딥러닝 프레임워크 중 케라스가 가장 사용많이 한다네요. 그 이유는 새로운 실험을 쉽게 할 수 있기 때문에 더 많은 아이디어에 대한 시도를 해볼 수 있답니다.
* TensorFlow 2.0 위에서 케라스는 대규모 GPU 클러스터나 전체 TPU pod 클러스터로 확장할 수 있답니다. 이게 가능한 것도 신기하지만 게다가 쉽다는...
* 자바스크립트로 브라우저 상에서 바로 실행이 가능하고, TF Lite로 배포하여 iOS, Android 및 임베디드 장치에서 실행할 수 있습니다. 게다가 웹 API를 통해 케라스 모델을 서비스하는 것도 아주 쉽답니다.
* 케라스는 데이터 관리부터 배포까지 머신러닝 워크플로우의 각 스텝을 관장하면서 TensorFlow 2.0 생태계의 중앙 역할을 한답니다.
* 연구란 많은 실험이 필요하죠. 쉽고 빠르게 실험하기 위해서 CERN, NASA, NIH, LHC 등 많은 연구기관에서 케라스를 사용한답니다.
* 딥러닝을 배우는 가장 좋은 방법으로 많은 대학 과정에서 케라스를 선택했답니다.

---
### About Keras (케라스란?)

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_2.png)

* 엔진을 텐서플로우 2.0을 사용함과 동시에 강력한 호환을 강조하면서 GPU/TPU 사용, 대규모 병렬분산 시스템 사용, 다양한 기기에 배포 가능합니다.
* 케라스의 핵심 구조는 레이어(layers)와 모델(models)인데, 그냥 쌓아올 릴 것면 Sequential 모델을, 복잡한 구조를 사용할 거면 Functional API, 임의의 구조로 짜려면 Subclassing 방법을 지원합니다.
* 그 밖에 설치 방법 및 케라스의 의미에 대해서 언급합니다. 케라스 의미에 대해서 궁금하시다면 >> [케라스 이야기](https://tykimos.github.io/2017/01/27/Keras_Talk/)

---
### Getting started (시작하기)

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_3.png)

서브로 열러는 메뉴를 보니깐 너무나 두근 거리네요. 요약해봅니다.

* 엔지니어를 위한 케라스
  * 모델을 학습하기 전에 데이터를 준비하는 방법에 대해서 설명합니다.
  * 정규화나 자연어 처리 같은 전처리 방법에 대해서 설명합니다.
  * 학습 과정을 살펴볼 수 있는 방법과 평가하는 방법 그리고 신규 데이터로 부터 사용하는 방법을 배웁니다.
  * GAN 모델 같이 툭수하게 학습시키는 법을 배웁니다.
  * 멀티 GPU 상에서 학습시키는 법을 배웁니다.
  * 하이퍼파라미터 튜닝을 통해 모델을 개선시키는 법을 배웁니다.
* 연구자를 위한 케라스
  * 자신만의 레이어를 만드는 법을 배웁니다.
  * 로우레벨에서의 학습 제어를 하는 법을 배웁니다.
  * 손실이나 평가를 학습 과정에서 추적하는 법을 배웁니다.
  * 실행 속도를 향상 시키는 법으 배웁니다.
* 케라스 에코시스템
  * Keras Tuner 더보기 >> [하이퍼튜닝을 손쉽게 - 케라스 튜너](https://tykimos.github.io/2019/05/10/KerasTuner/)
  * AutoKeras
  * TensorFlow Cloud
  * TensorFlow.js
  * TensorFlow Lite
  * Model optimization toolkit
  * TFX integration
* 배움터 : 관련 도서, MOOCs, 웹사이트 등이 소개되어 있습니다.
* 자주 묻는 질문 : 평소 궁금했던 부분들이 사이다처럼 정리되어 있네요. 꼭 한 번 살펴보세요.
  * General questions
  * Training-related questions
  * Modeling-related questions

---
### Developer guides (개발자 가이드)

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_4.png)

레이어 서브클래싱, 파인 튜닝, 모델 저장 등등 케라스 전문가가 되기 위한 다양한 주제를 포함하고 있습니다. 각 페이지에는 구글 코랩과 연동이 되어 있어서 클릭 한 번으로 실습 환경이 제공되네요. 구글 코랩이 궁금하다면 >> [코랩 시작하기](https://tykimos.github.io/2019/01/22/colab_getting_started/)

---
### Keras API reference (케라스 API 레퍼런스)

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_5.png)

* Models API : 모델, 학습 및 저장에 관련된 API
* Layers API : 코어 및 다양한 레이어 API
* Callbacks API : 학습 과정을 모니터링하거나 제어할 수 있는 다양한 콜백함수 API
* Data preprocessing : 이미지, 시계열, 텍스트를 위한 전처리 API
  * Image data preprocessing
  * Timeseries data preprocessing
  * Text data preprocessing
* Optimizers
* Metrics : 다양한 태스크에 대한 평가 API
* Losses
* Built-in small datasets
  * 이미지 : MNIST, CIFAR10, CIFAR100, Fashion MNIST
  * 텍스트 : IMDB movie review, Reuters newswire
  * 수치 : Boston Housing price
* Keras Applications : Xception, EfficientNet 등 다양한 모델
* Utilities: 유용한 부가 기능들
  * Model plotting utilities
  * Serialization utilities
  * Python & NumPy utilities
  * Backend utilities

---
### Code examples (코드 예제)

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_6.png)

모든 예제가 400줄 이하로 간단 명료하게 되어 있다고 하네요. 게다가 구글 코랩으로 되어 있어 클릭한 번으로 실습도 가능합니다. 하루에 한 예제 어떠신지요?

* Computer Vision
  * Grad-CAM class activation visualization
  * Image classification from scratch
  * Image segmentation with a U-Net-like architecture
  * Simple MNIST convnet
  * Next-frame prediction with Conv-LSTM
* Natural language processing
  * Bidirectional LSTM on IMDB
  * Using pre-trained word embeddings
  * Character-level recurrent sequence-to-sequence model
  * Sequence to sequence learning for performing number addition
  * Text classification from scratch
* Structured Data
  * Imbalanced classification: credit card fraud detection
* Generative Deep Learning
  * Character-level text generation with LSTM
  * GAN overriding Model.train_step
  * Deep Dream
  * Variational AutoEncoder
  * Neural style transfer
* Quick Keras recipes
  * Endpoint layer pattern
  * A Quasi-SVM in Keras
  * Simple custom layer example: Antirectifier

---
### Why choose Keras? (왜 케라스를 선택하는가?)

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_6.png)

케라스를 선택해야 하는 이유에 대해서 하나 하나 설명합니다. 저는 개인적으로 이름이 제일 마음에 듭니다.

---
### Community & governance (커뮤니티와 거버넌스)

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_7.png)

케라스 운영진과 그들의 철학을 살펴볼 수 있습니다. 다양한 채널을 통해서 소통하고 있으니 케라스와 함께 성장하고 싶다면 참여하십시요~

---
### Contributing to Keras (케라스에 기여하는 법)

![img](http://tykimos.github.io/warehouse/2020-5-9-Keras_Web_Renewal_8.png)

케라스 오픈소스에 기여하는 법에 대해서 설명합니다. 기여 자체도 의미가 깊지만 기여하는 과정에서도 많은 걸 배울 수 있답니다. 여러가지 방법으로 기여할 수 있으니 한 번 살펴보시기 바랍니다.

* Bug reporting
* Requesting a Feature
* Proposing a design for a new API
* Submitting a Pull Request
* Adding new examples

---
### 마무리

이상으로 간단하게 리뉴얼된 케라스 홈페이지를 살펴봤습니다. 체계적인 문서 구조와 방대한 팁들, 모든 예제들이 구글 코랩으로 되어 있어 바로 실습해볼 수 있어 딥러닝 입문하기에 너무나 좋은 프레임워크로 거듭나고 있는 것 같습니다. 앞으로도 케라스 많이 관심가져주세요~

---
### 함께보기

#### AIFactory

* [AI팩토리 머신러닝 경연대회](http://aifactory.space)

#### 케라스 코리아 커뮤니티

* [케라스 코리아 페북](https://www.facebook.com/groups/KerasKorea/)
* [케라스 코리아 슬랙 초대](https://join.slack.com/t/keraskorea/shared_invite/enQtNTUzMTUxMzIyMzg4LWQ3YmQ1YTdmNTYxOTAwZTExNmFmOGM3M2QyMjIyNzYwYTY2YTY2ZjBlNDNlZDdmMTU0NGVjYzFkMWYxNzE0ZDA)
* [케라스 코리아 단톡방](https://open.kakao.com/o/g93MSBV)
* [케라스 코리아 블로그](http://keraskorea.github.io)
* [케라스 공식 문서 한글번역 참여방법](https://tykimos.github.io/2019/02/06/Contribution_of_Keras_Document_to_Korean_Translation/)

#### 캐글 코리아 커뮤니티

* [캐글 코리아 페북](https://www.facebook.com/groups/KaggleKoreaOpenGroup/)
* [캐글 코리아 단톡방](https://open.kakao.com/o/gP24T89)
* [캐글 코리아 블로그](https://kaggle-kr.tistory.com/)

#### MLOps KR 커뮤니티

* [엠엘옵스 코리아 페북](https://www.facebook.com/groups/MLOpsKR/)
