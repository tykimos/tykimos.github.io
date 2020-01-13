---
layout: post
title:  "이제 인공지능은 AIFactory에 맡기세요"
author: 김태영
date:   2020-1-15 13:00:00
categories: python
comments: true
image: http://tykimos.github.io/warehouse/2020-1-15-AIFactory_title_2.png
---

"이제 인공지능은 AIFactory에게 맡기세요"라는 주제로 이야기를 하고자 합니다. 본 발표는 AI 프렌즈의 멤버쉽데이에서 이루어지며 ETRI의 손영성님이 이어서 "사물지능, 미완의 꿈"이란 주제로 발표가 진행됩니다.

![img](http://tykimos.github.io/warehouse/2020-1-15-AIFactory_poster.jpg)

* 일시: 2020년 1월 15일 PM7~
* 장소: 대전 대덕테크비즈센터 1층

--- 

### 발표소개

인공지능 교육부터 실제 운영 시스템 구축까지 해보면서 입문부터 활용까지의 갭이 상당히 크다는 것을 매번 할 때마다 느낍니다. 이러한 갭을 채우기 위해서는 사실 많은 기술요소들이 필요합니다. 인공지능 기술 열풍으로 인해 인공지능 개발을 위해서는 적절한 문제정의와 평가방법, 그리고 데이터셋이 필요하다는 것을 대부분 알고 있습니다. 모델 개발도 AutoML이나 하이퍼파라미터튜너 기술 발전으로 최적 모델 개발에 더욱 더 쉽고 빠르게 다가갈 수 있습니다. 캐글 같은 머신러닝 경연대회 플랫폼도 크라우드 소싱 기반의 협업 툴로서 자리매김을 하고 있습니다. 하지만 모델 개발에 성공하였다 하더라도 실제 인공지능 모델을 운영하기 위해서는 여러가지 고려해야할 것들이 있습니다. 데이터셋 변동, 평가척도 변경에 따른 태스크 관리과 모델 모니터링 및 최적화가 필요하고 또한 모델을 서비스하기 위해서는 별도의 클라우드 시스템이 필요할 지도 모릅니다. 이러한 것들을 AIFactory이란 이름으로 하나로 조립하여 인공지능 적용을 가속화시키는 플랫폼을 제안합니다. 

![img](http://tykimos.github.io/warehouse/2020-1-15-AIFactory_title_0.png)

--- 

### 상세설명

#### 기본구성

* Task(태스크)
    * 인공지능을 통해서 해결하고자 하는 문제를 정의하는 것입니다. 문제가 지도학습인지 비지도학습인지 또한 강화학습인지 등 방법론에 대해서 정하고, 분류 문제인지 예측 문제인 지 등 문제 유형을 정하거나 입출력 데이터 형태에 대해서 정의합니다.
    * 문제를 정의할 때 함께 정의해야하는 것이 평가방법입니다. 메트릭(metric)이라고도 불리는 데요, 개발한 인공지능 모델을 객관적으로 평가할 척도를 정하는 겁니다. 대게 앞서 정의한 문제 정의에 따라 평가 척도가 정해지기도 하지만 사용처에 따라 사용자 정의 매트릭을 사용하기도 합니다.
* Dataset(데이터셋)
    * "데이터셋"은 "데이터"와 구분되는 용어인데요. 이 "셋" 구성을 보면, 훈련셋, 검증셋, 시험셋으로 나눠집니다. 이 셋은 샘플들의 집합인데요, 하나의 샘플은 X와 Y로 구성됩니다. X는 모델에 입력하는 데이터 즉 문제이며, Y는 모델을 통해 얻기를 희망하는 값 즉 정답을 의미합니다.
    * 참고: [데이터셋 이야기](https://tykimos.github.io/2017/03/25/Dataset_and_Fit_Talk/)
* Model(모델)
    * 데이터셋을 통해 학습하는 대상인 인공지능 모델을 의미합니다. 모델의 범주가 학습에 대해서만 국한되는 것은 아닙니다. 훈련셋없이 알고리즘을 통해서만 모델을 개발할 수 있습니다. 하지만 전통적인 방법으로 알고리즘을 개발한다하더라도 검증셋과 시험셋으로 평가를 해볼필요는 있습니다. 
    * 머신러닝 모델이나 딥러닝 모델을 데이터 기반의 모델이기 때문에 훈련셋이 필요합니다. 
* Operation(운영)
    * 태스크, 데이터셋, 모델 이렇게 3가지 요소는 별개로 구성할 수 있는 것이 아니라 서로 연관되어 있으며, 하나가 변경되었을 때 다른 것에 영향을 미칩니다. 
    * 따라서 3가지 요소가 따로 놀지않고 유기적으로 지속적으로 운영될 수 있도록 해줘야 합니다.

#### 시스템구성

* Management(관리)
    * 하나의 데이터셋에 여러 태스크가 만들어질 수 있으며, 반대로 하나의 태스크에 여러 버전의 데이터셋이 존재할 수 있습니다. 각 버전마다 모델을 개발하다보면 관리하기가 용이치 않습니다. 언제 구동으로 하더라도 재현성이 보장되는 모델을 관리하려면 별도의 버전관리가 필요합니다. 
* Big Data(빅데이터)
    * 인공지능 학습에 필요한 데이터셋을 생성하기 위해서는 선행적으로 수집되는 대량의 데이터를 수집할 수 있는 기술이 필요합니다. 
    * 개발 가능한 인공지능 모델 범위가 현재 시점에서 판단하기 힘들기때문에 가능한 대부분의 데이터를 수집하는 경우가 대부분인데, 이를 효율적으로 그리고 확장가능한 시스템이 필요합니다.
* Learning(러닝)
    * 설명
* Service(서비스)
    * 설명

#### 비즈니스 및 솔루션 연계

* Consulting(컨설팅)
    * 설명
* Annotation(어노테이션)
    * 설명
* AutoML
    * 설명
* Tuner
    * 설명
* KubeFlow
    * 설명
* Azure(애저)
    * 설명

---

### 발표자료

* [보기](https://docs.google.com/presentation/d/1pPNc1Inc9gNVcQ6YbT6038HWtvCuPsIATCzS3AqbE3s/edit?usp=sharing)

---

### 신청하기

* [AI프렌즈 멤버쉽데이 알아보기](https://aifrenz.github.io/)
* [신청하기](https://docs.google.com/forms/d/1gyond3JDvzvNcGFhXKEEtcVn6dUWB6NBDa-FDy8wlXc/edit)

---

### 둘러보기

인공지능 및 머신러닝 관련된 커뮤니티입니다. 편하게 놀러오셔요~

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
