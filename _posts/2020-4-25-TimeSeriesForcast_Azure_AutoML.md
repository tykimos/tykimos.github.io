---
layout: post
title:  "시계열 데이터 예측 애저 오토엠엘(Azure AutoML)"
author: 김태영
date:   2020-04-25 12:00:00
categories: etc
comments: true
image: http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_title1.png
---

기상, 금융, 에너지 등등 시간에 따라 변하는 데이터를 다루고 있는 분야가 많습니다. 이를 시계열 데이터라고 부르고, 이러한 데이터를 입력하여, 분류를 하거나 이상징후를 검출하거나 미래의 수치를 예측하기도 합니다. 이번 발표에서는 시계열 예측 문제로 과거의 정보를 입력하여 미래의 요소를 예측하는 방법 중 AutoML에 대해서 말씀드리고자 합니다. AutoML은 머신러닝 모델을 인공지능이 데이터셋을 보면서 스스로 전처리 및 모델링을 수행하는 것을 말합니다. 시계열 데이터 분석 및 예측에서는 전통적인 통계 방법, 머신러닝 모델, 딥러닝 모델 등 많은 연구가 이뤄지고 있는데요, 마이크로소프트 애저 오토엠엘(Azure AutoML)에서도 시계열 예측 문제를 지원하게 되어 이를 소개드립니다.

---
### 발표영상

본 발표는 2020년 4월 25일 Global Azure Virtual 2020 온라인 행사에서 진행되었습니다.

[![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_1.png)](https://youtu.be/C-HVF9TkcLQ)

---
### 발표자료

발표 슬라이드는 아래 링크에서 확인할 수 있습니다.

[![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_title1.png)](https://docs.google.com/presentation/d/1XvdwZpkPoxjVyI3Ld7KOX9yfse2xoeoTFTEg8YmUvDE/edit?usp=sharing)

---
### 내용요약

전체 중 일부 슬라이드에 대해서 요약해봤습니다.

![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_5.png)

현재 애저 오토엠엘에서 지원하는 태스크는 분류, 회귀, 시계열 예측 문제입니다. 이번 발표에서는 시계열 예측에 대해서 다룹니다.

![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_6.png)

애저 오토엠엘은 작동원리를 그림으로 표시한 것입니다. 사용자는 데이터셋, 평가기준, 자원제약사항만 입력하면 피처 엔지니어링, 모델링, 하이퍼파라미터 튜닝 등은 오토엠엘에서 알아서 수행하고, 평가 결과를 리더보드(점수판)에 기록합니다.

![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_7.png)

시계열 예측 문제에 대한 교차 검증 방법으로 Rolling Origin 방식이 있습니다. 타입스텝이 진행됨에 따라 새로생긴 샘플로 검증을 수행합니다.

![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_8.png)

오토엠엘에서 가장 신경써야 하는 부분은 AutoMLConfig입니다. 시계열 예측 문제에서는 아래와 같은 추가 파라미터가 필요합니다.
* time_column_name : 시간스텝을 의미하는 열을 지정합니다.
* grain_column_name : 복수 개의 시계열이 존재할 경우 그룹핑할 열을 지정합니다.
* max_horizon : 최대 예측 구간을 지정합니다.

![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_9.png)

실제 예제 파이썬 코드를 살펴보면서 데이터 준비, 환경 설정, 오토엠엘 설정, 실행, 평가 결과 보는 법에 대해서 설명합니다.

![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_10.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_11.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_12.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_13.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_14.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_15.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_16.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_17.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_18.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_19.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_20.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_21.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_22.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_23.png)
![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_24.png)

---
### 둘러보기

#### Global Azure Virtual 2020

이번 행사에는 4개 트랙으로 마이크로소프트 애저 클라우드(Microsoft Azure Cloud)와 관련된 다양하고 재미있고 흥미진진한 기술 세미나가 열렸습니다.

* 트랙 A: DevOps
* 트랙 B: 보안
* 트랙 C: AI & IoT
* 트랙 D: 모던 앱스

아래 링크로 꼭 한 번 둘러보시기 바랍니다~

[![img](http://tykimos.github.io/warehouse/2020-4-25-TimeSeriesForcast_Azure_AutoML_3.png)](https://github.com/krazure/gab2020kr/blob/master/README.md?fbclid=IwAR3VHyVtqVjsKiNi91sod9yDP_PNzQWscAwcfVZVP9LauUuZAV0xTaJTA3A)

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
