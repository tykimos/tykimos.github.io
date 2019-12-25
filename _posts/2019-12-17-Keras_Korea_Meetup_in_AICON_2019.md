---
layout: post
title:  "케라스 코리아 밋업 in AICON 2019[발표자료포함]"
author: 김태영
date:   2019-12-17 13:00:00
categories: seminar
comments: true
image: http://tykimos.github.io/warehouse/2019-12-17-Keras_Korea_Meetup_in_AICON_2019_title_2.png
---

케라스 코리아 공식 밋업을 AICON에서 열리게 되었습니다. 

![img](http://tykimos.github.io/warehouse/2019-12-17-Keras_Korea_Meetup_in_AICON_2019_title_2.png)

* 일시: 12월 17일 
* 시간: 13:00~18:00
* 장소: 서울 양재R&D혁신허브 (서울 서초구 태봉로 114, 한국교원단체총연합회관)
* 규모: 본행사 1000명, 밋업 100명

### 전체 행사 프로그램
---

![img](http://tykimos.github.io/warehouse/2019-12-17-Keras_Korea_Meetup_in_AICON_2019_1.png)

많은 분들이 함께 해주셨습니다!

![img](http://tykimos.github.io/warehouse/2019-12-17-Keras_Korea_Meetup_in_AICON_2019_pic.jpg)

### 상세 내용
---

* 김태영 
    * 주제: 케라스와 함께하는 AIFactory
    * 소개: 기업이 의뢰한 인공지능 문제를 각 분야의 전문가및 머신러닝 엔지니어들의 크라우드소싱으로 해결하여 인공지능 모델 개발의 수요와 공급을 이어주고 인공지능 적용 가속화 시키는 플랫폼인 AIFactory를 소개합니다. 이 플랫폼의 주요 요소인 AutoML, 데이터셋 암호화, 인공지능 모델 자동 관리와 연동방안에 대해서도 논의합니다.
* 박남욱
    * 주제: 인공지능의 불확실성: 만약 목숨이 달린 일이라면 딥러닝에게 맡기시겠습니까?
    * 소개: While deep neural networks have better prediction accuracy than human in some areas, it is not possible to estimate the uncertainty of the predictions yet. The prediction cannot be perfectly performed and the misprediction might result in fatal consequences in some areas such as autonomous vehicles control, robotics, and medical analysis. Therefore estimating uncertainty as well as predictions will be crucial for the safer application of machine learning based systems. In this lecture, I explain how deep learning can estimate the uncertainty of its prediction.
    * 발표자료:

[![img](http://tykimos.github.io/warehouse/2019-12-17_uncertainty-in-ai.png)](http://tykimos.github.io/warehouse/2019-12-17_uncertainty-in-ai.pdf)

* 임도형
    * 주제: GAN을 사용한 이상탐지 사례
    * 소개: GAN을 사용한 이상탐지 사례를 설명한다. 일반적으로 레이블링 데이터를 확보하는 것은 어렵고, 정상이 아닌 이상상태에 대한 레이블링 데이터는 그 수도 적다. 이러한 제약을 극복하기 위하여 비지도 학습인 GAN을 사용하여 정상상태를 학습하고 이를 사용하여 비정상 상태를 탐지한다. 생체신호에 대한 시계열데이터에 대한 이상탐지 사례를 설명한다.
    * 발표자료: 
    
[![img](http://tykimos.github.io/warehouse/2019-12-17_GAN_Anomaly_Detection.png)](http://tykimos.github.io/warehouse/2019-12-17_GAN_Anomaly_Detection.pptx)

* 김영하 
    * 주제: TensorFlow 2.0와 Keras의 인연 그리고 이어질 사연들
    * 소개: (추후 업데이트)
    * 발표자료: 
    
[![img](http://tykimos.github.io/warehouse/2019-12-17_TensorFlow_2_0_NN_CV.png)](http://tykimos.github.io/warehouse/2019-12-17_TensorFlow_2_0_NN_CV.pdf)

* 김수정 
    * 주제: YOLK(You Only Look Keras)
    * 소개: Object detection은 최근 자율주행 자동차, 무인점포, 제조업체 등 실생활에서 정말 많이 사용되고 있는 알고리즘입니다. 본 발표에서는 Opensource contributon 프로젝트인 YOLK를 소개합니다. YOLK는 케라스의 창시자 프랑소와 숄레의 철학을 이어받아 Keras만 알면 누구나 쉽고 간단하게 Object detection을 할 수 있도록 만든 API입니다.
    * 발표자료: 
    
[![img](http://tykimos.github.io/warehouse/2019-12-17_AICON2019_YOLK_SoojungKim.png)](http://tykimos.github.io/warehouse/2019-12-17_AICON2019_YOLK_SoojungKim.pdf)

* 차금강 
    * 주제: 분산 텐서플로우를 이용한 분산 강화학습(IMPALA)
    * 소개: 임팔라는 분산형 강화학습 중 가장 성능이 좋다고 알려져 있으며 스타크래프트2를 정복한 알파스타에 적용된 핵심 알고리즘입니다. 본 발표에서는 이전의 분산 강화학습의 단점을 설명하고 이를 어떻게 극복했는지를 설명하며 이를 구현하는데에 느꼈던 어려웠던 점과 어떻게 구현을 했는지에 대해 설명을 합니다. 또한 구현이 정확히 되었는지를 벤치마크하며 다른 분산 강화학습과의 특성까지 비교합니다.
    * 발표자료: (추후 업데이트)

* 이태영
    * 주제: 다양한 업종의 프로세스를 확인하고 효과적인 딥러닝 모델을 적용하기
    * 소개: 딥러닝에서 무엇보다 중요한 것은 적용할 시스템에 대한 legacy 아키텍처이고, 제조업의 경우 공정 프로세스에 대한 이해가 무엇보다 중요하다. 그 이유는 효율을 높이기 위한 프로세스를 찾아낼 수 있고 생산성 향상과 직결되는 프로세스에 적용한 딥러닝 모델이야 말로 엄청난 경비 절감을 창출할 수 있기 때문이다. 이에 제가 몸 담았었던 철강업과 잘은 모르지만 반도체업의 프로세스를 비교해 보고 어디에 딥러닝 알고리즘을 적용하는게 좋을 것인지를 고찰해 본다. 또한 의료업의 경우 무엇보다 보안이 중요하기에 Federated Learning을 활용해야 하고 챗봇의 경우 Intent Scope에 따른 아키텍처 설계가 중요하고 이에 Transfer Learning이 나오게 되었다는 전반적인 오버뷰를 진행할 예정이다.
    * 발표자료: 

[![img](http://tykimos.github.io/warehouse/2019-12-17_DeepLearningThroughVariousProcesses.png)](http://tykimos.github.io/warehouse/2019-12-17_DeepLearningThroughVariousProcesses.pdf)

### 신청하기
---
현장 등록 가능합니다. 선착순 100분에게 예쁜 케라스 티 선물드립니다~

![png](http://tykimos.github.io/warehouse/2019-12-17-Keras_Korea_Meetup_in_AICON_2019_4.jpeg)


### 표지설명
---
표지에 있는 크리스마스 트리는 TF2.0의 케라스 함수형API 모델로 디자인을 한 것입니다. 자세한 사용법은 아래 링크에서 확인할 수 있습니다. 

* The Keras functional API in TensorFlow: https://www.tensorflow.org/guide/keras/functional

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 모델 구성하기
inputs = tf.keras.Input(shape=(32,), name='A') 
a = layers.Dense(64, activation='relu', name='AAAAAAAA')(inputs)
a = layers.Dense(64, activation='relu', name='AAAAAAAAAAAAAA')(a)
b = layers.Dense(64, activation='relu', name='B')(a)
b1 = layers.Dense(64, activation='relu', name='B1')(b)
b1 = layers.Dense(64, activation='relu', name='BBBBBBB1')(b1)
b2 = layers.Dense(64, activation='relu', name='B2')(b)
b2 = layers.Dense(64, activation='relu', name='BBBBBBB2')(b2)
c = layers.concatenate([b1, b2], name='C')
c1 = layers.Dense(64, activation='relu', name='CCCCCCCCCC1')(c)
c1 = layers.Dense(64, activation='relu', name='CCCCCCCCCCCCCCCCCCCCCCC1')(c1)
c2 = layers.Dense(64, activation='relu', name='CCCCCCCCCC2')(c)
c2 = layers.Dense(64, activation='relu', name='CCCCCCCCCCCCCCCCCCCCCCC2')(c2)
d = layers.concatenate([c1, c2], name='D')

predictions = layers.Dense(10, activation='softmax', name='Z')(d)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

# 모델 표출하기
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

%matplotlib inline

#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
```

![png](http://tykimos.github.io/warehouse/2019-12-17-Keras_Korea_Meetup_in_AICON_2019_2.png)

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
