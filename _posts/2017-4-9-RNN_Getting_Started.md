---
layout: post
title:  "순환 신경망 레이어 이야기"
author: 김태영
date:   2017-04-09 04:00:00
categories: Lecture
comments: true
image: http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_LSTM3.png
---
순환 신경망 모델은 순차적인 자료에서 규칙적인 패턴을 인식하거나 그 의미를 추론할 수 있습니다. 순차적이라는 특성 때문에 간단한 레이어로도 다양한 형태의 모델을 구성할 수 있습니다. 케라스에서 제공하는 순환 신경망 레이어는 SimpleRNN, GRU, LSTM이 있으나 주로 사용하는 LSTM에 대해서 알아보겠습니다. 

---

### 긴 시퀀스를 기억할 수 있는 LSTM (Long Short-Term Memory units)  레이어

LSTM 레이어는 아래와 같이 간단히 사용할 수 있습니다.

#### 입력 형태

    LSTM(3, input_dim=1)

기본 인자는 다음과 같습니다.
* 첫번째 인자 : 메모리 셀의 개수입니다.
* input_dim : 입력 속성 수 입니다.

이는 앞서 살펴본 Dense 레이어 형태와 비슷합니다. 첫번째 인자인 메모리 셀의 개수는 기억용량 정도와 출력 형태를 결정짓습니다. Dense 레이어에서의 출력 뉴런 수와 비슷하다고 보시면 됩니다. input_dim에는 Dense 레이어와 같이 일반적으로 속성의 개수가 들어갑니다. 

    Dense(3, input_dim=1)

LSTM의 한 가지 인자에 대해 더 알아보겠습니다.

    LSTM(3, input_dim=1, input_length=4)

* input_length : 시퀀스 데이터의 입력 길이

Dense와 LSTM을 블록으로 도식화 하면 다음과 같습니다. 왼쪽이 Dense이고, 중앙이 input_length가 1인 LSTM이고 오른쪽이 input_length가 4인 LSTM 입니다. 사실 LSTM의 내부구조는 복잡하지만 간소화하여 외형만 표시한 것입니다. Dense 레이어와 비교한다면 히든 뉴런들이 밖으로 도출되어 있음을 보실 수 있습니다. 그리고 오른쪽 블록인 경우 input_length가 길다고 해서 각 입력마다 다른 가중치를 사용하는 것이 아니라 중앙에 있는 블록을 입력 길이 만큼 연결한 것이기 때문에 모두 동일한 가중치를 공유합니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_LSTM1.png)

#### 출력 형태

* return_sequences : 시퀀스 출력 여부

LSTM 레이어는 return_sequences 인자에 따라 마지막 시퀀스에서 한 번만 출력할 수 있고 각 시퀀스에서 출력을 할 수 있습니다. many to many 문제를 풀거나 LSTM 레이어를 여러개로 쌓아올릴 때는 return_sequence=True 옵션을 사용합니다. 자세한 것은 뒤에서 살펴보겠습니다. 아래 그림에서 왼쪽은 return_sequences=False일 때, 오른쪽은 return_sequence=True일 때의 형상입니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_LSTM2.png)

#### 상태유지(stateful) 모드

* stateful : 상태 유지 여부

학습 샘플의 가장 마지막 상태가 다음 샘플 학습 시에 입력으로 전달 여부를 지정하는 것입니다. 하나의 샘플은 4개의 시퀀스 입력이 있고, 총 3개의 샘플이 있을 때, 아래 그림에서 위의 블록들은 stateful=False일 때의 형상이고, 아래 블록들은 stateful=True일 때의 형상입니다. 도출된 현재 상태의 가중치가 다음 샘플 학습 시의 초기 상태로 입력됨을 알 수 있습니다.

![img](http://tykimos.github.com/Keras/warehouse/2017-4-9-RNN_Layer_Talk_LSTM3.png)

---

### 요약

순환 신경망 레이어 중 LSTM 레이어에 대해서 알아봤습니다. 사용법은 Dense 레이어와 비슷하지만 시퀀스 출력 여부와 상태유지 모드 설정으로 다양한 형태의 신경망을 구성할 수 있습니다.

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/Keras/lecture/)
* 이전 : [컨볼루션 신경망 모델을 위한 데이터 부풀리기](https://tykimos.github.io/Keras/2017/06/10/CNN_Data_Augmentation/)
* 다음 : [순환 신경망 모델 만들어보기](https://tykimos.github.io/Keras/2017/04/09/RNN_Layer_Talk/)
