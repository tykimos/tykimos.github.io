---
layout: post
title:  "하이퍼튜닝을 손쉽게 - 케라스 튜너"
author: 김태영
date:   2019-05-10 00:00:00
categories: etc
comments: true
image: http://tykimos.github.io/warehouse/2019-5-10-KerasTuner_title.png
---
딥러닝에 입문하여 어느정도 모델을 구성할 수 있다면, 그 다음 고민은 어떻게 이 모델을 튜닝해서 성능을 높일까?입니다. 이 과정을 하이퍼파라미터 튜닝이라고 하는데, 모델을 구성하는 여러 요소 중에 최적의 요소 값을 찾아내는 과정을 말합니다. 케라스 모델을 쉽게 튜닝하는 프레임워크를 구글에서 개발했다고 하니 살펴보도록 하겠습니다.

![img](http://tykimos.github.io/warehouse/2019-5-10-KerasTuner_title.png)

* 소개 페이지 - [https://elie.net/talk/](https://elie.net/talk/cutting-edge-tensorflow-keras-tuner-hypertuning-for-humans/)
* 발표자료 - [PDF](https://elie.net/static/files/cutting-edge-tensorflow-keras-tuner-hypertuning-for-humans/cutting-edge-tensorflow-keras-tuner-hypertuning-for-humans-slides.pdf) [구글슬라이드](https://docs.google.com/presentation/d/e/2PACX-1vT7Tc0KiUW7HX36Dck4YxKc1M8bX701PsVFUDiVK7ZN220efFGmukg0N1UJowBVnOR6Awsx_SFL9cxd/embed?slide=id.g5746e5591c_0_92)

Google I/O'19에서 발표된 세미나 동영상도 있네요. 
* [Cutting Edge TensorFlow: New Techniques (Google I/O'19)](https://www.youtube.com/watch?v=Un0JDL3i5Hg)

---
### 하이퍼파라미터 살펴보기

먼저 간단한 컨볼루션 신경망 모델을 살펴보겠습니다. 숫자 손글씨(MNIST)를 분류하는 케라스의 시퀀스 모델로 구성된 간단한 신경망입니다.  

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,
28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001))
model.summary()
```

레이어의 설정 및 레이어의 층 구조에 따라서 모델의 성능이 달라지는 데, 변경해볼 수 있는 요소를 하이퍼파라미터라고 하고, 이를 하이퍼파라미터 튜닝이라고 합니다. 그럼 먼저 하이퍼파라미터를 살펴보겠습니다. 

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,
28, 1)))

컨볼루션 레이어인 Conv2D에서는 필터의 수 "32"와 필터의 사이즈 "(3,3)"가 성능에 영향을 미칩니다.

    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))

Dense 레이어의 출력 뉴런 수인 "20"과 Dropout 레이어의 드랍비율인 "0.2"가 하이퍼파라미터가 되겠네요. 그리고 이 조합의 은닉층을 여러 번 쌓을 수도 있기에 몇 번 반복하느냐도 모델 구성에도 중요합니다. 

    model.add(Dense(10, activation='softmax'))

참고로 마지막 출력층은 하이퍼파라미터가 없습니다. 분류의 수인 "10"과 출력층의 활성화함수(activation function)은 풀고자하는 문제에 따라 정의되는 것입니다.

    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001))

마지막으로 구성한 네트워크에 학습목표(objective function)과 최적화기(optimizer)를 달아주는데, 최적화기에도 여러 설정을 할 수 있습니다. 여기서는 Adam을 사용했고, 학습율(learning rate)인 "0.001"이 학습 성능에 영향을 미칩니다. 

주요 하이퍼파라미터에게 별칭을 붙혀보겠습니다. 

    * 첫번째 Conv2D의 필터 수 : L1_NUM_FILTERS        
    * 두번째 Conv2D의 필터 수 : L2_NUM_FILTERS
    * Dense - Dropout 은닉층의 반복횟수 : NUM_LAYERS
    * Dense 레이어의 출력뉴런 수 : NUM_DIMS
    * Dropout 레이어의 드랍율 : DROPOUT_RATE
    * 최적화기의 학습률 : LR
---
### 하이퍼파라미터 별칭 사용하기

하이퍼파라미터들을 우리가 붙힌 별칭으로 교체하겠습니다.

```python
model = Sequential()
model.add(Conv2D(L1_NUM_FILTERS, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(L2_NUM_FILTERS, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
for _ in range(NUM_LAYERS):
 model.add(Dense(NUM_DIMS, activation='relu'))
 model.add(Dropout(DROPOUT_RATE))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(LR))
model.summary()
```

---
### 하이퍼파라미터 탐색 범위 지정하기

각 하이퍼파라미터마다 탐색 범위를 지정합니다. 

```python
LR = Choice('learning_rate', [0.001, 0.0005, 0.0001], group='optimizer')
DROPOUT_RATE = Linear('dropout_rate', 0.0, 0.5, 5, group='dense')
NUM_DIMS = Range('num_dims', 8, 32, 8, group='dense')
NUM_LAYERS = Range('num_layers', 1, 3, group='dense')
L2_NUM_FILTERS = Range('l2_num_filters', 8, 64, 8, group='cnn')
L1_NUM_FILTERS = Range('l1_num_filters', 8, 64, 8, group='cnn')
```

---
### 하이퍼파라미터 튜닝하기

먼저 하이퍼파라미터 탐색 범위를 지정한 코드와 별칭을 사용한 모델을 "model_fn()"이란 함수로 만듭니다.

```python
def model_fn():

    LR = Choice('learning_rate', [0.001, 0.0005, 0.0001], group='optimizer')
    DROPOUT_RATE = Linear('dropout_rate', 0.0, 0.5, 5, group='dense')
    NUM_DIMS = Range('num_dims', 8, 32, 8, group='dense')
    NUM_LAYERS = Range('num_layers', 1, 3, group='dense')
    L2_NUM_FILTERS = Range('l2_num_filters', 8, 64, 8, group='cnn')
    L1_NUM_FILTERS = Range('l1_num_filters', 8, 64, 8, group='cnn')

    model = Sequential()
    model.add(Conv2D(L1_NUM_FILTERS, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(L2_NUM_FILTERS, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    for _ in range(NUM_LAYERS):
        model.add(Dense(NUM_DIMS, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(LR))

    return model
```

실제 튜닝은 Tuner를 사용합니다. Tuner를 생성할 때, 위에서 정의한 "model_fn()"을 사용합니다.

```python
tuner = Tuner(model_fn, 'val_accuracy' epoch_budget=500, max_epochs=5)
tuner.search(train_data,validation_data=validation_data)
```

---
### 신청하기

이 멋진 기능을 체험하고 싶다면 아래 양식을 작성하여 구글팀에 보내세요~

* [keras-tuner 신청하기](https://services.google.com/fb/forms/kerastuner/)

---
### 같이 보기

* [케라스 기초 강좌](https://tykimos.github.io/lecture/)

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
