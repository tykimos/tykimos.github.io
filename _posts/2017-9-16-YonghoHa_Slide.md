---
layout: post
title:  "하용호님의 '네 뇌에 딥러닝 인스톨' + 딥브릭 + 실습"
author: 김태영
date:   2017-09-16 01:00:00
categories: Lecture
comments: true
image: http://tykimos.github.io/warehouse/2017-9-16-YonghoHa_Slide_title.png
---
딥러닝 개념을 아주 쉽고 제미있게, 직관적으로 설명한 하용호님의 '네 뇌에 딥러닝 인스톨' 자료를 공유합니다. 이 발표자료에는 케라스로 구현된 VGG16 모델을 이해하기 위한 여정이 시작되는 데요. 그 여정의 끝으로 딥브릭을 통해 뇌에 모델을 각인시킨 후 실습을 통해 손으로 익혀보겠습니다. 먼저 그 유명한 '네 뇌의 딥러닝 인스톨' 발표자료 입니다. (주의)'한 번 클릭하면 10분~15분동안 다른 일을 할 수가 없습니다.

[발표자료 링크](https://www.slideshare.net/yongho/ss-79607172)

![img](http://tykimos.github.io/warehouse/2017-9-16-YonghoHa_Slide_01.png)

여기서 딥브릭이 무엇인지 궁금하시다면, ['딥브릭(DeepBrick) 이야기'](https://tykimos.github.io/2017/08/09/DeepBrick_Talk/)

---
### VGG16 모델

딥러닝 모델이 수만가지가 있지만, 하용호님 발표자료엔 왜 VGG16 모델이 소개되었을까요? VGG16이 그렇게 간단한 모델도 아닌데 말이죠. 아래 사진으로 모두 설명이 가능할 것 같습니다. 참고로 이 사진으로 유재준님이 CVPR 2017 발표현장에서 찍은 사진입니다.

![img](http://tykimos.github.io/warehouse/2017-9-16-YonghoHa_Slide_02.jpg)

VGG16 모델은 영상 처리에 있어서는 빠질 수 없는 일꾼이죠. 지금 이 순간에도 아마 GPU 머신에 의해 계속 시달리고 있을 겁니다. 케라스에서도 VGG16 코드를 공식적으로 제공하고 있습니다. [VGG16 소스코드 보기](https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py)

같이 보기
* 컨볼루션 신경망에 대해서 더 알고 싶으시다면, [컨볼루션 신경망 레이어 이야기](https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/) 클릭
* 컨볼루션 신경망 모델을 만들어보고 싶으시다면, [컨볼루션 신경망 모델 만들어보기](https://tykimos.github.io/2017/03/08/CNN_Getting_Started/) 클릭
* 영상을 입력하여 이진분류 모델을 만들고 싶다면, [영상입력 이진분류 모델 레시피](https://tykimos.github.io/2017/08/18/Image_Input_Binary_Classification_Model_Recipe/) 클릭
* 영상을 입력하여 다중클래스분류 모델을 만들고 싶다면, [영상입력 다중클래스분류 모델 레시피](https://tykimos.github.io/2017/08/18/Image_Input_Multiclass_Classification_Model_Recipe/) 클릭

#### 모델 구성 코드

소스코드 중 모델 구성에 관련된 핵심 부분만 발췌해봤습니다. 하용호님의 소스코드가 블록 단위로 구성된 함수를 호출하는 식이라 더 깔끔합니다만, 미리 학습된 가중치를 로딩하기 위해서 케라스에서 제공하는 원본 그대로를 살펴보겠습니다.


```python
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

if include_top:
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
else:
    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)
```

---
### 딥브릭으로 쌓기

VGG16 모델을 딥브릭으로 쌓아보겠습니다. 

#### 브릭 목록

먼저 필요한 브릭 목록입니다. 더 다양한 브릭을 보고 싶다면, [딥브릭 스튜디오](https://tykimos.github.io/DeepBrickStudio/)

|브릭|이름|설명|
|:-:|:-:|:-|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dataset_Vector_s.png)|Input data, Labels|1차원의 입력 데이터 및 라벨입니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dataset2D_s.png)|2D Input data|2차원의 입력 데이터입니다.주로 영상 데이터를 의미하며 샘플수, 너비, 높이, 채널수로 구성됩니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dense_s.png)|Dense|모든 입력 뉴런과 출력 뉴런을 연결하는 전결합층입니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Conv2D_s.png)|Conv2D|필터를 이용하여 영상 특징을 추출합니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_MaxPooling2D_s.png)|MaxPooling2D|영상에서의 사소한 변화가 특징 추출에 크게 영향을 미치지 않도록 합니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Flatten_s.png)|Flatten|2차원의 특징맵을 전결합층으로 전달하기 위해서 1차원 형식으로 바꿔줍니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_softmax_s.png)|softmax|활성화 함수로 입력되는 값을 클래스별로 확률 값이 나오도록 출력시킵니다. 이 확률값을 모두 더하면 1이 됩니다. 다중클래스 모델의 출력층에 주로 사용되며, 확률값이 가장 높은 클래스가 모델이 분류한 클래스입니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_Relu_s.png)|relu|활성화 함수로 주로 은닉층에 사용됩니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_relu_2D_s.png)|relu|활성화 함수로 주로 Conv2D 은닉층에 사용됩니다.|

#### Block 1

먼저 첫번째 블록을 살펴보겠습니다. 컨볼루션 레이어 2개와 맥스풀링 레이어 1개로 구성되어 있습니다.


```python
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
```

첫번째 블록을 딥브릭으로 표현하면 다음과 같습니다.

![img](http://tykimos.github.io/warehouse/2017-9-16-YonghoHa_Slide_02m.png)

#### Block 5

다섯번째 블록을 살펴보겠습니다. 컨볼루션 레이어 3개와 맥스풀링 레이어 1개로 구성되어 있습니다.


```python
# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
```

다섯번째 블록을 딥브릭으로 표현하면 다음과 같습니다.

![img](http://tykimos.github.io/warehouse/2017-9-16-YonghoHa_Slide_03m.png)

#### Top

마지막 블록을 살펴보겠습니다. 전결합층을 위한 렐루(relu) 활성화 함수를 가진 덴스(Dense)레이어와 출력층을 위한 소프트맥스(softmax) 활성화함수를 가진 덴스(Dense)레이어로 구성되어 있습니다.


```python
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(classes, activation='softmax', name='predictions')(x)
```

마지막 블록을 딥브릭으로 표현하면 다음과 같습니다.

![img](http://tykimos.github.io/warehouse/2017-9-16-YonghoHa_Slide_04m.png)

####  전체

VGG16 모델 전체을 딥브릭으로 표현하면 다음과 같습니다.

![img](http://tykimos.github.io/warehouse/2017-9-16-YonghoHa_Slide_01m.png)

---
### 미리 학습된 VGG16  모델 사용하기

VGG16 모델은 층이 깊어 학습하는 데 오랜 시간이 걸립니다. 다행이도 케라스에서는 VGG16 모델에 대해 미리 학습된 가중치를 제공하고 있습니다. 우리는 미리 학습된 가중치를 VGG16 모델에 셋팅하여 사용할 예정입니다. 가중치 불러온다는 것이 생소하시다면, [학습 모델 보기/저장하기/불러오기](https://tykimos.github.io/2017/06/10/Model_Save_Load/) 클릭

아래는 테스트 시에 사용할 이미지입니다.

![img](http://tykimos.github.io/warehouse/elephant.jpg)

#### 텐서플로우로 백엔드 설정하기

먼저 백엔드를 케라스 설정 파일(keras.json)에서 텐서플로우로 지정하겠습니다. 단 경로는 설치 환경에 따라 차이가 날 수 있습니다. 

    vi ~/.keras/keras.json

keras.json 파일을 열어서 다음과 같이 수정합니다.

    {
        "image_data_format": "channels_last",
        "epsilon": 1e-07,
        "floatx": "float32",
        "backend": "tensorflow"
    }
    
여기서 중요한 인자는 `backend`이 입니다. 이 항목이 `tensorflow`로 지정되어 있어야 합니다.

#### 테스트 이미지 다운로드

[이미지다운](http://tykimos.github.io/warehouse/elephant.jpg)

#### 전체 소스

VGG16() 함수를 통해서 VGG16 모델 구성 및 미리 학습된 가중치 로딩을 손쉽게 할 수 있습니다. 테스트를 위해서 임의의 영상을 입력으로 넣아보세요.


```python
# 0. 사용할 패키지 불러오기
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications import imagenet_utils
import numpy as np

# 1. 모델 구성하기
model = VGG16(weights='imagenet')

# 2. 모델 사용하기 

# 임의의 이미지 불러오기
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
xhat = image.img_to_array(img)
xhat = np.expand_dims(xhat, axis=0)
xhat = preprocess_input(xhat)

# 임의의 이미지로 분류 예측하기
yhat = model.predict(xhat)

# 예측 결과 확인하기
P = imagenet_utils.decode_predictions(yhat)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
```

    Using TensorFlow backend.


    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    553459712/553467096 [============================>.] - ETA: 0sDownloading data from https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    40960/35363 [==================================] - 0s 
    1. African_elephant: 49.05%
    2. tusker: 46.28%
    3. Indian_elephant: 4.67%
    4. warthog: 0.00%
    5. triceratops: 0.00%


위 코드를 실행하면 먼저 다음 파일들을 다운로드 받습니다. 

* vgg16_weights_tf_dim_ordering_tf_kernels.h5 : VGG16 가중치 파일입니다.
* imagenet_class_index.json : 인덱스 값에 해당하는 클래스 정보를 얻을 수 있는 파일입니다. 

한 번 다운로드 받아 놓으면 재실행 시에는 다운로드 하지 않습니다. 그 다음 문장이 예측 결과입니다. 1위에서 5위까지 표시가 되었는데, 1위, 2위가 코끼리로 나오네요~

---

### 요약

하용호님 발표자료에 사용하였던 VGG16 모델에 대해 딥브릭으로 형상화 시켜보고, 케라스 코드를 통해 실습해봤습니다. 

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/lecture/)
