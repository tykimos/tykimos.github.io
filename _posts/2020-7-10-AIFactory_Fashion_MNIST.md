---
layout: post
title:  "패션 아이템 이미지 분류 과제 제출 방법"
author: 김태영
date:   2020-07-10 12:00:00
categories: etc
comments: true
image: http://tykimos.github.io/warehouse/2020-7-10-AIFactory_Fashion_MNIST_title_2.png
---

패션 아이템 이미지 분류 모델을 실습한 뒤 모델에서 추론한 결과를 AIFactory에 제출한 뒤 점수를 확인하는 방법에 대해서 아래 동영상으로 설명드리겠습니다.

[![img](http://tykimos.github.io/warehouse/2020-7-10-AIFactory_Fashion_MNIST_title_2.png)](https://www.youtube.com/watch?v=CSQvWf4rbiw&feature=youtu.be){: target="_blank"}

---
### 샘플 코드

간단히 실습해볼 수 있는 샘플 코드를 아래와 같이 작성해봤습니다. 

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

prob_pred = model.predict(test_images)
prob_label = prob_pred.argmax(axis=-1)

np.savetxt('submit.txt', prob_label,fmt='%d')
```

---
### 마무리

위 코드로 제출하면 간단한 모델로도 87% 이상의 정확도가 나옴을 확인할 수 있습니다. CNN 등의 다양한 레이어 및 기법을 이용해서 정확도를 더 높혀보시길 바랍니다~

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
