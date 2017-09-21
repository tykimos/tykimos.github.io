---
layout: post
title:  "케라스와 텐서플로우와의 통합"
author: Taeyoung, Kim
date:   2017-02-22 01:00:00
categories: News
comments: true
image: http://tykimos.github.io/Keras/warehouse/2017-2-22_Integrating_Keras_and_TensorFlow_1.png
---
지난 주에 개최된 TensorFlow DEV SUMMIT 2017 행사에서 프랑소와 쏠레(François Chollet)님이 케라스(Keras)와 텐서플로우(TensorFlow)와의 통합이란 주제로 발표를 했습니다. 발표에서 이번 통합이 케라스 사용자와 텐서플로우 사용자에게 어떤 의미를 뜻하는 지를 설명하였고, 비디오 QA 예제를 통해 텐서플로우에서 케라스를 어떻게 사용하는 지를 보여주었습니다. 케라스 사용자인 저에겐 중요한 발표이기도 하고, 케라스가 가장 빠른 성장세를 보이고 있는 프레임워크 중 하나이기에 주요 내용을 정리해보고자 합니다.

[![video](http://tykimos.github.io/Keras/warehouse/2017-2-22_Integrating_Keras_and_TensorFlow_1.png)](https://youtu.be/UeheTiBJ0Io)

---

### 케라스 목적

케라스의 목적은 다음의 한 문장으로 표현될 수 있습니다.

    케라스는
    많은 이들이 딥러닝을 쉽게 접할 수 있도록,
    다양한 플랫폼 위에서 딥러닝 모델을 만들 수 있는 
    API이다.
    
이 문장에는 많은 것을 내포하고 있는데요.
- 먼저 많은 이들이 사용하려면 쉬워야 합니다. 케라스 개발자는 조만간 딥러닝 툴박스는 딥러닝 전문가나 연구자가 아니더라도 쉽게 사용할 수 있게 될 것이라고 믿습니다.
- 케라스는 라이브러리나 코드가 아닌 `API 스펙`으로 되길 바랍니다. 즉 케라스 API 하나로 여러 플랫폼 위에서 동작되는 것을 목표로 삼고 있습니다. Theano나 TensorFlow 위에서는 이미 동작을 하고 있고 미래에는 마이크로소프트사의 CNTK 플랫폼 위에서도 동작이 될 겁니다.

---

#### 어떤 일이 벌어지고 있는가?

이번 통합에서 케라스가 텐서플로우 위에서 새롭게 구현되었습니다. 따라서 다음과 같은 일이 가능해집니다.

- `tf.keras`으로 텐서플로우 안에서 케라스를 사용할 수 있습니다.
- 코어 텐서플로우 레이어와 케라스 레이어는 같은 객체입니다.
- 케라스 모델이 텐서플로우 코어에서 사용가능합니다.
- 깊은 통합으로 `Experiment`와 같은 텐스플로우의 기능을 사용할 수 있습니다. 


#### 케라스 사용자에서의 의미

텐서플로우-케라스는 텐서플로우 코어 위에서 만들어졌기 때문에,

- 순수 텐서플로우 기능과 케라스 기능이 쉽게 섞여지고 매칭됩니다.
- 케라스 사용자는 다음의 텐서플로우의 기능을 사용할 수 있습니다.
    - Distributed training
    - Cloud ML
    - Hyperparameter tuning
    - TF-Serving
    
#### 텐서플로우 사용자에서의 의미

- 모델 정의할 때 케라스의 고차원 API를 사용할 수 있습니다.
- 텐서플로우 코어와 tf.keras와의 깊은 호환성 때문에 유연성에 대한 손실이 없습니다.
- Experiment, Cloud ML, TF-Serving와 같은 당신의 TF workflow를 기존의 케라스 코드에 쉽게 적용가능합니다.

---

### 예제

분산 텐서플로우 워크플로우(distributed TensorFlow workflow)안에서의 케라스 기반의 비디오-QA 모델 예제를 살펴 보겠습니다. 비디오-QA 문제를 풀기 위한 딥러닝 모델 개념도는 아래 그림의 왼쪽과 같습니다. 이를 레이어 구성으로 표시해보면 아래 그림의 오른쪽과 같습니다.

![model](http://tykimos.github.io/Keras/warehouse/2017-2-22_Integrating_Keras_and_TensorFlow_2.png)

이 모델을 케라스 코드로 작성하면 다음과 같습니다. (문제 및 코드의 상세한 설명은 주제를 벗어나는 것 같아 생략하도록 하겠습니다.)


```python
video = tf.keras.layers.Input(shape=(None, 150, 150, 3))
cnn = tf.keras.applications.InceptionV3(weights='imagenet',
                                        include_top=False,
                                        pool='avg')
cnn.trainable = False
encoded_frames = tf.keras.layers.TimeDistributed(cnn)(video)
encoded_vid = tf.keras.layers.LSTM(256)(encoded_frames)

question = tf.keras.layers.Input(shape=(100), dtype='int32')
x = tf.keras.layers.Embedding(10000, 256, mask_zero=True)(question)
encoded_q = tf.keras.layers.LSTM(128)(x)

x = tf.keras.layers.concat([encoded_vid, encoded_q])
x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
outputs = tf.keras.layers.Dense(1000)(x)

model = tf.keras.models.Model([video, question], outputs)
model.compile(optimizer=tf.AdamOptimizer(), 
              loss=tf.softmax,_crossentropy_with_logits)

train_panda_dataframe = pandas.read_hdf(...)

train_inputs = tf.inputs.pandas_input_fn(train_pada_dataframe,
                                         batch_size=32,
                                         shuffle=True,
                                         target_column='answer')

eval_inputs = tf.inputs.pandas_input_fn(...)

exp = tf.training.Experiment(model,
                             train_input_fn=train_inputs,
                             eval_input_fn=eval_inputs)
```


      File "<ipython-input-2-526432ea0bf3>", line 21
        train_panda_dataframe = pandas.read_hdf(...)
                                                ^
    SyntaxError: invalid syntax



---

### 릴리즈 계획

- TF 1.1에서 contrib.keras으로서 Keras 모듈을 사용할 수 있으며, 
- 최종적으로 TF 1.2에서 tf.keras로 사용할 수 있을 겁니다.

케라스 개발자인 프랑소와 쏠레는 `이것이 TensorFlow와 딥러닝을 모든 사람들이 이용할 수 있게 할 큰 걸음`이라고 언급하면서 발표를 마쳤습니다.





---

### 같이 보기

* [케라스 강좌 목차](https://tykimos.github.io/Keras/2017/01/27/Keras_Lecture_Contents/)
* [딥러닝 이야기/케라스 이야기](https://tykimos.github.io/Keras/2017/01/27/Keras_Talk/)


```python

```
