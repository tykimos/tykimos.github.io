---
layout: post
title:  "맥에서 케라스 설치하기"
author: 김태영
date:   2017-08-07 16:00:00
categories: Lecture
comments: true
image: http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Mac_4.png
---
본 강좌에서는 맥에서 케라스 개발 환경을 구축해보겠습니다.


### 프로젝트 디렉토리 만들기

사용자 로컬 디렉토리에서부터 시작하겠습니다. 아래 명령을 입력하면 사용자 로컬 디렉토리로 이동합니다.

```
$ cd ~
```

"Projects"라는 폴더를 생성 뒤 이동합니다.

```
~ $ mkdir Projects
~ $ cd Projects
Projects $ _
```

케라스 프로젝트를 하나 생성합니다. 이름은 "keras_talk"라고 해보겠습니다. 

```
Projects $ mkdir keras_talk
Projects $ cd keras_talk    
keras_talk $ _
```

### 가상 개발환경 만들기

프로젝트별로 개발환경이 다양할 수 있기 때문에 가상환경을 이용하면 편리합니다. 위에서 생성한 프로젝트에 가상 환경을 구축해보겠습니다. 가상환경을 제공하는 virtualenv을 먼저 설치하겠습니다. 이 과정은 프로젝트 별로 할 필요는 없고, 시스템에 한 번만 수행하면 됩니다.

```
keras_talk $ sudo pip install virtualenv
```

virtualenv를 설치했다면 실제 가상환경을 만들겠습니다. 'ls' 명령어를 입력하면, 프로젝트 폴더 내에 'venv'라는 폴더가 생성됨을 확인 할 수 있습니다.

```
keras_talk $ virtualenv venv
...
Installing setuptools, pip, wheel...done.
keras_talk $ ls
venv
```
    
가상환경을 만들었으니 가상환경을 실행하겠습니다. '(venv)' 라는 문구가 입력창에 보이면 성공적으로 가상환경이 실행된 것입니다.

```
keras_talk $ source venv/bin/activate 
(venv) keras_talk $ _
```

### 웹기반 파이썬 개발환경인 주피터 노트북 설치 

주피터 노트북은 파이썬 코드를 웹 환경에서 작성 및 실행시킬 수 있도록 제공하는 툴입니다. pip 툴을 이용하여 주피터 노트북을 설치합니다.

```
(venv) keras_talk $ pip install ipython[notebook]
```

> "Your pip version is out of date, ..."이라는 에러가 발생하면 pip 버전을 업그레이드 한 후 다시 설치합니다.
> ``` 
(venv) keras_talk $ pip install --upgrade pip
(venv) keras_talk $ pip install ipython[notebook]
> ```

주피터 노트북을 다음 명령으로 실행시킵니다. 

```
(venv) keras_talk $ jupyter notebook
```

정상적으로 설치되었다면 웹 브라우저가 실행되면서 아래와 같은 페이지가 띄워집니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Mac_1.png)

계속 다른 패키지를 설치하기 위해 터미널 창에서 'Control-C'를 누른 뒤 'y'를 입력하여 ipython notebook를 종료시킵니다.

```
Shutdown this notebook server (y/[n])?  y
(venv) keras_talk $ _
```

### 주요 패키지 설치

케라스를 사용한 데 있어서 필요한 주요 패키지를 다음 명령을 통해서 설치합니다.

```
(venv) keras_talk $ pip install numpy
(venv) keras_talk $ pip install scipy
(venv) keras_talk $ pip install scikit-learn
(venv) keras_talk $ pip install matplotlib
(venv) keras_talk $ pip install pandas
(venv) keras_talk $ pip install pydot
(venv) keras_talk $ pip install h5py
```

### 딥러닝 라이브러리 설치

케라스에서 사용하는 딥러닝 라이브러리인 티아노(Theano)와 텐서플로우(Tensorflow)를 설치합니다. 둘 중에 하나만 사용한다면 해당하는 것만 설치하시면 됩니다. 

```
(venv) keras_talk $ pip install theano
(venv) keras_talk $ pip install tensorflow
```

성공적으로 설치하였다면, 케라스를 설치합니다.

```
(venv) keras_talk $ pip install keras
```

### 설치 환경 테스트 해보기

#### 설치된 패키지 버전 확인

케라스가 정상적으로 설치되어 있는 지 확인하기 위해 예제 코드를 실행시켜보겠습니다. 먼저 주피터 노트북을 실행시킵니다.

```
(venv) keras_talk $ jupyter notebook
```

아래 그림처럼 우측 상단에 있는 'new' 버튼을 클릭해서 예제 코드를 작성할 파이션 파일을 생성합니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Mac_2.png)

성공적으로 파인썬 파일이 생성되었다면, 아래 그림처럼 코드를 작성할 수 있는 페이지가 띄워집니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Mac_3.png)

녹색 박스로 표시된 영역에 아래 코드를 삽입한 뒤 'shift키 + enter키'를 눌러서 실행시킵니다.


```python
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import pydot
import h5py

import theano
import tensorflow
import keras

print('scipy ' + scipy.__version__)
print('numpy ' + numpy.__version__)
print('matplotlib ' + matplotlib.__version__)
print('pandas ' + pandas.__version__)
print('sklearn ' + sklearn.__version__)
print('pydot ' + pydot.__version__)
print('h5py ' + h5py.__version__)

print('theano ' + theano.__version__)
print('tensorflow ' + tensorflow.__version__)
print('keras ' + keras.__version__)
```

각 패키지별로 버전이 표시되면 정상적으로 설치가 된 것입니다. 

#### 딥러닝 기본 모델 구동 확인

아래 코드는 기본적인 딥러닝 모델에 손글씨 데이터셋을 학습시킨 뒤 평가하는 기본 예제입니다. 새로운 셀에서 실행시키기 위해 상단 메뉴에서 'Insert > Insert Cell Below'을 선택하여 새로운 셀을 생성합니다. 새로 생긴 셀에 아래 코드를 입력한 후 'shift키 + enter키'를 눌러 해당 코드를 실행합니다.


```python
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5, batch_size=32)

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print('loss_and_metrics : ' + str(loss_and_metrics))
```

에러없이 다음과 같이 화면이 출력되면 정상적으로 작동되는 것입니다.

```
Epoch 1/5
60000/60000 [==============================] - 1s - loss: 0.6558 - acc: 0.8333     
Epoch 2/5
60000/60000 [==============================] - 1s - loss: 0.3485 - acc: 0.9012     
Epoch 3/5
60000/60000 [==============================] - 1s - loss: 0.3037 - acc: 0.9143     
Epoch 4/5
60000/60000 [==============================] - 1s - loss: 0.2759 - acc: 0.9222     
Epoch 5/5
60000/60000 [==============================] - 1s - loss: 0.2544 - acc: 0.9281     
 8064/10000 [=======================>......] - ETA: 0sloss_and_metrics : [0.23770418465733528, 0.93089999999999995]
 ```

#### 딥러닝 모델 가시화 기능 확인

아래 딥러닝 모델 구성을 가시화하는 코드입니다. 마찬가지로 새로운 셀에서 실행시키기 위해 상단 메뉴에서 'Insert > Insert Cell Below'을 선택하여 새로운 셀을 생성합니다. 새로 생긴 셀에 아래 코드를 입력한 후 'shift키 + enter키'를 눌러 해당 코드를 실행합니다.


```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```

에러없이 다음과 같이 화면이 출력되면 정상적으로 작동되는 것입니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Mac_4.png)

#### 딥러닝 모델 저장 기능 확인

아래 딥러닝 모델의 구성 및 가중치를 저장 및 로딩하는 코드입니다. 마찬가지로 새로운 셀에서 실행시키기 위해 상단 메뉴에서 'Insert > Insert Cell Below'을 선택하여 새로운 셀을 생성합니다. 새로 생긴 셀에 아래 코드를 입력한 후 'shift키 + enter키'를 눌러 해당 코드를 실행합니다.


```python
from keras.models import load_model

model.save('mnist_mlp_model.h5')
model = load_model('mnist_mlp_model.h5')
```

위 코드 실행 시 에러가 발생하지 않고, 로컬 디렉토리에 'mnist_mlp_model.h5' 파일이 생성되었으면 정상적으로 작동되는 것입니다. 지금까지 정상적으로 실행이 되었다면 상단 메뉴에서 'File > Save and Checkpoint'로 현재까지 테스트한 파일을 저장합니다. 

---

### 요약

맥 환경에서 케라스를 구동하기 위해, 주피터 노트북 개발환경, 주요 패치키, 딥러링 라이브러리 설치 및 구동을 해봤습니다. 

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/lecture/)
