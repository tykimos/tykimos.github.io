---
layout: post
title:  "윈도우에서 케라스 설치하기"
author: 김태영
date:   2017-08-07 16:00:00
categories: Lecture
comments: true
image: http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_4.png
---
윈도우에서 케라스 개발 환경을 구축해보겠습니다. 진행 순서는 다음과 같습니다.

* 아나콘다 설치하기
* 프로젝트 디렉토리 만들기
* 가상 개발환경 만들기
* 웹기반 파이썬 개발환경인 주피터 노트북 설치
* 주요 패키지 설치
* 딥러닝 라이브러리 설치
* 설치 환경 테스트 해보기
* 딥러닝 엔진 바꾸기
* 다시 시작하기
* 오류 대처

---

### 아나콘다 설치하기

https://repo.continuum.io/archive/ 에 접속 후 시스템환경에 맞는 버전의 Anaconda3을 다운로드합니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_5.png)

다운로드 받은 파일을 실행시켜 다음과 같이 Anaconda를 설치합니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_16.png)

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_22.png)

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_17.png)
위 화면이 나오면 All Users로 체크하고 Next를 눌러 다음 단계를 진행합니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_1.png)

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_21.png)
첫번째 체크박스는 아래의 환경 변수 추가 단계를 자동으로 해주는 옵션이므로 체크합니다. 두번째 체크박스는 실습에 영향을 끼치지 않습니다. 필요에 따라 선택적으로 체크합니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_3.png)

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_6.png)

설치가 정상적으로 완료되지 않고 ‘Failed to create Anaconda menus’라는 메시지가 뜨는 경우,  제어판 > 시스템 및 보안 > 시스템 > 고급 시스템 설정 > 환경 변수에서 [시스템변수] 중 java와 관련된 환경변수가 있는지 확인합니다. 관련 변수가 있는 경우 해당 변수를 임시 저장해두었다가 일시적으로 삭제한 후 설치를 진행합니다. 오류 메시지가 뜨지 않고 설치가 정상적으로 완료되면 해당 변수를 다시 생성합니다.

제어판 > 시스템 및 보안 > 시스템 > 고급 시스템 설정 > 환경 변수에서 [시스템변수] 중 Path 에 아래 경로들을 추가합니다. 이미 경로가 있는 경우 다시 추가하지 않으셔도 됩니다.

```
    [추가할 경로]
    C:\ProgramData\Anaconda3
    C:\ProgramData\Anaconda3\Scripts
    C:\ProgramData\Anaconda3\Library\bin
```

이 때, Path에 파이썬 경로가 있는 경우 파이썬 경로 보다 앞 쪽에 위의 경로를 추가합니다. 아래 경로가 없는 경우에는 이 단계를 건너뜁니다.

```
    [파이썬 경로(예시)]
    C:\Python27
    C:\Python27\Scripts
    C:\Python27\Lib\site-packages
```

Windows키 + r을 눌러 cmd(명령 프롬프트)를 실행시키고, cmd 창에서 다음과 같이 명령어 입력 후 설치가 완료되었음을 확인합니다.
```
    >conda --version [Enter]
    conda 4.3.21
```    

그 다음 아래 명령어를 입력하여 파이썬이 잘 동작하는지 확인합니다. 파이썬이 정상적으로 실행되면 설치가 성공적으로 된 것입니다.
```
    >python [Enter]
```

---

### 프로젝트 디렉토리 만들기
Windows키+ r을 눌러 'cmd'를 입력하여 명령 프롬프트를 실행시킵니다. 이 때, 권한 문제를 막기 위해 관리자 권한으로 명령 프롬프트를 실행시킵니다. 다음 명령어를 입력하여 C드라이브로 이동합니다.

```
>cd c:\
c:\>_
```

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_13.png)

실습을 위해 “Projects”라는 폴더를 생성한 뒤 이동합니다.

```
c:\>mkdir Projects
c:\>cd Projects
c:\Projects>_
```
![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_8.png)

“keras_talk”라는 이름으로 케라스 프로젝트 하나를 생성합니다.

```
c:\Projects>mkdir keras_talk
c:\Projects>cd keras_talk
c:\Projects\keras_talk>_
```
![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_15.png)

---

### 가상 개발환경 만들기

프로젝트별로 개발환경이 다양할 수 있기 때문에 가상환경을 이용하면 편리합니다. 위에서 생성한 프로젝트에 가상 환경을 구축해보겠습니다. 명령 프롬프트에서 다음 명령어를 실행하여 가상환경을 생성합니다. 이 때, 권한 문제를 막기 위해 관리자 권한으로 명령 프롬프트가 실행되어 있어야 합니다. 설치를 확인하는 문장이 나타나면 ‘y’를 입력하여 설치를 진행합니다.

```
    c:\Projects\keras_talk>conda create -n venv python=3.6 anaconda
```

다음 명령으로 생성한 가상환경을 실행시킵니다. ‘(venv)’라는 문구가 입력창에 나타나면 성공적으로 가상환경이 실행된 것입니다.

```
    c:\Projects\keras_talk>activate venv
```

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_14.png)

---

### 웹기반 파이썬 개발환경인 주피터 노트북 설치 

다음 명령으로 주피터 노트북을 설치합니다. 중간에 설치를 묻는 창이 뜨면 ‘y’를 입력하여 설치를 진행합니다.

```
    (venv) c:\Projects\keras_talk>conda install -n venv ipython notebook
```

다음 명령으로 주피터 노트북을 실행시키면 명령 프롬프트 창에는 아래 그림과 같이 출력됩니다. 

```
    (venv) c:\Projects\keras_talk>jupyter notebook
```

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_18.png)

정상적으로 실행되면 아래와 같이 웹 브라우저가 실행됩니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_12.png)

다른 패키지를 설치하기 위해 명령 프롬프트 창에서 Control+C 를 입력하여 notebook을 종료시킵니다. 

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_7.png)

---

### 주요 패키지 설치

다음 명령어를 입력하여 케라스 사용에 필요한 주요 패키지들을 설치합니다. 중간에 설치를 묻는 창이 뜨면 ‘y’를 입력하여 설치를 진행합니다.

```
    (venv) c:\Projects\keras_talk>conda install -n venv numpy matplotlib pandas pydotplus h5py scikit-learn
    (venv) c:\Projects\keras_talk>conda install -n venv scipy mkl-service libpython m2w64-toolchain   
```

---

### 딥러닝 라이브러리 설치

다음 명령어를 입력하여 케라스 사용하는 딥러닝 라이브러리인 티아노(Theano)와 텐서플로우(Tensorflow)를 설치합니다. 둘 중 하나만 사용한다면 해당 라이브러리만 설치하시면 됩니다.

```
    (venv) c:\Projects\keras_talk>conda install -n venv theano pygpu
    (venv) c:\Projects\keras_talk>conda install -n venv git graphviz
    (venv) c:\Projects\keras_talk>pip install --ignore-installed --upgrade tensorflow
```

다음과 같이 명령어를 입력하여 케라스를 다운로드 받은 후 ‘cd’ 명령어를 이용하여 keras 폴더로 이동합니다. 

```
    (venv) c:\Projects\keras_talk>git clone https://github.com/fchollet/keras.git
    (venv) c:\Projects\keras_talk>cd keras
    (venv) c:\Projects\keras_talk\keras>_
```

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_9.png)

다음과 같이 명령어를 입력하여 케라스를 설치합니다.

```
    (venv) c:\Projects\keras_talk\keras>python setup.py install
```

---

### 설치 환경 테스트 해보기

#### 설치된 패키지 버전 확인

모든 환경이 정상적으로 설치되어 있는지 확인하기 위해 프로젝트 폴더로 이동하고, 다음과 같이 명령어를 입력하여 주피터 노트북을 실행시킵니다.

```
    (venv) c:\Projects\keras_talk\keras>cd ..
    (venv) c:\Projects\keras_talk>_
    (venv) c:\Projects\keras_talk>jupyter notebook
```

아래 그림처럼 우측 상단에 있는 'New' 버튼을 클릭해서 예제 코드를 작성할 파이썬 파일을 생성합니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_19.png)

성공적으로 파인썬 파일이 생성되었다면, 아래 그림처럼 코드를 작성할 수 있는 페이지가 띄워집니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_2.png)

녹색으로 표시된 영역에 아래 코드를 삽입한 뒤 'shift + enter'를 눌러서 실행시킵니다.

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

에러없이 다음과 같은 화면이 출력되면 정상적으로 작동되는 것입니다.

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

에러없이 다음과 같은 화면이 출력되면 정상적으로 작동되는 것입니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_10.png)

#### 딥러닝 모델 저장 기능 확인

아래 딥러닝 모델의 구성과 가중치를 저장 및 로딩하는 코드입니다. 마찬가지로 새로운 셀에서 실행시키기 위해 상단 메뉴에서 'Insert > Insert Cell Below'를 선택하여 새로운 셀을 생성합니다. 새로 생성된 셀에 아래 코드를 입력한 후 'shift + enter'를 눌러 해당 코드를 실행합니다.

```python
from keras.models import load_model

model.save('mnist_mlp_model.h5')
model = load_model('mnist_mlp_model.h5')
```

위 코드 실행 시 에러가 발생하지 않고, 로컬 디렉토리에 'mnist_mlp_model.h5' 파일이 생성되었으면 정상적으로 작동되는 것입니다. 지금까지 정상적으로 실행이 되었다면 상단 메뉴에서 'File > Save and Checkpoint'로 현재까지 테스트한 파일을 저장합니다. 

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_4.png)

---

### 딥러닝 엔진 바꾸기

백엔드로 구동되는 딥러닝 엔진을 바꾸려먼 'C:/Users/사용자이름/.keras/keras.json' 파일을 열어서 'backend' 부분을 수정하시면 됩니다. 만약 현재 설정이 텐서플로우일 경우 아래와 같이 표시됩니다.

```
    ...
    "backend": "tensorflow"
    ...
```

텐서플로우에서 티아노로 변경할 경우 위의 설정을 아래와 같이 수정합니다.

```
    ...
    "backend": "theano"
    ...
```

---

### 다시 시작하기

재부팅하거나 새로운 명령창에서 다시 시작할 때는 다음의 명령을 수행합니다.

```
    c:\Projects\keras_talk>activate venv
    (venv) c:\Projects\keras_talk>jupyter notebook
```

---

### 오류 대처

#### pydot 에러

> 딥러닝 모델 가시화 기능 실행 시 ‘GraphViz's executables not found’ 또는 ‘Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.’ 문장이 뜨면서 에러가 납니다.

이 오류는 graphviz가 정상적으로 설치되지 않았거나 경로가 설정되지 않은 경우에 발생합니다.

* http://www.graphviz.org/Download_windows.php 에 접속하여 graphviz-2.38.msi 파일을 다운로드 받습니다.
* graphviz-2.38.msi을 실행시켜 graphviz를 설치합니다.
* 설치가 완료되면 제어판 > 시스템 및 보안 > 시스템 > 고급 시스템 설정 > 환경 변수에서 다음과 같이 변수를 추가합니다.

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_11.png)

* 환경 변수의 [시스템 변수] 중 Path 에 다음과 같이 경로를 추가합니다.

```
    C:\Program Files (x86)\Graphviz2.38\bin
```

![img](http://tykimos.github.io/warehouse/2017-8-7-Keras_Install_on_Windows_20.png)

* 환경 변수를 저장한 후 jupyter notebook이 실행되고 있는 cmd 창을 종료하고 다시 시작합니다.
* 다시 위 예제 코드를 실행시켜서 잘 그림과 같이 이미지가 잘 나오면 성공적으로 설치된 것입니다.

#### 주피터 실행 에러

> 주피터 실행 시 아래와 같은 에러가 발생합니다. 
> Copy/paste this URL into your browser when you connect for the first time,
> to login with a token:
> http://localhost:8888/?token=7c0dxxx

이 경우 브라우저의 권한 등의 문제로 발생하는 것인데, 브라우저를 열어서 콘솔창에 출력된 링크로 한 번 접속하시면 해결됩니다.

#### 텐서플로우 라이브러리 import 에러

> 텐서플로우 라이브러리 import 시 아래와 같은 에러가 발생합니다.

Traceback (most recent call last):
  File "C:\...\Python36\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 18, in swig_import_helper
    return importlib.import_module(mname)
  File "C:\...\Python36\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 658, in _load_unlocked
  File "<frozen importlib._bootstrap>", line 571, in module_from_spec
  File "<frozen importlib._bootstrap_external>", line 922, in create_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
ImportError: DLL load failed: A dynamic link library (DLL) initialization routine failed.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\...\Python36\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "C:\...\Python36\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 21, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "C:\...\Python36\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 20, in swig_import_helper
    return importlib.import_module('_pywrap_tensorflow_internal')
  File "C:\...\Python36\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ModuleNotFoundError: No module named '_pywrap_tensorflow_internal'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\...\Python36\lib\site-packages\tensorflow\__init__.py", line 24, in <module>
    from tensorflow.python import *
  File "C:\...\Python36\lib\site-packages\tensorflow\python\__init__.py", line 49, in <module>
    from tensorflow.python import pywrap_tensorflow
  File "C:\...\Python36\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 74, in <module>
    raise ImportError(msg)
ImportError: Traceback (most recent call last):
  File "C:\...\Python36\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 18, in swig_import_helper
    return importlib.import_module(mname)
  File "C:\...\Python36\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 658, in _load_unlocked
  File "<frozen importlib._bootstrap>", line 571, in module_from_spec
  File "<frozen importlib._bootstrap_external>", line 922, in create_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
ImportError: DLL load failed: A dynamic link library (DLL) initialization routine failed.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\...\Python36\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "C:\...\Python36\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 21, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "C:\...\Python36\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 20, in swig_import_helper
    return importlib.import_module('_pywrap_tensorflow_internal')
  File "C:\...\Python36\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ModuleNotFoundError: No module named '_pywrap_tensorflow_internal'

Failed to load the native TensorFlow runtime.

See https://www.tensorflow.org/install/install_sources#common_installation_problems

for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.
이 경우 http://tykimos.github.io/warehouse/files/tensorflow-1.6.0-cp36-cp36m-win_amd64.whl 의 파일을 다운로드 받아 텐서플로우 라이브러리 재설치를 진행합니다. 재설치 시 권한 문제를 막기 위해 관리자 권한으로 명령 프롬프트를 실행합니다. 아래의 명령어를 입력하여 설치되어 있는 텐서플로우 라이브러리를 삭제합니다. 계속 진행 여부를 묻는 창이 뜨면 'y'를 입력하여 계속 진행합니다.

```
    (venv) c:\Projects\keras_talk>pip uninstall tensorflow
    .
    .
    .
    Successfully uninstalled tensorflow-1.6.0
    (venv) c:\Projects\keras_talk>_
```

다운로드 받은 파일을 'c:/Projects/keras_talk' 경로로 이동시킨 후 아래의 명령어를 입력하여 설치를 진행합니다. 계속 진행 여부를 묻는 창이 뜨면 'y'를 입력하여 계속 진행합니다. 경우에 따라 텐서플로우 설치에 필요한 다른 라이브러리가 같이 설치될 수 있습니다.

```
    (venv) c:\Projects\keras_talk>pip install tensorflow-1.6.0-cp36-cp36m-win_amd64.whl
    .
    .
    .
    Successfully installed tensorflow-1.6.0
    (venv) c:\Projects\keras_talk>_
```

설치가 완료되면 주피터 노트북을 실행하여 텐서플로우 라이브러리가 정상적으로 import 되는 지 확인합니다.

---

### 요약

윈도우 환경에서 케라스를 구동하기 위해, 주피터 노트북 개발환경, 주요 패키지, 딥러닝 라이브러리 설치 및 구동을 해봤습니다. 컴퓨터 환경에 따라 설치 및 테스트 시에 오류가 발생할 수 있습니다. 주로 발생하는 오류와 각 오류별 대처 방법에 대해서 살펴보왔습니다.

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/lecture/)
* 이전 : [케라스 이야기](https://tykimos.github.io/2017/01/27/Keras_Talk/)
* 이전 : [맥에서 케라스 설치하기](https://tykimos.github.io/2017/08/07/Keras_Install_on_Mac/)
* 다음 : [데이터셋 이야기](https://tykimos.github.io/2017/03/25/Dataset_and_Fit_Talk/)
