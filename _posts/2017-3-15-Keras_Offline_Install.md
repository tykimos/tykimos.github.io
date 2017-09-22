---
layout: post
title:  "텐서플로우, 티아노, 케라스 오프라인 설치 (주피터 포함)"
author: Taeyoung, Kim
date:   2017-03-15 12:00:00
categories: Lecture
comments: true
---
이후 텐서플로우, 티아노, 케라스 오프라인 설치는 인스페이스 딥러닝랩([InSpace Deep Learning Lab](https://inspace4u.github.io/dllab/)) 페이지에서 제공합니다. 아래 링크를 참고하세요. 파일 요청은 본 포스트 댓글로도 가능합니다. 댓글에 원도우 버전 명기해주세요~

* [Windows 7에서 텐서플로우, 케라스 오프라인 설치](https://inspace4u.github.io/dllab/lecture/2017/09/04/Windows_7_Keras_Offline_Install.html)
* [Windows 10에서 텐서플로우, 케라스 오프라인 설치](https://inspace4u.github.io/dllab/lecture/2017/09/04/Windows_10_Keras_Offline_Install.html)


> 버전 관리 등의 이유로 본 게시물은 더 이상 유효하지 않습니다. 단 댓글 기능을 통해 파일 요청을 하실 수 있습니다.

본 포스트에서는 텐서플로우, 티아노, 케라스를 오프라인으로 설치를 해보겠습니다. 파이썬 기반의 패키지들은 온라인에서 쉽게 설치가 가능하지만 오프라인에서는 설치가 조금 까다롭습니다. 보안 문제로 망분리된 서버나 워크스테이션에서 딥러닝 모델을 사용하려면 오프라인 설치가 필요합니다. '설치 준비' 파트에서 원하시는 환경에 맞는 파일을 다운로드 받아서, '환경 및 버전별 설치법' 파트에서 해당하는 항목을 찾아 순서대로 설치하시면 됩니다. 

---

### 설치 준비

#### 설치 파일 다운로드

설치하고자 하는 환경에 해당하는 파일을 다운로드 받습니다. 고용량이라 파일 다운로드 링크는 아래 댓글 창에 이메일이나 연락처를 남겨주시면 보내드리도록 하겠습니다. 요청하실 때 밑줄 친 파일명을 기입해주세요.

* 윈도우 7 64비트 환경, TensorFlow 1.0.0, Theano 0.9.0, Keras 1.2.2 : <U>tf100_th090_keras122_cpu_gpu_win7_x64.zip</U> (5.86GB)
    * TensorFlow '1.0.0'
    * TensorFlow GPU '1.0.0'
    * Theano '0.9.0rc1'    
    * Keras '1.2.2'
    
#### 설치 환경 정리

기존에 설치된 프로그램과 충돌이 날 수 있으므로 설치를 하기 전에 파이썬에 관련된 기존 프로그램을 삭제합니다. 

1. Anaconda 2가 설치되어 있다면 Anaconda 2를 삭제합니다.
1. 제어판 > 프로그램 및 기능(또는 프로그램 추가/제거)에서 python 2.7.13(Anaconda 4.3.0 64-bit)을 선택하여 삭제할 수 있습니다.
1. Anaconda이외의 다른 파이썬이 설치되어 있다면 삭제합니다.

### 환경 및 버전별 설치법

---

### 윈도우 7 64비트 환경, TensorFlow 1.0.0, Theano 0.9.0, Keras 1.2.2

"tf100_th090_keras122_cpu_gpu_win7_x64.zip" 파일(5.86GB)을 다운로드 받습니다. 압축 파일을 풀면 아래와 같은 폴더 및 파일들이 있습니다.

* Anaconda3-4.2.0-Windows-x86_64.exe
* cuda_8.0.61_windows.exe
* cudatools_4.0.17_win_64.msi
* cudnn-8.0-windows7-x64-v5.0-ga.zip
* [폴더] Keras
* [폴더] packages
* requirements.txt
* [폴더] Theano
* vs2015.com_enu.iso

#### Visual Studio 설치
1. vs2015.com_enu.iso 파일을 실행시켜 Visual Studio 2015를 설치합니다. iso 파일을 수 있어야 합니다.
1. 설치가 완료되면 아래 두 개 항목이 시스템 환경 변수로 잘 설정되어 있는 지 확인합니다. '제어판 > 시스템 > 고급 시스템 설정 > 고급 > 환경 변수'의 메뉴로 확인할 수 있습니다. 만약 없을 경우 추가합니다. 
    * PATH - C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin
    * VS140COMNTOOLS - C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools\
              
#### Anaconda 설치
1. "Anaconda3-4.2.0-Windows-x86_64.exe" 파일을 실행하여 설치를 시작합니다.
1. 사용자 설정 부분에서는 'all user'로 선택합니다.
1. 이후 항목들에 대해서는 모두 체크하여 설치를 진행합니다.
1. 설치가 완료 후 시스템 환경 변수 PATH에 아래 항목이 있는 지 확인합니다. '제어판 > 시스템 > 고급 시스템 설정 > 고급 > 환경 변수'의 메뉴로 확인할 수 있습니다. 만약 없을 경우 추가합니다. 
    * C:\Program Files\Anaconda3
    * C:\Program Files\Anaconda3\Scripts
    * C:\Program Files\Anaconda3\Library\bin

#### 파이썬 패키지 설치
1. 다운로드 폴더 및 파일 중 아래 목록을 로컬의 특정 경로('/inspace/dl/'이라고 가정)에 복사합니다.
    * packages 폴더
    * Theano 폴더
    * Keras 폴더
    * requirements.txt
        
1. 명령 프롬프트 창을 열어서 '/inspace/dl/'로 디렉토리를 이동합니다. 참고로 '[ENTER]'는 엔터키를 입력하라는 의미입니다.
```
> cd /inspace/dl/ [ENTER]
```
1. 다음 명령어를 입력하여 통합 패키지를 설치합니다. 
```
> pip install --no-index --find-links=./packages -r requirements.txt [ENTER]
```
    
#### TensorFlow 설치

1. 다음 명령어를 입력하여 TensorFlow를 설치합니다.   
```
> pip install --no-index --find-links=./packages tensorflow [ENTER]
```
1. CMD 창에서 다음과 같이 실행하여 정상적으로 설치된 것을 확인합니다.
```
> python [ENTER]   
>>> import tensorflow as tf [ENTER]
>>> hello = tf.constant(‘Hello, Tensorflow’) [ENTER]
>>> sess = tf.Session() [ENTER]
>>> print(sess.run(hello)) [ENTER]
```
1. “Hello, Tensorflow”라는 문구가 출력되면 정상적으로 설치된 것 입니다.

#### GPU용 TensorFlow 설치

1. 다음 명령어를 입력하여 TensorFlow GPU를 설치합니다.
```
> pip install --no-index --find-links=./packages tensorflow-gpu
```
1. "cuda_8.0.61_windows.exe"을 실행시켜 CUDA를 설치합니다.
1. "cudatools_4.0.17_win_64.msi"을 실행시켜 CUDA toolkit 설치합니다.
1. "cudnn-8.0-windows7-x64-v5.0-ga.zip"을 압축 해제합니다.
1. 압축 해제한 폴더에서 아래와 같이 파일을 복사합니다.

|구분|원본경로|대상경로|
|:-:|-|-|
|bin|\cudnn-8.0-windows7-x64-v5.0-ga\cuda\bin|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin|
|include|\cudnn-8.0-windows7-x64-v5.0-ga\cuda\include|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include|
|lib|\cudnn-8.0-windows7-x64-v5.0-ga\cuda\lib\x64|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64|

1. 설치가 완료 후 시스템 환경 변수 PATH에 아래 항목이 있는 지 확인합니다. '제어판 > 시스템 > 고급 시스템 설정 > 고급 > 환경 변수'의 메뉴로 확인할 수 있습니다. 만약 없을 경우 추가합니다. 
    * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin
    * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp

#### Theano 설치

1. Theano 폴더로 이동한 후 Theano를 설치합니다. 
```   
> cd /inspace/dl/Theano [ENTER]
> python setup.py install [ENTER]
```

#### Keras 설치

1. Keras 폴더로 이동한 후 Keras를 설치합니다.
```
> cd /inspace/dl/Keras [ENTER]
> python setup.py install [ENTER] 
```
1. Theano와 Keras가 정상적으로 설치되었는 지 확인합니다.
```
> python [ENTER] 
>>> import Keras [ENTER]
```

1. "Using Theano backend." 또는 "Using Tensorflow backend.”라는 문구가 출력되면 정상적으로 설치된 것입니다. 
    
#### 오류 대처
1. pydot, find_graphviz() 관련 에러가 발생 시 Keras, Theano 재설치합니다.
```
> pip uninstall keras theano [ENTER]
> pip install --no-index --find-links=./packages keras theano [ENTER]
```

---

### 결론

차차 다양한 환경과 원하시는 버전을 설치할 수 있도록 정리해볼 예정입니다. 필요하신 환경 및 버전을 댓글로 달아주세요. 같은 환경에서도 동일하게 설치가 안될 수 있으니 오류 사항을 공유해주세요.

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/lecture/)
* 이전 : [딥러닝 이야기/학습인자와 데이터셋 이야기](https://tykimos.github.i.io017/03/25/Dataset_and_Fit_Talk/)
* 다음 : [딥러닝 모델 이야기/다층 퍼셉트론 레이어 이야기](https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/)
