---
layout: post
title:  "DeepBrick for Keras (케라스를 위한 딥브릭)"
author: 김태영 (Taeyoung Kim)
date:   2017-09-10 16:00:00
categories: DeepBrick
comments: true
image: http://tykimos.github.io/warehouse/DeepBrick/title.png
---
The Keras is a high-level API for deep learning model. The API is very intuitive and similar to building bricks. So, I have started the DeepBrick Project to help you understand Keras's layers and models.

딥러닝과 케라스를 공부하면서 느낀 점은 층을 쌓고 모델을 만들고 하는 과정들이 블록 쌓는 것과 비슷한 느낌을 많이 받았고, 실제로 딥러닝 모델을 설명할 때 블록 그림을 많이 이용하기도 했습니다. 그러다가 (실제 혹은 웹에서) 블록을 쌓으면 딥러닝 모델까지 자동으로 만들 수 있겠다는 생각이 들었습니다. 그래서 딥브릭(DeepBrick)이란 이름으로 프로젝트를 진행해볼까 합니다.

---
### Bricks

There are bricks supported by DeepBrick. 

#### Dataset

|Brick|Name|Description|
|:-:|:-:|:-|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dataset_Vector_s.png)|Input data, Labels|Input data and labels are encoded as vector.<br>1차원의 입력 데이터 및 라벨입니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dataset2D_s.png)|2D Input data|Input data are encoded as 2D vector.<br>2차원의 입력 데이터입니다.<br><br>In case of imagery, the dimention consists of sample, width, height and channel.<br>주로 영상 데이터를 의미하며 샘플수, 너비, 높이, 채널수로 구성됩니다.|

#### Layers

|Brick|Name|Description|
|:-:|:-:|:-|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dense_s.png)|Dense|Regular densely-connected neual network layer.<br>모든 입력 뉴런과 출력 뉴런을 연결하는 전결합층입니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Embedding_s.png)|Embedding|Turns positive integer representations of words into a word embedding.<br>단어를 의미론적 기하공간에 매핑할 수 있도록 벡터화시킵니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Conv1D_s.png)|Conv1D|Extracts local features using 1D filters.<br>필터를 이용하여 지역적인 특징을 추출합니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Conv2D_s.png)|Conv2D|Extracts local features of images using 2D filters.<br>필터를 이용하여 영상 특징을 추출합니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_GlobalMaxPooling1D_s.png)|GlobalMaxPooling1D|Returns the largest vector of several input vectors.<br>여러 개의 벡터 정보 중 가장 큰 벡터를 골라서 반환합니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_MaxPooling1D_s.png)|MaxPooling1D|Returns the largest vectors of specific range of input vectors.<br>입력벡터에서 특정 구간마다 값을 골라 벡터를 구성한 후 반환합니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_MaxPooling2D_s.png)|MaxPooling2D|Reduces to affect feature extaction by minor changes in the image.<br>영상에서의 사소한 변화가 특징 추출에 크게 영향을 미치지 않도록 합니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Flatten_s.png)|Flatten|Flattens the input.<br>2차원의 특징맵을 전결합층으로 전달하기 위해서 1차원 형식으로 바꿔줍니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_LSTM_s.png)|LSTM|Long-Short Term Memory unit, one of RNN layer.<br>Long-Short Term Memory unit의 약자로 순환 신경망 레이어 중 하나입니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dropout_1D_s.png)|Dropout|Excludes random input neurons (one-dimensional) at a specified rate during learning to prevent overfitting.<br>과적합을 방지하기 위해서 학습 시에 지정된 비율만큼 임의의 입력 뉴런(1차원)을 제외시킵니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Dropout_2D_s.png)|Dropout|Excludes random input neurons (two-dimensional) at a specified rate during learning to prevent overfitting.<br>과적합을 방지하기 위해서 학습 시에 지정된 비율만큼 임의의 입력 뉴런(2차원)을 제외시킵니다.|

#### Activation Functions

|Brick|Name|Description|
|:-:|:-:|:-|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_sigmoid_s.png)|sigmoid|Returns a value between 0 and 1.<br>활성화 함수로 입력되는 값을 0과 1사이의 값으로 출력시킵니다.<br><br>This is mainly used for the activation function of the output layer of the binary classification model it can be judged as positive if the output value is above a certain threshold value (for example, 0.5) or negative if it is below.<br>출력값이 특정 임계값(예를 들어 0.5) 이상이면 양성, 이하이면 음성이라고 판별할 수 있기 때문에 이진분류 모델의 출력층에 주로 사용됩니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_softmax_s.png)|softmax|Returns the probability value per class.<br>활성화 함수로 입력되는 값을 클래스별로 확률 값이 나오도록 출력시킵니다.<br><br>If all of these probabilities are added, it becomes 1.<br>이 확률값을 모두 더하면 1이 됩니다.<br><br>It is used mainly for the activation function of the output layer of a multi-class model, and the class with the highest probability value is the class classified by the model.<br>다중클래스 모델의 출력층에 주로 사용되며, 확률값이 가장 높은 클래스가 모델이 분류한 클래스입니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_tanh_s.png)|tanh|Returns a value between -1 and 1.<br>활성화 함수로 입력되는 값을 -1과 1사이의 값으로 출력시킵니다.<br><br>It is used for the activation function of LSTM layer.<br>LSTM의 출력 활성화 함수로 사용됩니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_Relu_s.png)|relu|It is mainly used of the activation funition of the hidden layer.<br>활성화 함수로 주로 은닉층에 사용됩니다.|
|![img](http://tykimos.github.io/warehouse/DeepBrick/Model_Recipe_Part_Activation_relu_2D_s.png)|relu|It is mainly used of the activation funition of the hidden layer such as Conv2D.<br>활성화 함수로 주로 Conv2D 은닉층에 사용됩니다.|

---
### Models

#### Numerical Input / Numerical Prediction - Perceptron NN Model <br>(수치입력 수치예측 퍼셉트론 신경망 모델)
    
![img](http://tykimos.github.io/warehouse/2017-8-12-Numerical_Prediction_Model_Recipe_1m.png)


```python
model = Sequential()
model.add(Dense(1, input_dim=1))
```

[more...](https://tykimos.github.io/2017/08/13/Numerical_Prediction_Model_Recipe/)

#### Numerical Input / Numerical Prediction - MLP NN Model<br>(수치입력 수치예측 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-12-Numerical_Prediction_Model_Recipe_2m.png)


```python
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(1))
```

[more...](https://tykimos.github.io/2017/08/13/Numerical_Prediction_Model_Recipe/)

#### Numerical Input / Binary Classification - Deep MLP NN Model<br>(수치입력 수치예측 깊은 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-12-Numerical_Prediction_Model_Recipe_3m.png)


```python
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
```

[more...](https://tykimos.github.io/2017/08/13/Numerical_Prediction_Model_Recipe/)

#### Numerical Input / Binary Classification - Perceptron NN Model<br>(수치입력 이진분류 퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-18-Numerical_Input_Binary_Classification_Model_Recipe_1m.png)


```python
model = Sequential()
model.add(Dense(1, input_dim=12, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/13/Numerical_Input_Binary_Classification_Model_Recipe/)

#### Numerical Input / Binary Classification - MLP NN Model<br>(수치입력 이진분류 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-18-Numerical_Input_Binary_Classification_Model_Recipe_2m.png)


```python
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/13/Numerical_Input_Binary_Classification_Model_Recipe/)

#### Numerical Input / Binary Classification - Deep MLP NN Model<br>(수치입력 이진분류 깊은 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-18-Numerical_Input_Binary_Classification_Model_Recipe_3m.png)


```python
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/13/Numerical_Input_Binary_Classification_Model_Recipe/)

#### Numerical Input / Multiclass Classification - Perceptron NN Model<br>(수치입력 다중클래스분류 퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-19-Numerical_Input_Multiclass_Classification_Model_Recipe_1m.png)


```python
model = Sequential()
model.add(Dense(10, input_dim=12, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Numerical Input / Multiclass Classification - MLP NN Model<br>(수치입력 다중클래스분류 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-19-Numerical_Input_Multiclass_Classification_Model_Recipe_2m.png)


```python
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/19/Numerical_Input_Multiclass_Classification_Model_Recipe/)

#### Numerical Input / Multiclass Classification - Deep MLP NN Model<br>(수치입력 다중클래스분류 깊은 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-19-Numerical_Input_Multiclass_Classification_Model_Recipe_3m.png)


```python
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/19/Numerical_Input_Multiclass_Classification_Model_Recipe/)

#### Image Input / Numerical Prediction - MLP NN Model<br>(영상입력 수치예측 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-20-Image_Input_Numerical_Prediction_Model_Recipe_1m.png)


```python
model = Sequential()
model.add(Dense(256, activation='relu', input_dim = width*height))
model.add(Dense(256, activation='relu'))
model.add(Dense(256))
model.add(Dense(1))
```

[more...](https://tykimos.github.io/2017/08/20/Image_Input_Numerical_Prediction_Model_Recipe/)

#### Image Input / Numerical Prediction - CNN Model<br>(영상입력 수치예측 컨볼루션 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-20-Image_Input_Numerical_Prediction_Model_Recipe_2m.png)


```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Image Input / Binary Classification - MLP NN Model<br>(영상입력 이진분류 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_0m.png)


```python
model = Sequential()
model.add(Dense(256, input_dim=width*height, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/18/Image_Input_Binary_Classification_Model_Recipe/)

#### Image Input / Binary Classification - CNN Model<br>(영상입력 이진분류 컨볼루션 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_1m.png)


```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/18/Image_Input_Binary_Classification_Model_Recipe/)

#### Image Input / Binary Classification - Deep CNN Model<br>(영상입력 이진분류 깊은 컨볼루션 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Binary_Classification_Model_Recipe_2m.png)


```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/18/Image_Input_Binary_Classification_Model_Recipe/)

#### Image Input / Multiclass Classification - MLP NN Model<br>(영상입력 다중클래스분류 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Multiclass_Classification_Model_Recipe_0m.png)


```python
model = Sequential()
model.add(Dense(256, input_dim=width*height, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Image Input / Multiclass Classification - CNN Model<br>(영상입력 다중클래스분류 컨볼루션 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Multiclass_Classification_Model_Recipe_1m.png)


```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/18/Image_Input_Multiclass_Classification_Model_Recipe/)

#### Image Input / Multiclass Classification - Deep CNN Model<br>(영상입력 다중클래스분류 깊은 컨볼루션 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-18-Image_Input_Multiclass_Classification_Model_Recipe_2m.png)


```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/18/Image_Input_Multiclass_Classification_Model_Recipe/)

#### Time-series Numerical Input / Numerical Prediction - MLP NN Model<br>(시계열수치입력 수치예측 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_1m.png)


```python
model = Sequential()
model.add(Dense(32,input_dim=40,activation="relu"))
model.add(Dropout(0.3))
for i in range(2):
    model.add(Dense(32,activation="relu"))
    model.add(Dropout(0.3))
model.add(Dense(1))
```

[more...](https://tykimos.github.io/2017/09/09/Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe/)

#### Time-series Numerical Input / Numerical Prediction - RNN Model<br>(시계열수치입력 수치예측 순환신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_2m.png)


```python
model = Sequential()
model.add(LSTM(32, input_shape=(None, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))
```

[more...](https://tykimos.github.io/2017/09/09/Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe/)

#### Time-series Numerical Input / Numerical Prediction - Stateful RNN Model<br>(시계열수치입력 수치예측 상태유지 순환신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_3m.png)


```python
model = Sequential()
model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))
```

[more...](https://tykimos.github.io/2017/09/09/Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe/)

#### Time-series Numerical Input / Numerical Prediction - Stateful Stack RNN Model<br>(시계열수치입력 수치예측 상태유지 스택 순환신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-9-9-Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe_4m.png)


```python
model = Sequential()
for i in range(2):
    model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))
```

[more...](https://tykimos.github.io/2017/09/09/Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe/)

#### Text Input / Binary Classification - MLP NN Model<br>(문장입력 이진분류 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_1m.png)


```python
model = Sequential()
model.add(Embedding(20000, 128, input_length=200))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Text Input / Binary Classification - RNN Model<br>(문장입력 이진분류 순환신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_2m.png)


```python
model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Text Input / Binary Classification - CNN Model<br>(문장입력 이진분류 컨볼루션 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_3m.png)


```python
model = Sequential()
model.add(Embedding(20000, 128, input_length=200))
model.add(Dropout(0.2))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Text Input / Binary Classification - RNN & CNN Model<br>(문장입력 이진분류 순환 컨볼루션 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Binary_Classification_Model_Recipe_4m.png)


```python
model = Sequential()
model.add(Embedding(20000, 128, input_length=200))
model.add(Dropout(0.2))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Text Input / Multiclass Classification - MLP NN Model<br>(문장입력 다중클래스분류 다층퍼셉트론 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Multiclass_Classification_Model_Recipe_1m.png)


```python
model = Sequential()
model.add(Embedding(15000, 128, input_length=120))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(46, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Text Input / Multiclass Classification - RNN Model<br>(문장입력 다중클래스분류 순환신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Multiclass_Classification_Model_Recipe_2m.png)


```python
model = Sequential()
model.add(Embedding(15000, 128))
model.add(LSTM(128))
model.add(Dense(46, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Text Input / Multiclass Classification - CNN Model<br>(문장입력 다중클래스분류 컨볼루션 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Multiclass_Classification_Model_Recipe_3m.png)


```python
model = Sequential()
model.add(Embedding(15000, 128, input_length=120))
model.add(Dropout(0.2))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(46, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)

#### Text Input / Multiclass Classification - RNN & CNN Model<br>(문장입력 다중클래스분류 순환 컨볼루션 신경망 모델)

![img](http://tykimos.github.io/warehouse/2017-8-17-Text_Input_Multiclass_Classification_Model_Recipe_4m.png)


```python
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Dropout(0.2))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(46, activation='softmax'))
```

[more...](https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/)
