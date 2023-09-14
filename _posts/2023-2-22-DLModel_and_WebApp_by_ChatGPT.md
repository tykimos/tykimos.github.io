---
layout: post
title:  "챗GPT로 딥러닝 모델에서 웹앱개발까지"
author: 김태영
date:   2023-2-21 00:00:00
categories: tech
comments: true
image: https://cdn.aifactory.space/images/20230222103438_kRbn.png
---
이번 사례는 챗GPT로 특정 태스크에 대한 인공지능 모델 개발부터 웹앱서비스 개발까지 요청해보도록 하겠습니다. 전체 시나리오는 다음과 같습니다. 

*   태스크를 정의하고 이에 대한 파이썬 모델링을 요청합니다.
*   챗GPT가 모델 코드를 제작해주면, 이를 코랩을 활용해서 모델을 학습시킵니다.
*   학습한 모델로 서비스 만들기 위해 웹앱 코드를 요청합니다.
*   웹앱 코드를 받으면 허깅페이스의 Space에 등록하여 웹앱을 만듭니다.

전체 시리즈는 타이키모스 블로그에서 보실 수 있습니다. >> [https://tykimos.github.io/2023/02/21/DLModel\_and\_WebApp\_by\_ChatGPT/](https://tykimos.github.io/2023/02/21/DLModel_and_WebApp_by_ChatGPT/)

![](https://cdn.aifactory.space/images/20230222103438_kRbn.png)

프롬프트와 답변
--------

이번 시나리오에 사용된 프롬프트 및 답변을 가지고 왔습니다. 팁이나 추가 설명이 필요할 경우 아래 서식을 이용했습니다.

> 팁 혹은 추가설명

![](https://cdn.aifactory.space/images/20230222204319_LmVO.png)

소나무재선충병 감염 여부 분류를 하기 위한 파이썬 기반의 딥러닝 모델 소스 코드를 작성해줘. 입력은 이미지고 출력은 감염여부 확률을 반환해줘.

> 팁: 태스크, 프로그램 종류 등에 대해서 명확하게 작성합니다. 입력과 출력에 대한 스펙도 상세하기 지정해주면 좋습니다.

### ![](https://cdn.aifactory.space/images/20230222204335_FIWw.png)

코드를 작성하기 위해서는 딥러닝 라이브러리인 TensorFlow나 PyTorch 등이 필요합니다. 여기서는 TensorFlow를 사용하여 모델을 구현해 보겠습니다.

> 설명: 텐서플로우으로 작성해주네요. 파이토치 사용자라면 프롬프트에 파이토치라고 명시해주면 되겠죠?

먼저, 필요한 라이브러리들을 import 하겠습니다.

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    

다음으로, 모델을 정의할 함수를 작성합니다.

    def create_model(input_shape):
        inputs = keras.Input(shape=input_shape)
        # 이미지 전처리
        x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(128, 3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        # 밀집층
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    

이 모델은 입력으로 이미지를 받아들이고, 감염 여부에 대한 이진 분류 문제를 해결합니다. 모델은 3개의 합성곱 층, 3개의 풀링 층, 1개의 밀집 층, 그리고 출력층으로 구성되어 있습니다.

데이터셋을 불러오고 모델을 학습시키기 위한 코드는 다음과 같습니다.

    # 데이터셋 불러오기
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "/path/to/dataset/",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(256, 256),
        batch_size=32,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "/path/to/dataset/",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(256, 256),
        batch_size=32,
    )
    
    # 모델 생성
    model = create_model(input_shape=(256, 256, 3))
    model.summary()
    
    # 모델 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    # 모델 학습
    epochs = 10
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    

위 코드에서는 **image\_dataset\_from\_directory** 함수를 사용하여 이미지 데이터셋을 불러옵니다. 모델은 **create\_model** 함수로 생성하고, 컴파일합니다. 마지막으로 **fit** 함수를 호출하여 모델을 학습시킵니다.

> 설명: 데이터셋 준비는 정상과 간염 폴더를 만든 후 이미지를 해당 폴더에 넣어주기만 하면 됩니다.

![](https://cdn.aifactory.space/images/20230222204319_LmVO.png)

만든 소나무재선충병 감염 여부 분류 모델을 gradio로 웹 서비스를 할 수 있는 코드도 작성해줘.

> 팁: 허깅페이스에서 많이 사용하는 그라디오 패키지로 요청했습니다. 과거 데이터로 학습되어 있으므로 최신 프레임워크는 모를 수 있으니 이점 참고하세요.

### ![](https://cdn.aifactory.space/images/20230222204335_FIWw.png)

Sure, Gradio를 사용하여 모델을 웹 서비스로 제공하는 코드를 작성해 드리겠습니다. 아래 코드는 이미지를 업로드하여 감염 여부를 예측하는 간단한 예제입니다.

    import gradio as gr
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    import requests
    from io import BytesIO
    
    # 모델 로드
    model = tf.keras.models.load_model("your_model_path")
    
    # 이미지를 전처리하는 함수
    def preprocess(image):
        img = image.resize((256, 256))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape((1,) + img.shape)
        return img
    
    # 예측 함수
    def predict_image(img):
        img = preprocess(img)
        prediction = model.predict(img)[0][0]
        return {'정상': 1-prediction, '감염': prediction}
    
    # 인터페이스 구성
    imagein = gr.inputs.Image()
    label = gr.outputs.Label(num_top_classes=2)
    
    # 예측 인터페이스 실행
    gr.Interface(fn=predict_image, inputs=imagein, outputs=label, 
                 title='소나무재선충병 감염 여부 예측',
                 description='산림청에서 제공하는 소나무재선충병 데이터셋을 이용한 딥러닝 모델을 사용하여 감염 여부를 예측합니다.').launch()
    

이제 위와 같은 코드를 실행하면 Gradio 인터페이스가 실행됩니다. 웹 인터페이스에서 이미지를 업로드하면 모델이 감염 여부를 예측하고 예측 결과를 표시합니다.

따라해보기
-----

만들어준 코드를 바로 동작시켜보니 에러가 발생되었습니다. 라이브러리가 지속적으로 업데이트가 되다보니 버전 차이 문제로 발생한 것 같습니다. 챗GPT가 학습하기 이전의 업데이트이라면 에러에 대해서 말해주겠지만, 저는 그냥 웹에서 찾아서 두 군데 조금 손을 봤습니다. 

*   float로 명시적으로 형변환시키기
*   gr.inputs.Image 타입을 “pil”로 지정하기

![](https://cdn.aifactory.space/images/20230222145121_IaAH.png)

아래가 완성된 웹앱입니다. 이미지를 입력할 수 있는 인터페이스와 결과를 확률값으로 나오는 인터페이스가 생성됩니다. 

> 실제 이미지 데이터로 학습한 것은 아니고, 인터페이스만 맞춘 모델이기 때문에 결과는 보장못합니다.

![](https://lh3.googleusercontent.com/KoiLcTgxKf7Wv760Lh7g02Coxgh9vUA9t--nnsWNFcu_5Pqkxy3T0osydxU0jOJIvyi5bDz7Ccv7foAfJ9e1tfZt5aVgau83EGS5PFuGftjdt3byQ4vur-rOBYwueiYCtVYbzdChNh8Y_ckn8RP8L6-Ec6vrUd1DwRCpSFEbrvTcJDBn5tiByzrEXkAFPGjA=s2048)

결론
--

챗GPT를 통해 딥러닝 모델 개발과 웹앱 서비스까지 소스코드를 작성해봤습니다. 명확하고 상세한 프롬프트일수록 원하는 결과 소스코드를 받을 수 있을 것 같습니다.
