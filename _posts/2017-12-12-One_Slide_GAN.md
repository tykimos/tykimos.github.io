---
layout: post
title:  "한 장 간"
author: 김태영
date:   2017-12-12 01:00:00
categories: Study
comments: true
image: https://tykimos.github.com/warehouse/2017-12-12-One_Slide_GAN_title.png
---
제 마음대로 '네트워크'와 '모델'을 구분해서 개념 정리를 해본 것을 그림으로 표현해봤습니다.

![img](https://tykimos.github.com/warehouse/2017-12-12-One_Slide_GAN_title.png)

이 그림을 그리게 된 계기가 있는데요. 케라스의 GAN 코드를 보다가 엄청 헷갈리는 부분이 있었는데 이것 때문에 개념삽질 좀 했습니다. 그 부분은 바로 "discriminator.trainable = False" 이것입니다. 일단 대부분의 GAN 코드는 다음과 같은 구성을 가지고 있습니다.


```python
# generator 생성
generator = Sequential()
generator.add(...)
generator.compile(...)

# discriminator 생성
discriminator = Sequential()
discriminator.add(...)
discriminator.compile(...)

# adversarial network 생성
discriminator.trainable = False ## << 바로 이부분
gan_in = Input(shape=(randomDim,))
gan_out = discriminator(generator(gan_in))
gan = Model(inputs=gan_in, outputs=gan_out)
gan.compile(...)
```

discriminator 학습 시킬 때는 참/거짓 데이터를 주고 가중치를 업데이트를 해야되지만, 이 discriminator가 gan 안에서 generator와 같이 학습할 때는 가중치가 고정되어 있어야 합니다. 그래서 gan 모델을 생성하기 전에 discriminator.trainable = False으로 설정하긴 했는데... 여기서 헷갈리기 시작했습니다.
- discriminator.trainable = False 으로 하면 gan에서는 고정되겠지만 discriminator을 학습할 때도 가중치가 고정되는 것이 아니야?
- 앞에서 생성한 discriminator과 gan에 삽입할 때의 discriminator는 다른 객체인가?

등등으로 생각을 했었는데, 알고보니 compile() 함수가 호출될 때 trainable 속성이 모델에 적용되더라구요. 즉 다음과 같습니다.

- discriminator을 생성한 뒤 compile() 하면 trainable = True로 컴파일 됨
- discriminator.trainable = False으로 적용하면 일단 trainable 속성만 비활성화된 상태임
- gan 모델에 discriminator가 삽입됨
- gan.compile() 하면 gan 모델 안에서 discriminator의 가중치가 업데이트 되지 않음
- gan.compile()과 discriminator.compile()은 별개이고, discriminator.compile()가 다시 호출 되지 않았으므로, discriminator 모델에서의 trainable 속성은 True임
- 여기서 하나 알 수 있는 것은 discriminator이라는 네트워크는 discriminator 모델과 gan 모델에 둘 다 사용되고 가중치도 공유되나 discriminator 모델에서는 가중치 갱신이 일어나고, gan 모델에서는 가중치 갱신이 일어나지 않음
- gan 모델에서의 discriminator 네트워크는 단순 가중치를 가진 네트워크로만 받아들이고 discriminator 모델에 적용된 compile()은 아무 영향을 주지 않음. 즉 gan 모델은 따로 complie()을 해야 함

배성호교수님의 '간은 로스일 뿐이야'라는 말씀을 이제야 이해한 듯 합니다.

---

### 같이 보기

* [강좌 목차](https://tykimos.github.io/lecture/)
