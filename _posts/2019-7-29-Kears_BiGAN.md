---
layout: post
title:  "케라스 BiGAN"
author: 김태영
date:   2019-07-29 10:30:00
categories: seminar
comments: true
image: http://tykimos.github.io/warehouse/2019-7-29-Kears_BiGAN_title1.png
---

얼마 전 DeepMind에서 BigBiGAN(BigGAN + BiGAN) 모델이 발표되어서 이슈되고 있습니다. 이 중 케라스 기반의 BiGAN 깃헙을 프랑소와 쏠레님이 트윗해주셔서 소개드립니다.

보통 GAN은 노이즈를 Generator에 입력한 후 Generator가 생성한 이미지와 실제 이미지를 Discriminator가 분류하는 방면, BiGAN은 이미지로부터 뽑아낸 노이즈와 Generator로 생성한 이미지를 같이 Discrimiator에 입력합니다. 여기서 핵심은 "이미지로부터 뽑아낸 노이즈"라는 것인데, 이미지를 인코딩한 잠재벡터라고 보시면 됩니다.

![img](http://tykimos.github.io/warehouse/2019-7-29-Kears_BiGAN_model.png)
(출처: Adversarial Feature Learning, Jeff Donahue, [https://arxiv.org/abs/1605.09782](https://arxiv.org/abs/1605.09782))

이렇게 이미지로부터 잠재벡터를 잘 뽑아내는 네트워크를 학습시켰다면, 이미지간 유사도를 측정해볼 수 있겠죠? 즉 이미지를 입력하면, 잠재벡터를 뽑아내고, 가지고 있는 이미지들에 대해서 뽑아낸 잠재벡터와 가장 유사한 것을 골라내는 식입니다. (혹은 만들거나 말이죠)

풍경 사진으로부터 유사도를 뽑아내는 BiGAN 케라스 코드를 통해서 한 번 살펴보시죠~

![img](http://tykimos.github.io/warehouse/2019-7-29-Kears_BiGAN_title1.png)

가장 왼쪽에 있는 사진이 모델에 질문을 던진 사진이고, 오른쪽 두번째부터는 데이터셋에서 가장 유사한 순서대로 검출된 이미지입니다. 그 밖에,
* BiGAN의 특징 공간 내에서 클러스터링을 하거나
* 실제 이미지와 유사한 이미지를 만들어내는 등
다양한 활용이 소개되어 있네요.

### 소스코드
---
* Matthew님(manicman1999)의 깃헙: 
[https://github.com/manicman1999/Keras-BiGAN](https://github.com/manicman1999/Keras-BiGAN)

![img](http://tykimos.github.io/warehouse/2019-7-29-Kears_BiGAN_ref.png)

파일 구조는 다음과 같습니다. 

* bigan.py: BiGAN 네트워크를 정의하고 학습하는 코드입니다.
* guess.py: BiGAN 특징 공간의 Inverse Distance Weighting 유사도를 이용하여 게임을 해보는 코드입니다.
* idw.py: Inverse Distance Weighting을 계산하는 코드로 거리가 가까울수록(유사도가 높을 수록) 높은 값을 반환하는 함수입니다.

