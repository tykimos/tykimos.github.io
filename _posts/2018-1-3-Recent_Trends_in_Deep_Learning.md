---
layout: post
title:  "[세미나] 딥러닝 최신동향"
author: 최성준님 발표, 김태영 작성
date:   2018-01-04 23:00:00
categories: seminar
comments: true
image: http://tykimos.github.io/warehouse/2018-1-4-Recent_Trends_in_Deep_Learning_1.png
---
저희회사 사내세미나로 Google Developer Expert이시고 서울대학교 사이버물리시스템 연구실 박사과정에 계신 최성준님이 '딥러닝 최신 동향'이란 주제로 발표하셨습니다. 열정적으로 강의해주신 최성준님께 감사드리고, 장시간 동안 끝까지 관심있게 들어주시고 좋은 질문 많이해주신 참석자분들에게 감사드립니다.

![img](http://tykimos.github.io/warehouse/2018-1-4-Recent_Trends_in_Deep_Learning_1.png)

---

### 개요

딥러닝 분야에서 다루고 있는 연구 주제에 대해서 간략히 살펴보고, 본 세미나에서 다룰 네 가지 주제(Generative Model, Domain Adaptation, Meta Learning, Uncertainty in Deep Learning)를 소개합니다.

[intro.pdf](https://drive.google.com/open?id=1nDe8Kp3-1UM51lgSQBpnLBq0wSe3OLkh)

---

### Generative Model 

VAE와 GAN의 기본과 여러가지 GAN에 대해서 살펴봅니다. GAN의 발전속도와 활용성을 보니 신기할 따름이네요. 발표에서 다룬 논문들은 다음과 같습니다.

* Generative Adversarial Network (GAN), Goodfellow et al. (2014)
* Variational Autoencoder (VAE), Kingma (2017)
* DCGAN, Radford et al. (2016)
* Info GAN, Chen et al. (2016)
* Text2Image, Reed et al. (2016)
* Pix2pix, Isola et al. (2017)
* PatchGAN, Li and Wand (2016)
* Generative Adversarial What-Where Networks (GAWWN), Reed et al. (2016)
* Puzzle-GAN, Lee et al. (2017)
* Domain Transfer Network, Taigman et al. (2016)
* DiscoGAN, Kim et al. (2017)
* CycleGAN, Zhu et al. (2017)
* StarGAN, Choi et al. (2017)
* Progressive GAN, Karras et al. (2017)

[generative-model.pdf](https://drive.google.com/open?id=1AYPHYxQ44IOsA9C49aV_odZ2IorjAleK)

---

### Domain Adaptation

데이터를 비교적 얻기 쉬운 도메인에서 학습한 모델을 어떻게 우리가 적용하고자 하는 도메인에서 활용해볼 수 있을까에 대한 연구 소개 입니다. 위닝일레븐 잘 하시는 분이 진짜 축구를 잘 할 수 있다는... 발표에서 다룬 논문들은 다음과 같습니다.

* Analysis for domain adaptation, Ben-David et al. (2006)
* Domain Adversarial Neural Network (DANN), Ganin et al. (2016)
* Domain Separation Network (DSN), Bousmalis et al. (2016)
* Coupled Generative Adversarial Network (CoGAN), Liu & Tuzel (2016)
* Adversarial Discriminative Domain Adaption, Tzeng et al. (2017)
* Addressing Appearance Change in Outdoor Robotics with Adversarial Domain Adaption, Wulfmeier et al. (2017)
* Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks, Bousmalis et al. (2017)
* Associative Domain Adaptation, Haeusser et al. (2017)

[domain-adaptation.pdf](https://drive.google.com/open?id=1ZxCcfOBOWnfcOoEiuEWkKd7-iekr-Bfz)

---

### Meta Learning 

배우는 방법을 배우는 모델인데, 다소 생소할 수 있지만, 우리도 어떤 과목을 공부하지만 공부 잘하는 법 자체도 공부하기도 했었죠? 바로 그것입니다. 자동화된 딥러닝 모델로 보시면 될 듯 합니다. 이게 발전되면 더이상 딥러닝 모델을 만들 필요가 없습니다. 딥러닝 모델이 문제를 보고 적당한 딥러닝 모델을 만들테니깐요.

* Learning to Learn without Gradient Descent by Gradient Descent, Chen (2017)
* Siamese Neural Networks for One-shot Image Recognition, Koch et al. (2015)
* Deep Metric Learning using Triplet Network, Hoffer & Ailon (2015)
* Meta-Learning with Memory-Augmented Neural Networks, Santoro et al. (2016)
* Metching Networks for One Shot Learning, Vinyals et al. (2017)
* Prototypical Networks for Few-shot Learning, Snell et al. (2017)
* Low-shot Visual Recognition by Shrinking and Hallucinating Features, Hariharan & Girshick (2017)
* Model-Agnostic Meta-Learning, Finn et al. (2017)

[meta-learning.pdf](https://drive.google.com/open?id=1Ts3FeLDU32vsg0BG8B12FRaYcNB-uWKq)

---

### Uncertainty in Deep Learning

딥러닝 모델은 4지선다문제에서 찍기만 할 뿐입니다. 모르겠다고 고개를 흔드는 법은 없죠. 하지만 실무에서는 딥러닝에서 불확실하다는 것은 불확실하다고 알리고 사람에게 넘기길 원할 때가 많습니다. 하지만 딥러닝 모델에서 불확실성을 알기란 쉽지 않습니다. 이 불확실성에 대한 연구를 소개합니다. 발표에서 다룬 논문들은 다음과 같습니다.

* Uncertainty in Deep Learning, Gal (2016)
* Representing Inferential Uncertainty in Deep Neural Networks Through Sampling, McClure & Kriegeskorte (2017)
* Uncertainty-Aware Reinforcement Learning from Collision Avoidance, Khan et al. (2016)
* Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, Lakshminarayanan et al. (2017)
* What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, Kendal & Gal (2017)
* Uncertainty-Aware Learning from Demonstration Using Mixture Density Networks with Sampling-Free Variance Modeling, Choi et al. (2017)
* Bayesian Uncertainty Estimation for Batch Normalized Deep Networks, Anonymous (2018)

[uncertainty-in-deep-learning.pdf](https://drive.google.com/open?id=13hSLdWeCjGKj02At69c0qmgOCZ65LGdi)

---

### Interpretable Deep Learning

딥러닝 모델이 높은 성능을 보이고 있지만 그 속은 블랙박스라고 여겨져왔습니다. 딥러닝 분야 연구자들인 딥러닝 모델을 해석하고자 하는 노력을 해왔고, 조금씩 사람이 이해할 수 있는 방안에 대해서 결과나 나오기 시작했습니다. 이번 발표에서는 다루지 않았지만 발표자로를 추가로 공유해주셨고 다음 논문들이 포함되어 있습니다.

* Visualizing and Understanding Convolutional Networks, Zeiler & Fergus (2013)
* Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, Simonyan et al. (2014)
* Learning Deep Features for Discriminative Localization, Zhou et al. (2016)
* Grad-CAM, Selvaraju et al. (2017)
* Interpretable Explanations of Black Boxes by Meaningful Perturbation, Fong & Vedaldi (2017)
* Learning Important Features Through Propagating Activation Difference, Shrikumar et al. (2017)

[interpretable-deep-learning.pdf](https://drive.google.com/open?id=1z8-mSWFADJOiVUcAlw8s2qqLw_Xgo6Yj)

---

### 같이 보기

* [최성준님 블로그](http://enginius.tistory.com/) : 관련 주제로 더 자세히 알고 싶으신 분이나 이후 최신 연구 내용에 대해서 궁금하신 분들은 최성준님 블로그에서 확인하세요~
* [강좌 목차](https://tykimos.github.io/lecture/)
