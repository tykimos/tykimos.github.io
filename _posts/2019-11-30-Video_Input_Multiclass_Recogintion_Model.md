---
layout: post
title:  "동영상입력 행동분류 모델 튜토리얼 소개"
author: 김태영
date:   2019-11-30 03:00:00
categories: Lecture
comments: true
image: http://tykimos.github.io/warehouse/2019-11-30-Video_Input_Multiclass_Recogintion_Model_title.png
---
딥러닝 기술이 이미지 분류에서는 정점은 찍었고, 여러 이미지 처리 분야로 확장되어 있는 가운데, 동영상에 대한 시도도 많이 늘어나고 있습니다. 아래 링크는 딥러닝기반 동영상을 다루고 싶은 분들에게 좋은 입문 자료가 될 것 같습니다.

![img](http://tykimos.github.io/warehouse/2019-11-30-Video_Input_Multiclass_Recogintion_Model_title.png)

[https://www.pyimagesearch.com/2019/11/25/human-activity-recognition-with-opencv-and-deep-learning/](https://www.pyimagesearch.com/2019/11/25/human-activity-recognition-with-opencv-and-deep-learning/)

원글의 콘텐츠를 보호하기 위해서 글로만 적어봤습니다.

---
### 문제정의

동영상을 틀어주면서 좌측 상단에 딥러닝이 판단한 행동을 표시해줍니다. 손 씻고 있는 영상에서 보면 "손을 씻고있다"라고 글로 표시됩니다. 

동영상이라고 하면 시간축이 하나 더 늘어난 자료 형태라고 할 수 있습니다. 이미지가 2D라면 동영상은 시간축이 포함된 3D인 것이죠. 이미지 분류를 시작으로 많은 어플리케이션이 나왔듯이 동영상 분류를 시작으로 또 많은 성공적인 어플리케이션이 나올 것 같은 예감입니다. 

---
### 데이터셋

학습에 사용한 데이터셋은 "Kinetics"이라고 합니다. 400개의 클래스가 있고, 각 클래스마다 400개 이상 동영상이 있으며, 전체 30만개의 정도라고 합니다. (우와~)
* [관련 논문](https://arxiv.org/abs/1705.06950)

튜터리얼에서는 Kinetics로 학습한 모델을 사용합니다. 튜터리얼에서 학습까지 하면 너무나 오래 걸리겠죠?

---
### 구현

몇가지 구현 사항을 요약해봤습니다. 
* 기 학습한 모델을 사용하기에 모데 가중치 파일을 제공하고 있습니다. 
* 영상처리 라이브러리인 OpenCV 안에 있는 딥러닝 모듈(DNN)을 사용하네요. (참고: https://github.com/opencv/opencv/blob/master/samples/dnn/action_recognition.py)
* 백본 모델은 ResNet-34를 사용했네요. (파이토치 버젼 참고: https://github.com/kenshohara/video-classification-3d-cnn-pytorch)
* 입문하시는 분들은 자료 형태 맞추기가 쉽지 않은데, 이미지에 익숙하신 분이라면 "시간 차원"이 늘어난 것만 유의하세요. 즉 몇 "프레임"인가를 정의해줘야 합니다. 

고려해야할 재미있는 컨셉은 "프레임 다루기"입니다.
* 특정 프레임 단위로 잘라서 사용하는 방법 (혹은 겹쳐서 잘라도 무방할 것 같습니다.)
* 시간을 조금씩 이동시키면서 프레임을 자릅니다. 중복되는 프레임이 엄청 많겠죠?

사용하는 어플리케이션에 따라 다르겠지만 어떻게 프레임을 다뤄야할 것인가에 대해서 세심하게 신경써야 될 것 같아요.

---
### 맺음말

이 게시물 초반에 동영상 인식에 대한 예시(식당에서)가 나오는 부분이 있는데, 조금 섬뜩하네요. 
* 신입직원이 지정된 절차대로 제대로 일을 하는 지 체크
* 화장실 다녀온 뒤 손씻는 지 확인
* 손님이 제대로 서빙을 받고 있는지

조금 더 깊게 알고싶으신 분은 아래 논문(2018, CVPR)을 읽어보시라고 하네요. 
* [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/abs/1711.09577)
