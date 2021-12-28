---
layout: post
title:  "티처블머신을 사용한 게더타운 크리스마스 미로찾기"
author: 김태영
date:   2021-12-25 12:00:00
categories: tech
comments: true
image: https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_play.gif
---
게더타운은 메타버스 화상회의 플랫폼으로 맵을 만들어서, 해당 맵에서 아바타를 통해서 여러가지 활동 및 소통을 할 수 있는 공간입니다. 게더타운을 놀이공간으로 활용하는 사례도 많은 데, 그 중 하나로 미로 찾기 게임을 할 수도 있습니다.

게더타운에서의 미로찾기는 자동 길찾기 알고리즘을 통해서 목적지를 클릭하면 알아서 찾아주기 때문에 직접 마우스 컨트롤을 통해서 미로 찾기를 하는 것은 권장하지 않습니다. 오프라인에서 나무 등으로 만들어진 미로 찾기 처럼 게더타운에서도 비슷하게 구현한 뒤, 티처블 머신을 이용해서 가만이 있기, 위/아래, 왼쪽/오른쪽 행동을 정의하고, 이러한 행동을 통해서 미로찾기 게임을 해봅니다.

<iframe width="100%" height="400" src="https://www.youtube.com/embed/06PkLEtzNDM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### 특징
* 온라인에서 여러 사람이 접속하여 서로 경기를 할 수 있습니다.
* 원하는 행동으로 학습할 수 있기 때문에 몸이 불편한 사람도 쉽게 게임에 참여할 수 있습니다. 
* 실제 오프라인 공간에서 미로찾기 게임하듯이 즐길 수 있습니다. 

### 사전 준비

#### 게더타운 미로 제작

![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_gathertown.jpg)

#### 티처블 머신 포즈 정의

![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel.jpg)

디폴트로 제공하는 wef1yJup2 모델은 다음과 행동으로 학습되어 있습니다. 

|stay|up|down|right|left|
|---|---|---|---|---|
|![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_stay.jpg)|![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_up.jpg)|![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_down.jpg)|![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_right.jpg)|![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_left.jpg)|

#### 사용법

1. 미리 만들어놓은 미로 게더타운(https://gather.town/app/4BDf97jEEOpc6l8c/christmas%20maze)에 접속합니다.
1. gihtub 주소(https://github.com/teachableverse/christmas-maze-gathertown-teachablemachine)에서 소스코드를 다운로드 받습니다. 
1. 파이썬 프로그램인 app.py를 실행시킵니다. >> python app.py
1. 크롬 브라우저 주소창에 127.0.0.1:5001를 입력합니다.
1. Load 버튼을 클릭하여 이미 행동을 학습시킨 티처블머신 모델(wef1yJup2)을 로딩합니다. 만약 직접 행동을 정의한 모델이 있다면, 해당 모델의 아이디를 1. 입력한 후 Load 버튼을 클릭합니다.
1. 키 메시지가 게더타운에 입력될 수 있도록 미로 게더타운 창을 클릭합니다.
1. 미리 정의된 행동으로 게더타운의 아바타를 제어합니다.

#### 나만의 행동을 정의하는 법

Teachable Machine의 Pose Project로 머신러닝 모델을 아래 순서로 학습십니다. 

1. https://teachablemachine.withgoogle.com/ 에 접속하여 Pose Project을 생성합니다.

![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_training_0.jpg)

2. stay-up-down-right-left 순으로 class를 추가하여 데이터 샘플을 수집합니다.

![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_training_1.jpg)

3. Train Model을 클릭하여 수집한 데이터셋으로 모델을 학습시킵니다.

![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_training_2.jpg)

4. 학습한 모델이 정상적으로 작동되는 지 확인합니다.

![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_training_3.jpg)

5. [Upload my model] 버튼을 클릭하여 학습한 모델을 클라우드에 업로드합니다.

![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_training_4.jpg)

6. 업로드 된 모델 경로 및 id를 확인합니다. 

![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_training_5.jpg)

7. 확인한 id를 “http://127.0.0.1:5001/” 페이지의 입력 폼에 입력한 후 [Load] 클릭하면 학습한 모델이 적용됩니다.

![img](https://tykimos.github.io/warehouse/2021-12-25-christmas-maze-gathertown-teachablemachine_tmmodel_training_6.jpg)