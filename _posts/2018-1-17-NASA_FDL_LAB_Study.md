---
layout: post
title:  "NASA FDL 2017 연구 소개"
author: 박천용
date:   2018-01-27 04:00:00
categories: Study
comments: true
image: http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_title.jpeg
---
저희회사((주)인스페이스)에서 충남대학교 백마인턴쉽 과정으로 함께 하게된 박천용 인턴님의 첫과제로 조사한 내용을 공유드립니다. 주제는 NASA Frontier Development Lab에서 진행되고 있는 연구 소개입니다.

미국 NASA의 Frontier Development Lab (FDL)은 항공 기관인 Ames Research Center와 SETI에 의해 공동 운영 되고 있으며, 이들은 잠재적 위험성을 가지고 있는 소행성과 혜성으로부터 지구를 보호하는 방법을 연구하고자 인공지능을 이용하겠다고 발표하였습니다. 

* NASA FDL Homepage: [NASA FDL](http://frontierdevelopmentlab.org/)
* NASA FDL 팀의 발표영상과 프레젠테이션 자료: [NASA PPT](http://frontierdevelopmentlab.org/#/event)

지난 2014년에 매년 6월 30일을 국제 소행성 날 (International Asteroid Day)로 지정하고 Near Earth Objects(NEOs)로 부터 오는 잠재적 위협에 대한 연구결과를 발표하는 연내행사를 계획하는 것으로 만들어졌습니다. 6월 30일이 선택된 이유는 1908년에 일어난 러시에 시베리아 위치한 퉁구스카에 충돌체 사건을 기념하는 날이기 때문이었기 때문입니다. 연내 기념행사는 천체물리학자이자 Queen의 리드 기타리스트인 Brian May, 그리고 영화 제작자인 Grigorij Richters의 아이디어이었습니다. NASA FDL 팀은 NASA Frontier Development Lab 2017(FDL 2017)발표를 통해 딥러닝 기반의 다양한 연구성과를 알리게 되었습니다.

---

### 1. Solar storm prediction

#### 개요

인공지능을 사용하여 플레어를 탐지하고, 우주 비행 임무에 결정적인 태양활동 및 우주 기상 현상의 중요성을 이해합니다. 태양 자기장 complexity 분석을 수행하고, 태양 UV 이미지를 연결하기 위해 multiple CNN을 배치했습니다. 이 연구는 태양 플레어 예측의 신뢰성과 정확성을 향상시킬 수 있는 잠재력이 있음을 보여줍니다.

#### 플레어(Flare)란?

Flare란 태양 대기에서 발생하는, 수소폭탄 수천만 개에 해당하는 격렬한 폭발을 말합니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_flare.png)

태양의 **Flare**(이하 플레어)는 단파장의 전자기 방사선을 발생시킵니다. 이는 상층 대기의 이온화와 가열을 초래하고, GPS와 HF의 통신에 영향을 미칩니다.
 
#### FlareNet

어떻게 NOAA(기상위성)가 플레어를 예상할수 있을까요? 태양 흑점 형태와 지속성, 즉 태양이 변하지 않는 다는 것을 가정합니다. FDL팀은 전문가들 보다 최소한 한시간 빠르게 플레어의 위험을 예측하는 것을 목표로 하였습니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-storm-prediction-presentation-11-1024.jpg)
![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-storm-prediction-presentation-14-1024.jpg)

데이터셋으로는 태양의 **SDO*(Solar Dynamic Observatory)*/AIA*(Atmospheric Imaging Assembly)* Image**를 사용합니다. 그러나 AIA 데이터는 높은 동적 범위(High dynamic range)의 문제가 존재하기 때문에 Log Transform 을 하여 데이터를 전처리합니다. 전처리 과정을 거친 데이터를 아래의 CNN모델로 학습시킵니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-storm-prediction-presentation-44-1024.jpg)

#### 결론

* FlareNet은 상위 수준의 X 선 플럭스 활동을 생성 할 수 있습니다.
* FlareNet은 태양의 구조 뿐만 아니라 active regions의 중요성을 학습하였습니다.

---

### 2. Long Period Comets

#### 개요

천문학적인 유성우의 발견은 천년기에 지구 궤도를 가로지르는 긴 주기의 혜성의 존재를 암시합니다. 머신러닝과 딥러닝을 사용한 유성 분류의 자동화를 연구합니다. 유성우 관측에 기계 학습을 적용하여 오랜 기간 혜성 충돌에 대한 더 많은 경고를 제공합니다. 유성우 궤도는 예상 궤도를 따라 전용 검색을 가능하게 합니다.

#### CAMS

##### LSTM을 이용한 유성 판별

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_meteor_graph.png)
![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_non_meteor_graph.png)

각 개체의 시간에 따른 위치를 X,Y 좌표계로 나타내고, Intensity(밝기,빛의 모양 등)를 시간에 따라 graph로 나타내었습니다. 위 그림은 유성인 것과 유성이 아닌 것에 대한 비교 그래프입니다. 유성의 경우 규칙적인 값을 갖는 반면에, 유성이 아닌 경우 불규칙적인 값을 갖습니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-long-period-comets-final-presentation-15-1024.jpg)

위의 Tracklets(궤도)데이타를 입력으로 하는 LSTM 모델을 활용하여 유성을 판별합니다.

##### CNN을 이용한 유성 판별

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-long-period-comets-final-presentation-16-1024.jpg)

위 이미지와 같은 Label된 데이터를 바탕으로 <strong>*Convolution Neural Network*</strong>를 이용하여 유성 여부를 판별합니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-long-period-comets-final-presentation-17-1024.jpg)

이미지를 입력으로 하는 CNN 모델을 활용하여 유성을 판별합니다.

#### 결론

다음은 각 모델 별로 정확도와 F1 스코어를 계산한 값입니다. LSTM을 사용한 모델이 F1 score가 제일 높게 나온 것을 확인 할 수 있습니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-long-period-comets-final-presentation-18-1024.jpg)

---

### 3. Lunar Water and volatiles

#### 개요

수자원이 풍부한 지역을 탐사하기 위한 crater map의 자동 제작을 연구합니다.달의 남극에서 대규모 데이터 셋을 수집하고 크레이터 감지에 초점을 둔 고급의 feature 추출을 수행합니다. 98,4%의 높은 성공률로 전문가보다 100배 빠른 속도향상을 이루어냈습니다. 

#### 달에 물이 존재하는 곳

+ 극 근처에 존재하는 크레이터
+ 태양이 닿지 않는 크레이터의 바닥
+ 영원히 그림자 진 지역**(PSRs)** 

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-lunar-water-and-volatiles-7-1024.jpg)

대부분의 달에 존재하는 물은 **극점에 있는 PSRs**에 존재합니다. 따라서 달에 존재하는 물을 찾기위해서는 먼저 PSR 및 크레이터를 연구할 필요가 있습니다. 그러나 달의 극점을 Mapping 하기위한 문제점이 존재합니다.

* Co-regstration issues
* Artifacts
* Image illumination

따라서 의미 있는 실험을 수행하기 위해서는 노동집약적인 많은 데이터가 필요합니다.

#### Deep Learning Classifier

달의 크레이터를 분별하기 위해서 달의 사진을 데이터 셋으로 사용합니다. 달의 DEM(Digital Elevation Model)/NAC(Narrow Angle Camera)을 Annotation 한 뒤 사용합니다. 사용된 데이터 셋의 크기는 다음과 같습니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-lunar-water-and-volatiles-19-1024.jpg)

Annotation과정을 거친 데이터 셋은 CNN을 사용하는 Classifier 모델을 학습시키는 데 사용됩니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-lunar-water-and-volatiles-20-1024.jpg)

#### 결론

FDL팀이 연구한 CNN모델이 지난 팀들이 연구한 패턴인식을 사용한 방법이나 CNN모델 보다 월등히 뛰어난 정확도를 보이고 있습니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-lunar-water-and-volatiles-21-1024.jpg)

---

### 4. Radar 3D Shape Modeling

#### 개요

2016 년부터 작업을 계속하면서 팀은 형상 모델링 워크 플로에 따라 AI 기술을 적용했습니다. 소행체 모양 모델링은 기존 소프트웨어를 사용하는 전문가가 수동으로 개입하는 데 최대 4 주가 소요됩니다. 이 팀은 Neural Nets를 최적화하고 GAN(Generative Adversarial Nets)을 활용하여 몇 시간 내에 NEO(Near Earth Object)를 모델링 할 수있는 자동화를위한 파이프 라인을 시연했습니다.

#### 데이터셋

DBSCAN(Density-Based spatial clustering of applications with noise)은 주변 데이터들의 밀도를 이용해 군집을 생성해 나가는 방식을 말합니다. DBSCAN 결과에 image를 Masking 하여 이미지 데이터 전처리를 실시합니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-3d-shape-modeling-10-1024.jpg)

#### GAN모델을 이용한 소행성 모델링 생성

GAN(Generative adversarial networks)이란 Generator 네트워크와 Discriminator 네트워크가 서로 경쟁하며 성능을 점차 증진시켜나가는 방식의 모델입니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-3d-shape-modeling-34-1024.jpg)

Generator에서 만들어진 모델링을 Discriminator가 판별하여 결과를 Generator에게 알려줍니다. 이를 계속 반복하면 Discriminator는 가짜를 판별하는 능력이 향상될 것이며 Generator는 실제 모델링과 비슷한 모델링을 만들어 낼 것 입니다. 두 네트워크의 경쟁을 통해 실제와 비슷한 모델링을 만들어 내는 것이 목표입니다.

---

### 5. Solar Terrestrial Interactions

#### 개요

지구 자기권의 적도 전류링 문제를 풀면서 딥러닝 기술을 과학적인 돌파구를 위한 도구로서의 가능성을 살펴보겠습니다. 오픈 소스 시스템 학습 프레임 워크 위에 STING (Solar Terrestrial Interactions Neural Network Generator)이라는 지식 검색 모듈을 구축하여 연구자가 복잡한 데이터 세트를 더 자세히 탐색 할 수 있게 했습니다. 

#### Space Weather가 주는 영향

"우주 환경" 혹은 "우주 기상"이라는 용어는 우리가 지구에서 사용하는 기술의 성능에 영향을 미칠 수 있는 태양과 우주에서의 변수 상태를 지칭합니다. 우주 기상은 전선의 극심한 전류를 유도하는 전자기장을 생성하여 전선을 방해하고 광범위한 정전을 일으키기도 합니다. 심각한 우주 기상은 태양 에너지 입자도 생성하여 상업용 통신, 세계 위치 파악, 정보 수집 및 기상 예보에 사용되는 위성을 손상시킬 수 있습니다 다음은 우주 기상의 예시인 오로라가 끼치는 영향을 그림으로 나타낸 것입니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-terrestrial-interactions-5-1024.jpg)

#### B - Sting

우주 일기 예보를위한 데이터 기반 오픈 소스 도구로 지자기 장애를 포착하는 Kp 지수를 예측합니다. 데이터 소스로는 지구의 자기장과 태양풍 데이터를 사용합니다. 
Kp 지수는 매일 3 시간 간격으로 측정 한 지자기 활동 수준의 범위를 나타냅니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-terrestrial-interactions-7-1024.jpg)
![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-terrestrial-interactions-11-1024.jpg)

지구의 자기장과 태양풍 데이터를 입력으로 Kp지수를 측정하기 위해서 다음과 같은 두가지 프로젝트를 실시했습니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-terrestrial-interactions-13-1024.jpg)

##### LSTM RNN 모델

LSTM유닛은 여러 개의 게이트(gate)가 붙어있는 셀(cell)로 이루어져있으며 이 셀의 정보를 새로 저장/셀의 정보를 불러오기/셀의 정보를 유지하는 기능이 있다. 셀은 셀에 연결된 게이트의 값을 보고 무엇을 저장할지, 언제 정보를 내보낼지, 언제 쓰고 언제 지울지를 결정합니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-terrestrial-interactions-14-1024.jpg)

##### Gradient Boosting 모델

함께 합쳐지면 손실 함수가 최소화되도록 트리 앙상블을 찾습니다. 각 트리는 전반적인 문제에 대한 솔루션의 추정치입니다. 훈련이 반복 될 때마다 나무에 재사용됩니다. 각 취약한 트리가 전반적인 강력한 솔루션에 기여할 수 있게 하였습니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_GB.png)

#### 결론

다음은 Gradient Boosting 방법을 통해 3시간뒤의 Kp지수를 예측한 값과 실제 값을 비교한 그래프입니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-terrestrial-interactions-18-638.jpg)

Gradient Boosting 방법을 사용한 결과가 기존에 제시한 방법들보다 95%의 정확성을 보여주면서 뛰어난 성능을 보이고 있습니다.

![img](http://tykimos.github.io/warehouse/2018-1-17-NASA_FDL_LAB_Study_fdl-2017-solar-terrestrial-interactions-19-638.jpg)
