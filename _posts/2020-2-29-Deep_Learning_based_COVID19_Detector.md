---
layout: post
title:  "딥러닝 기반 코로나19 진단법 연구 - 우한대학, 케라스 Unet++"
author: 김태영
date:   2020-02-20 12:00:00
categories: etc
comments: true
image: http://tykimos.github.io/warehouse/2020-2-29-Deep_Learning_based_COVID19_Detector_title_1.png
---
우한 대학교의 Renmin 병원 이하 여러기관과 함께 딥러닝 기법을 사용하여 CT 촬영 이미지에서 코로나19를 진단하는 방법에 대한 논문이 있어 소개드립니다. 해당 모델은 케라스로 되어 있으며 프랑스와 쏠레님이 오늘 트위트로 소개해주셨습니다. 아래 이미지를 클릭하면 해당 논문을 다운로드 받으실 수 있습니다.

[![img](http://tykimos.github.io/warehouse/2020-2-29-Deep_Learning_based_COVID19_Detector_1.png)](https://www.medrxiv.org/content/10.1101/2020.02.25.20021568v1)

---
### 소개

저작권 문제 등이 있을 것 같아 간단하게 요약 정도만 해봤습니다. 

* 본 연구는 고해상도 CT 촬영 이미지로부터 코로나19 폐렴을 검출하여 빨간색 박스로 표시하는 딥러닝 기법을 연구한 것입니다.
* 딥러닝 모델 개발에는 수집된 46,096장 CT 이미지에서 품질이 좋은 것만 골라 35,355장의 이미지를 사용했습니다.
* 확진환자 51명과 기타 질병 환자 55명에서 촬영한 이미지입니다.
* 성능은 환자 당 민감도 100%, 특이도 93.55% 이고, 이미지 당 민감도 94.34%, 특이도 99.16%입니다.
* 사용된 모델은 케라스 기반의 UNet++ 입니다. 

모델 결과를 바로 사용하지 않고 진단을 하기 위해 후처리 과정이 있습니다. 결과 이미지에서 불필요한 영역을 제거하고 4등분한 다음 연속된 세 장의 이미지에서 동일한 사분면에서 검출되면 최종 양성으로 진단하고 그렇지 않으면 음성으로 진단합니다.

![img](http://tykimos.github.io/warehouse/2020-2-29-Deep_Learning_based_COVID19_Detector_title_0.png)

---
### 왜 CT인가?

2020년 2월 26일에 발표된 뉴스를 보면, CT가 코로나19의 최고 진단법이라고 합니다. CT는 Computerized Tomography의 약자로 컴퓨터 단층촬영을 의미합니다. 

https://www.sciencedaily.com/releases/2020/02/200226151951.htm

그리고 Radiology 지에 게재된 논문을 보면, 1,014개의 케이스에 대해서 흉부CT와 RT-PCR 테스트 결과를 비교한 것이 있네요.

https://pubs.rsna.org/doi/10.1148/radiol.2020200642

논문 안에 있는 이미지를 보면, 1,014환자 중 308명이 RT-PCR에서는 음성이라고 나왔는데, CT에서는 양성이라고 나왔습니다. 하지만 이 이야기가 CT가 완벽하다는 얘긴 아니니 오해하지 마세요. 모든 진단은 완벽한 건 없습니다.

![img](http://tykimos.github.io/warehouse/2020-2-29-Deep_Learning_based_COVID19_Detector_5.jpeg)

* 진단쪽에서는 평가(메트릭)을 민감도, 특이도로 나누어서 보고를 하는데요, 이 용어가 생소하다면 [평가 이야기](https://tykimos.github.io/2017/05/22/Evaluation_Talk/)을 참고하세요.

고려대학교안산병원 영상의학과에서 COVID-19(코로나19) 폐렴에 대한 영상의학적 진단 소견 세미나 발표한 동영상이 있습니다. 코로나19에 걸린 CT 폐사진은 보면, 간유리 처럼 부옇게 보이는데, 이를 GGO(Ground glass opacity, 간유리음영)이라 부릅니다. 이 영상에서 GGO의 다양한 패턴을 보실 수 있습니다.

[![img](http://tykimos.github.io/warehouse/2020-2-29-Deep_Learning_based_COVID19_Detector_6.png)](https://www.youtube.com/watch?v=nE0Zb6C-kzg)

---
### UNet++ 란

이 논문에서 사용된 모델은 UNet++ 입니다. 

[![img](http://tykimos.github.io/warehouse/2020-2-29-Deep_Learning_based_COVID19_Detector_3.png)](https://arxiv.org/abs/1807.10165)

* 만약 케라스가 궁금하시다면 >> [케라스 이야기](https://tykimos.github.io/2017/01/27/Keras_Talk/)

---
### 서비스

아래 웹사이트를 통해서 현재 서비스를 하고 있습니다. CT 이미지를 업로드하면, 해당 CT 이미지에 빨간 박스로 표시해줍니다.

[![img](http://tykimos.github.io/warehouse/2020-2-29-Deep_Learning_based_COVID19_Detector_2.png)](http://121.40.75.149/znyx-ncov/index)

---
### 의미

논문 마지막 부분에서 AI 기반 진단의 중요성을 잘 보여주는 그림이 있습니다. 전염병인 경우 치료제 만큼이나 중요한 것이 빠른 진단인데, 그렇기에 최전방의 전문의의 압박감이 상당하다고 합니다. AI가 도와준다면 압박감을 덜어주고 빠르게 진단할 수 있어 전염병 통제에 기여할 수 있다고 합니다. 실험결과 딥러닝 기반 모델의 도움으로 방사선 전문이의 판독 시간이 65%이나 단축되었다고 하네요.

![img](http://tykimos.github.io/warehouse/2020-2-29-Deep_Learning_based_COVID19_Detector_4.png)

데이터셋만 확보되면 우리나라도 충분히 개발할 수 있을 것 같습니다. 데이터셋을 제공받을 수만 있다면 머신러닝 경연대회 플랫폼에 올려서 머신러닝 개발자와 함께 다같이 이 문제를 풀어서 높은 성능의 모델을 확보하고 이를 서비스 하고 싶습니다. 이와 관련되어 아이디어가 있으신 분은 tykim@aifactory.page 로 메일 주시면 감사하겠습니다.

---
### 둘러보기

인공지능 및 머신러닝 관련된 커뮤니티입니다. 편하게 놀러오셔요~

#### AIFactory

* [AI팩토리 머신러닝 경연대회](http://aifactory.space)

#### 케라스 코리아 커뮤니티

* [케라스 코리아 페북](https://www.facebook.com/groups/KerasKorea/)
* [케라스 코리아 슬랙 초대](https://join.slack.com/t/keraskorea/shared_invite/enQtNTUzMTUxMzIyMzg4LWQ3YmQ1YTdmNTYxOTAwZTExNmFmOGM3M2QyMjIyNzYwYTY2YTY2ZjBlNDNlZDdmMTU0NGVjYzFkMWYxNzE0ZDA)
* [케라스 코리아 단톡방](https://open.kakao.com/o/g93MSBV)
* [케라스 코리아 블로그](http://keraskorea.github.io)
* [케라스 공식 문서 한글번역 참여방법](https://tykimos.github.io/2019/02/06/Contribution_of_Keras_Document_to_Korean_Translation/)

#### 캐글 코리아 커뮤니티

* [캐글 코리아 페북](https://www.facebook.com/groups/KaggleKoreaOpenGroup/)
* [캐글 코리아 단톡방](https://open.kakao.com/o/gP24T89)
* [캐글 코리아 블로그](https://kaggle-kr.tistory.com/)

#### MLOps KR 커뮤니티

* [엠엘옵스 코리아 페북](https://www.facebook.com/groups/MLOpsKR/)
