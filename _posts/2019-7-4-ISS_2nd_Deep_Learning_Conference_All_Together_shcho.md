---
layout: post
title:  "[2nd DLCAT] 3분 강화학습 순한맛 SAC(Soft Actor Critic) feat. AC(Actor Critic) - 조수현"
author: 김태영
date:   2019-07-04 10:00:00
categories: seminar
comments: true
image: http://tykimos.github.io/warehouse/2019-7-4-ISS_2nd_Deep_Learning_Conference_All_Together_shcho_title.png
---
RL(Reinforcement Learning) 알고리즘은 최적화 문제에 쓰일 수 있고 앞으로 미래 먹거리가 될 수 있습니다. 특히나 로보틱스나 제어에서 많이 쓰이구요. 하지만 RL은 어렵습니다. 특히나 수식의 향연이 더욱 부추깁니다. 그래서 각 알고리즘의 아이디어에 대한 직관적인 이해가 특히 더 요구됩니다. 본 시간은 2018년 ICML 논문인 SAC(Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor) 알고리즘에 대해서 기본에 충실한 핵심 내용을 최대한 알기 쉽게 빠르게 공유 및 전달 하는 시간입니다. SAC알고리즘은 SAC알고리즘은 RL의 기본 가정인 tabular MDP를 Soft-MDP로 재정의(확장)하여 확률적으로 문제를 학습, 처리 하는 아이디어입니다. 이를 실현하기 위해 특별한(specific) 가정 추가 없이 기존 문제영역(high variance)을 해결하여 비교적 좋은 성능을 가진 알고리즘입니다. 금과옥조 (金科玉條)처럼 모든 task에 사용되는 알고리즘은 없습니다. 하지만 대개의 문제에 비교적 높은 성능과 학습 안정성을 나타내어 Google 내부적으로 직원들 사이에서 제일 인기가 있는 베이스라인 알고리즘이 바로 SAC 입니다. 함께 지식을 쌓아나갑시다. 실력자 분들 께서는 본 강연이 평이 할 수 있으니 컨퍼런스 당일 이 점 참고 하시어 즐기시면 됩니다.

![img](http://tykimos.github.io/warehouse/2019-7-4-ISS_2nd_Deep_Learning_Conference_All_Together_shcho_title.png)

#### 연사소개
현재 디아이티에서 딥러닝 엔지니어로 현실문제를 어떻게 해결 할 수 있나 즐겁게 준비하고 고민하고 일하고 있습니다. 이전 회사들에서는 B2B, B2C 애플리케이션 기반 서비스들의 백엔드개발을 다양한 아키텍쳐에서 주로 경험 하였습니다. 프론트엔드 개발도 주로 하였구요. 현재는 최고의 실력자들이 계신 디아이티에서 딥러닝 엔지니어로 현실문제를 어떻게 해결 할 수 있나 고민하고 많이 배우며 즐겁게 일하고 있습니다. 인공지능의 한 분야로서 딥러닝과 또 다른 맥락인 강화학습에 대해 초보 연구자로써 틈틈히 반 취미로 공부하고 배우고 있습니다. 또한 Real Lab 강화학습 스터디의 운영진으로 있습니다. 하고 싶은 건 많고 몸은 하나인 평범한 엔지니어입니다. 가치 추구 실현과 시간적 자유를 추구합니다. 

* 깃헙:  https://github.com/humblem2
* 이메일: seanbrowncho@gmail.com 

#### 발표자료
* [보기](https://www.slideshare.net/SuHyunCho2/sac-overview)

#### 참고자료
* 논문: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

#### 장소 및 시간
* 장소: 대전광역시 유성구 가정로 217 UST 과학기술연합대학원대학교, UST 강당
* 시간: 7월 4일 오전 10시

|시간|A-USTaudi|B-USTsci|C-USTmeet|D-ETRI212|E-ETRI224|F-ETRI219|
|-|-|-|-|-|-|-|
|10시|<b>조수현</b><br>3분 강화학습 순한맛 SAC|<b>이수진</b><br>AI시대의 예술작품 - AI Atelier를 이용하여|<b>박해선</b><br>케라스 in 텐서플로우2.0|<b>유용균</b><br>딥러닝과 최적설계|<b>이현호</b><br>(실습)유니티 기반 드론 강화학습 (1)|<b>정연준</b><br>아기다리고기다리던딥러닝 - 케라스로 띄어쓰기 정복하기 (1)|
|11시|<b>안수빈</b><br>The Newbie Guide to Blogging & Visualization|<b>김준태</b><br>나도 너도 모르는 Graph Neural Network의 힘|<b>안종훈</b><br>설명가능한 AI for AI 윤리|<b>이유한</b><br>I'm Kaggler - Why need kaggle?|<b>이현호</b><br>(실습)유니티 기반 드론 강화학습 (2)|<b>정연준</b><br>아기다리고기다리던딥러닝 - 케라스로 띄어쓰기 정복하기 (2)|
|13시|<b>남기현</b><br>(이론)딥러닝으로 오디오 만나보기|<b>김유민</b><br>딥러닝 모델 엑기스 추출(Knowlege Distillation)|<b>홍원의</b><br>(실습)한페이지 논문잡기:찾고+읽고+쓰고+정리하기|<b>서정훈</b><br>빽 투 더 Representation Learning: Visual Self-supervision을 중심으로|<b>신경인</b><br>(실습)파이토치로 갈아타기 (1)|<b>전미정</b><br>(실습)MS Azure ML Service와 함께하는 AutoML 사용하기(1)|
|14시|<b>황준원</b><br>(실습)딥러닝으로 오디오 만나보기|<b>김영하</b><br>AutomatedML 동향|<b>홍원의</b><br>(실습)한페이지 논문잡기:찾고+읽고+쓰고+정리하기|<b>송규예</b><br>Deeplema, 딥러닝 서비스상용화의 딜레마|<b>신경인</b><br>(실습)파이토치로 갈아타기 (2)|<b>전미정</b><br>(실습)MS Azure ML Service와 함께하는 AutoML 사용하기 (2)|
|15시|<b>민규식</b><br>강화학습 환경 제작, Unity ML-agents와 함께하세요|<b>김태진</b><br>구글 코랩 TPU 알아보기|<b>김보섭</b><br>Structuring your first NLP project (1)|<b>이진원</b><br>Efficient CNN 톺아보기|<b>김경환,박진우</b><br>(실습)Rainbow로 달착륙부터 Atari까지 (1)|<b>대전AI거버넌스</b><br>AI 거버넌스 구성|
|16시|<b>옥찬호</b><br>카드게임 강화학습 환경 개발기 - 하스스톤|<b>김형섭</b><br>GAN 동향|<b>김보섭</b><br>Structuring your first NLP project (2)|<b>차금강</b><br>설명가능한 강화학습|<b>김경환,박진우</b><br>(실습)Rainbow로 달착륙부터 Atari까지 (2)|<b>대전AI거버넌스</b><br>AI 적용 가속화 방안|
|17시|<b>김태영</b><br>이제|<b>김태영</b><br>하이퍼파라미터|<b>김태영</b><br>튜닝은|<b>김태영</b><br>케라스 튜너에게|<b>김태영</b><br>맞기세요|<b>대전AI거버넌스</b><br>한계 및 목표치 설정|

* 점심시간은 12시 ~ 13시입니다.
* 각 세션은 45분 발표, 5분 질의응답, 10분 휴식 및 이동입니다.
* UST과 ETRI사이는 도보로 10분이내 거리에 있습니다. 따라서 쉬는 시간을 이용해서 이동하시면 됩니다.

[상세 프로그램 보기](https://tykimos.github.io/2019/07/04/ISS_2nd_Deep_Learning_Conference_All_Together/)

#### 참가신청
---

신청은 아래 링크에서 해주세요~

#### >> [신청하기](https://forms.gle/DFYtGWS7aDj1Bmow8) <<

딥러닝을 시작하는 이유는 달라도 딥러닝을 계속 하는 이유 중 하나는 바로 '함께하는 즐거움'이지 않을까합니다. 작년 6월 말 대전에서 ["1st 함께하는 딥러닝 컨퍼런스"](https://tykimos.github.io/2018/06/28/ISS_1st_Deep_Learning_Conference_All_Together/)에 400명 넘게 모여 즐겁게 인공지능 및 딥러닝 관한 다양한 주제로 톡을 나누었습니다.  그간 매일 논문만 읽어도 못 따라갈 만큼 새로운 연구가 쏟아지고 있고, 그 활용 사례 및 관심 또한 각 분야에 퍼져가고 있습니다. 대전은 전국 각지에서 오시기에 접근성이 용이하고, 정부출연연구원 및 정부청사, 우수한 대학교, 대기업의 기술 연구소, 최첨단 기술 중심의 벤처회사들이 밀집된 지역인 만큼 지식공유의 즐거움을 나누고자 합니다. 

    별도의 참가비는 없습니다. 연사분들도 여러분과 즐기게 위해 재능기부합니다. 주차공간이 협소하므로 대중교통을 이용해주세요.

![img](http://tykimos.github.io/warehouse/2019-7-4-ISS_2nd_Deep_Learning_Conference_All_Together_title8.png)

* 일시: 2019년 7월 4일 (10시~18시)
* 장소: 대전광역시 유성구 가정로 217 
    * 과학기술연합대학원대학교 - 대전광역시 유성구 가정로 217 
    * ETRI 융합기술연구생산센터 - 대전광역시 유성구 가정로 218 
* 주최: (주)인스페이스 - (주)인스페이스는 한국항공우주연구원 출신 연구원들이 시작한 벤처회사로 위성지상국 개발 및 활용 전문 기술을 기반으로 사업 영역을 확장하고 있습니다. 현재 “태양에서 세포까지 딥러닝”, “게임에서 우주까지 강화학습”의 모토로 여러분야 인공지능을 적용하고자 활발히 연구 개발 중입니다. 인공지능은 기술을 넘어 산업체, 학계, 정부출연연과 오픈 커뮤니티의 공동상생할 수 있는 생태계를 만들고 있기 때문에 인스페이스는 대전을 중심으로 인공지능 생태계를 위한 소통의 장을 형성하기 위해 노력하고 있으며 확대할 계획입니다.
* 주관: 대딥사, 케라스 코리아, 캐글 코리아, RL 코리아
* 후원: 
   * UST 과학기술연합대학원대학교
   * ETRI 융합기술연구생산센터
   * 대전정보문화산업진흥원
   * (주)유클리드소프트 - 유클리드소프트는 정부부처 및 공공기관 서비스 개발과정에서 축적한 솔루션 기반 탄탄한 기술력 위에 빅데이터 분석, AI기반 CCTV 행동 패턴 분석, CNN, RNN, GAN 등 최신 기술을  접목하여, 고객의 핵심 가치에 더 나은 핵 가치를 제공하고자 계속 노력하고 있는 덕후 개발자들의 회사입니다.
* 대상: 인공지능 및 딥러닝에 관심있거나 관련 연구를 수행하시는 분들 (약 700명)
    * 트랙 A: UST 강당 300명
    * 트랙 B: UST 사이언스홀 65명
    * 트랙 C: UST 대회의실 35명
    * 트랙 D: ETRI 융합기술연구생산센터 212호 대회의실 180명
    * 트랙 E: ETRI 융합기술연구생산센터 224호 중회의실 50명
    * 트랙 F: ETRI 융합기술연구생산센터 219호 중회의실3 20명

![img](http://tykimos.github.io/warehouse/2019-7-4-ISS_2nd_Deep_Learning_Conference_All_Together_room.png)