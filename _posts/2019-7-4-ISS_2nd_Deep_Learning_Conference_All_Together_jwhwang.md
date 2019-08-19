---
layout: post
title:  "[2nd DLCAT] (실습)딥러닝으로 오디오 만나보기 - 황준원"
author: 김태영
date:   2019-07-04 14:00:00
categories: seminar
comments: true
image: http://tykimos.github.io/warehouse/2019-7-4-ISS_2nd_Deep_Learning_Conference_All_Together_jwhwang_title.png
---
최근 프로젝트를 통해 음성 데이터를 처음 다뤄보면서 음성 데이터를 전처리하고 모델 학습에 사용하는 여러 방법을 접해보았습니다. 음성은 데이터 차원의 수가 이미지보다 적지만 그 특성이 꽤 달라 사전 지식과 테크닉들이 필요했습니다. 이에 처음 음성 데이터를 다루면서 겪을 만한 어려움과 새로이 배운 노하우들을 공유하기 위해 발표를 준비했습니다. 바로 전 시간에 준비되어 있는 남기현 님의 이론편에 이어, 실제로 audio를 불러오고 다양한 feature를 추출하는 전처리와 augmentation, 그리고 연산 속도를 높이기 위한 GPU 연산 사용법을 간단한 예시 코드와 함께 소개해 드리려 합니다. audio 데이터를 활용한 딥러닝 연구를 처음 시작하시는 분들께 도움이 되었으면 합니다.

![img](http://tykimos.github.io/warehouse/2019-7-4-ISS_2nd_Deep_Learning_Conference_All_Together_jwhwang_title.png)

#### 연사소개
성균관대학교 소프트웨어학과에 재학중인 학부생이고, 다양한 딥러닝 분야를 골고루 배우며 연구부터 개발까지 두루 갖추고자 합니다. 현재 케라스 코리아 운영진 및 케라스 코리아 오픈채팅 방장을 맡고 있으며, 페이스북 그룹과 오픈 채팅방에서 다른 연구자 및 개발자 분들과 교류하고 있습니다. 지난 KCD2019에서는 케라스 코리아 세션에서 "이상한 폰트 나라로 뛰어들기" 를 발표했습니다. 

* 홈페이지: https://nuxlear.github.io/
* 깃헙: https://github.com/nuxlear
* 페이스북: https://www.facebook.com/nuxlearHwang

#### 발표자료
* [보기](https://docs.google.com/presentation/d/1SsesME3qtCvvJy6yqW1EOXPC0Rc-vwb0OmS4QXi_cao/mobilepresent?slide=id.g5cf03feef5_0_860)

#### 장소 및 시간
* 장소: 대전광역시 유성구 가정로 217 UST 과학기술연합대학원대학교, UST 강당
* 시간: 7월 4일 오후 2시

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