---
layout: post
title:  "2019 케라스 코리아 컨트리뷰톤"
author: 김태영
date:   2019-08-13 13:00:00
categories: seminar
comments: true
image: http://tykimos.github.io/warehouse/2019-8-19-Keras_Korea_Contributon_2019_title_1.png
---

케라스 코리아에서 공개SW 컨트리뷰톤에 제안한 두 개의 프로젝트가 선정되어 멘티를 모집하게 되었습니다. 공개SW 컨트리뷰톤이란 기여(Contribute)와 마라톤(Marathon)의 합성어로 참여/공유/개방/협업을 통한 과제수행으로 공개SW(오픈소스)를 개발 및 기여하는 프로그램을 말합니다.

![img](http://tykimos.github.io/warehouse/2019-8-19-Keras_Korea_Contributon_2019_title_1.png)

* 참가대상: 공개SW 개발문화에 관심이 있고 직접 참여해보고 싶은 누구나!
* 참가신청날짜: 8.5(월)~8/25(일)
* 참가신청링크: https://www.oss.kr
* 주최: 과학기술정보통신부
* 주관: 정보통신산업진흥원

케라스 코리아에서는 총 두 개의 프로젝트가 진행되며, 각 프로젝트별 두 개의 태스크가 진행됩니다. 

* 일정
    * 8/5~8/25: 멘티 참가 신청
    * 8/30: 멘티 최종발표
    * 9/7: 발대식
    * 10/19: 6주간 팀별활동 종료
    * 11/2: 최종평가회

* 멘토
    * 전미정
        * 저는 냐옹이, 여행 그리고 공부를 좋아하는 평범한 사람이에요. 만들고 싶은 iOS 애플리케이션이 있어 개발을 시작하게 되었고, 모바일에 딥러닝을 심어보고 싶어 머신러닝을 공부하게되었죠. 다른 연사분들처럼 멋진 전문 지식이나 경험은 별로 없지만 제가 공부하고 경험한 내용을 많은 사람들과 공유하는걸 즐긴답니다. 즐거운 협업과 YOLK의 ObDe 프로젝트를 담당합니다. 
    * 정연준
        * 케라스 코리아 운영진 및 셀바스AI 컴퓨터 비전 연구원. 취미로 오픈소스를 하는 사람입니다. 기술고문과 YOLK의 KoKo 프로젝트를 담당합니다.
    * 김슬기
        * 머신러닝 엔지니어라는 잡타이틀을 가지고,이번 챗봇 프로젝트멘토로 참여하게되었습니다. 평소 케라스 코리아와 캐글 코리아에서 눈팅하는 것을 즐겨하며, 새로운 환경에서 새로운 것을 도전하는 것에 관심이 많습니다. 모두에게 의미있고, 즐거운  컨트리뷰톤이되길 바랍니다.
    * 김영하
        * 새로운 기술에 관심이 많은 엔지니어 성향을 가진 개발자입니다. 전사 시스템 모니터링 및 빅데이터 플랫폼 기반의 실무 프로젝트를 수행했습니다. 프리랜서처럼 일하는 회사 인디플러스에서 데이터 분석, 인공지능 관련된 프로젝트 수행 및 강의를 하고 있습니다. 지난2~7월간 진행된MS Azure Discovery Day 행사인 2일차에 Azure에서 하는 인공지능을 담당했습니다. 기술서를 읽고 공유하고자 공부하다 보니 어느덧 번역한 책이 데이터 분석을 위한판다스 입문, 파이썬 웹 스크래핑 등 6권이 되었습니다. 재미있고 쉽게 누구나 해볼 수 있는 프로그래밍 및 인공지능 컨텐츠 를 찾으며 만들고 있습니다.
    * 김태영
        * 비전공자분들이 직관적이고 간결한 딥러닝 라이브러리인 케라스를 이용해 딥러닝 입문을 쉽게 할 수 있도록<블록과 함께 하는 파이썬 딥러닝 케라스>의 집필과 김태영의 케라스블로그, 케라스 코리아, 캐글 코리아 를운영하고 있습니다. 또한 강화  학습 코리아의 알파오목 프로젝트의 팀원으로 알파고 모델을 대중들이 접할 수 있도록 서비스도 구축했습니다. 현재(주)인스페이스에서 기술이사로서 태양에서 세포까지 딥러닝, 게임에서 우주까지 강화 학습의 모토로 여러 분야 인공지능을 적용하고자 활발히 연구개발 하고 있습니다.

### YOLK(You Only Look Keras)
---

![img](http://tykimos.github.io/warehouse/2019-8-19-Keras_Korea_Contributon_2019_YOLK_img.png)

* 소개
    * 딥러닝 오픈소스인 Keras의 기술향상과 접근성향상 두가지 모두에 기여하는 프로젝트로, ObDe와 KoKo 두 개 주제로 이뤄집니다.
    * ObDe(옵디): Keras를 활용한 Object Detection(객체검출) Platform 생성 => 주어진 이미지 컨텐츠를 분석하는 데 가장 적합한 모델을 찾아주는 Objective Detection Platform생성/API 구축
    * KoKo(코코): Keras 공식문서 한글화작업 => Keras 공식 홈페이지에서 제공하는 영어문서를 한글화하여 딥러닝 사용자들의 기술 접근성 및 사용성 향상
* 멘토
    * 전미정
    * 정연준
    * 김태영
* 가이드
    * ObDe
        * Step 1: 협업 방법 이해하고 익숙해지기
        * Step 2: Object Detection 과정 이해하기
        * Step 3: Keras Object Detection Platform 구현
        * Step 4: KerasAPI 구축
        * Step 5: Deploy
    * KoKo
        * Step 1: 협업 방법 이해하고 익숙해지기
        * Step 2: 용어통일
        * Step 3: 문서번역
        * Step 4: 오류수정
        * Step 5: Deploy
* [상세내용 다운로드](http://tykimos.github.io/warehouse/2019-8-19-Keras_Korea_Contributon_2019_YOLK_file.pdf))

### 케라콘-케라스기반챗봇시스템
---

![img](http://tykimos.github.io/warehouse/2019-8-19-Keras_Korea_Contributon_2019_Keracorn_img.png)

* 소개
    * 이번 케라스 컨트리뷰톤의 프로젝트중 하나인챗봇은 실생활에서 가장 쉽게 접할 수 있고, 활용범위가 넓어 많은 사람들이 참여하고, 함께 만들어가는데 의미가 있기 때문에주제로 채택했습니다. 프로젝트는 한글데이터를 기반으로 진행 할 예정이고, 먼저자연어 처리의 기본 과정을 진행한후Python 환경 구축을 진행 후 본격적인 프로젝트를 진행 하고자 합니다. 
    * 그 후 순서는데이터 수집, 전처리, 라벨링, 모델 생성, 모델 평가로 진행 할 예정입니다. 영화 평점 데이터가 쉽게 구할 수 있기 때문에 이와 관련된 도메인으로 진행할 예정입니다. 팀원들의 의견을 조율하여 변경은 가능합니다.
* 멘토
    * 김슬기
    * 김영하
    * 김태영    
* 가이드
    * Step 1: 컨트리뷰톤 기본 협업 방법 숙지
    * Step 2: 개발 환경 구축
    * Step 3: 자연어처리에 대한 기본 과정 이해하기
    * Step 4: 챗봇 프로젝트구축진행
    * Step 5: Deploy
* [상세내용 다운로드](http://tykimos.github.io/warehouse/2019-8-19-Keras_Korea_Contributon_2019_Keracorn_file.pdf))

### 신청하기

* [신청서 작성 링크](https://www.oss.kr)

### 같이보기

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
