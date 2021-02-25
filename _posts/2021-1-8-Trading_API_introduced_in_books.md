---
layout: post
title:  "도서에서 소개되고 있는 주식매매 API"
author: 김태영 
date:   2021-1-9 00:00:00
categories: quant
comments: true
image: http://tykimos.github.io/warehouse/2021-1-8-Trading_API_introduced_in_books_title1.png
---

자동 매매 혹은 인공지능 모델을 이용한 매매를 하기 위해서는 매매를 위한 API 사용법을 익혀야 합니다. 퀀트 및 주식 자동 거래 시스템 구축에 관련된 도서 정보와 각 도서 안에서 소개된 매매를 위한 API가 어떤 것들이 있는 지 알아보겠습니다. 먼저 자동 매매 API가 무엇인 지 알아보겠습니다.

#### 자동 매매 API
---

"자동 매매 API" 단어 그대로를 풀어보겠습니다. 
- "자동"은 사람이 아닌 컴퓨터(시스템 혹은 프로그램)이 자동으로 무언가 하는 것이고,
- "매매"는 주식을 매수 혹은 매도를 하는 것을 의미하고, 
- "API"는 Application Programming Interface의 약자로 응용소프트웨어가 프로그램으로 서로 주고 받을 수 있도록 제공하는 인터페이스를 말합니다. 
즉 프로그램이 자동으로 주식을 사거나 팔 수 있도록 제공하는 인터페이스를 말합니다. 이러한 인터페이스가 있어야 파이썬이나 다른 프로그래밍 언어로 주식 매매 프로그램을 만들 수 있으며, 주요 증권사에서 자동 매매 API를 제공하고 있습니다. 

사람이 주식 매매를 하기 위해서는 각 증권사에서 제공하는 HTS을 이용하듯이 프로그램이 매매를 하기 위해서는 "자동 매매 API"를 이용하는 것입니다. 참고로
- HTS은 Home Trading System의 약자로 홈(집)에서 매매를 하기 위한 프로그램을 얘기하고, 
- WTS은 Web Trading Systemd의 약자로 웹에서 별도 프로그램 없이 바로 매매하는 서비스,
- MTS는 Mobile Trading System의 약자로 모바일에서 매매하는 앱을 의미합니다.

그럼 자동 매매 API 종류에 대해서 살펴보겠습니다. 

- 대신증권 API : CYBOS Plus >> [보기](https://money2.daishin.com/E5/WTS/Customer/GuideTrading/DW_CybosPlus_Page.aspx?p=8812&v=8632&m=9508)
- 대신증권 API : CREON Plus >> [보기](https://money2.creontrade.com/E5/WTS/Customer/GuideTrading/CW_TradingSystemPlus_Page.aspx?m=9505&p=8815&v=8633)
- 이베스트투자증권 API : xingAPI >> [보기](https://www.ebestsec.co.kr/xingapi)
- 키움증권 API : Open API+ >> [보기](https://www3.kiwoom.com/nkw.templateFrameSet.do?m=m1408010600)

대신증권에서는 CREON와 CYBOS 두 가지를 제공하는데요, CYBOS는 기존의 대신증권이 제공하는 서비스이고, CREON은 은행과 협업해서 만든 신규 매매 서비스라고 하네요. 다른 블로그에서는 역사가 깊은 CYBOS를 통해서 정보를 수집하고, CREON을 통해 매매를 한다고 합니다.

#### 자동 매매 관련 도서
---

지금까지 구입한 자동 매매 혹은 퀀트 관련 도서를 나열해봤습니다. 주로 파이썬 기반의 자동 매매나 머신러닝 및 인공지능 알고리즘 위주로 구매를 했었습니다.

* 파이썬으로 배우는 알고리즘 트레이딩 (내 손으로 만드는 자동 주식 거래 시스템), 조대표 지음, 위키북스, 2019년 03월 07일 출간 >> [보기](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791158391461&orderClick=LAG&Kc=)
* 퀀트 전략을 위한 인공지능 트레이딩 (파이썬과 케라스를 활용한 머신러닝/딥러닝 퀀트 전략 기술), 김태헌, 신준호 지음, 한빛미디어, 2020년 08월 20일 출간 >> [보기](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791162243312&orderClick=LEa&Kc=)
* 파이썬과 케라스를 이용한 딥러닝/강화학습 주식투자 (퀀트 투자 알고리즘 트레이딩을 위한 최첨단 해법 입문), 퀀티랩 지음, 위키북스, 2020년 04월 27일 출간 >> [보기](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791158392031&orderClick=LEa&Kc=)
* 핸즈온 머신러닝 딥러닝 알고리즘 트레이딩 (파이썬, Pandas, NumPy, Scikit-learn 케라스를 활용한 효과적인 거래 전략), 스테판 젠슨 지음, 홍창수 , 이기홍 옮김, 에이콘출판, 2020년 07월 31일 출간 >> [보기](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791161754321&orderClick=LEa&Kc=)
* 파이썬과 리액트를 활용한 주식 자동 거래 시스템 구축 (데이터 수집부터 거래자동화, API 서버, 웹 개발, 데이터분석까지 아우르는), 박재현 지음, 위키북스, 2020년 02월 12일 출간 >> [보기](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791158391881&orderClick=LAG&Kc=)
* 손가락 하나 까딱하지 않는 주식 거래 시스템 구축 (파이썬을 이용한 데이터 수집과 차트 분석, 매매 자동화까지), 장용준 지음, 위키북스, 2020년 04월 27일 출간 >> [보기](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791158392024&orderClick=LEa&Kc=)
* 파이썬 증권 데이터 분석 (파이썬 입문, 웹 스크레이핑, 트레이딩 전략, 자동 매매 (딥러닝을 이용한 주가 예측까지), 김황후 지음, 한빛미디어, 2020년 07월 01일 출간 >> [보기](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791162243206&orderClick=LAG&Kc=)

주제와 관련이 깊은 내용이 포함되어 있으나 위 목록에 빠진 도서가 있다면 댓글로 남겨주세요~ 한 번 살펴보겠습니다.

![img](http://image.kyobobook.co.kr/images/book/large/461/l9791158391461.jpg) ![img](http://image.kyobobook.co.kr/images/book/large/312/l9791162243312.jpg) ![img](http://image.kyobobook.co.kr/images/book/large/031/l9791158392031.jpg) ![img](http://image.kyobobook.co.kr/images/book/large/321/l9791161754321.jpg) ![img](http://image.kyobobook.co.kr/images/book/large/881/l9791158391881.jpg) ![img](http://image.kyobobook.co.kr/images/book/large/024/l9791158392024.jpg) ![img](http://image.kyobobook.co.kr/images/book/large/206/l9791162243206.jpg)


#### 책에서 소개된 자동 매매 API
---
저자마다 선호하는 API가 있으며, 독자들도 선호하는 API가 있기 때문에 각 책을 살펴보면서 어떤 매매 API를 이용하였는 지 정리를 해봤습니다. 책을 구매하거나 스터디 하실 때 참고하시면 좋을 것 같습니다. 책에 소개되는 API는 크게 데이터 수집을 위한 API와 매매를 위한 API가 있습니다. 데이터 수집을 위한 API는 다른 게시물로 살펴보기 이번에는 매매를 위한 API에만 집중해서 살펴보겠습니다.

|도서|자동매매API|
|-|-|
|파이썬으로 배우는 알고리즘 트레이딩|CYBOS Plus, xingAPI, Open API+(메인)|
|퀀트 전략을 위한 인공지능 트레이딩|-|
|파이썬과 케라스를 이용한 딥러닝/강화학습 주식투자|-|
|핸즈온 머신러닝 딥러닝 알고리즘 트레이딩|온라인 트레이딩 플래폼|
|파이썬과 리액트를 활용한 주식 자동 거래 시스템 구축|xingAPI|
|손가락 하나 까딱하지 않는 주식 거래 시스템 구축|Open API+|
|파이썬 증권 데이터 분석|CREON Plus|

살펴보면서 몇가지 의견을 적어보면, "핸즈온 머신러닝 딥러닝 알고리즘 트레이딩" 도서에는 매매API는 소개되어 있지 않지만, 매매 알고리즘을 구사하였다면 이를 탑재시킬 수 있는 온라인 트레이딩 플랫폼이 소개되고 있습니다. 이러한 온라인 트레이딩 플랫폼도 다른 게시물로 살펴보겠습니다.
* 퀀토피안(폐쇄됨)
* 퀀트코넥트 >> [보기](https://www.quantconnect.com/)
* 퀀트로켓 >> [보기](https://www.quantrocket.com/)

그리고 "손가락 하나 까딱하지 않는 주식 거래 시스템 구축"은 주식 거래가 주요 주제이다보니 Open API+만 소개되어 있긴 하지만 실전에서 필요한 부분들(미체결 종목 처리 등)에 대해서 상세하게 설명되어 있습니다.

결론
---
주식 API를 크게 데이터 수집용과 매매용으로 봤을 때, 두가지 용도를 하나의 API만을 사용할 이유는 없을 것 같습니다. 간단하게 입문용으로 하려면 어느 API나 상관없지만, 전문적으로 하기 위해서는 모델에 따라 매매 빈도수가 많을 수 있고 또는 타이밍도 중요하기 때문에, 수수료 혹은 응답속도 등도 확인을 해봐야될 것 같네요. 일단 저는 입문용으로 할 것이기에 이미 계좌가 개설된 일단 키움증권의 Open API+로 시작해보겠습니다.
