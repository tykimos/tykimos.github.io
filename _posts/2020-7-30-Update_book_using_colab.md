---
layout: post
title:  "코랩을 이용해서 3년 전 집필한 케라스 도서 소스코드 업데이트 해보기"
author: 김태영 
date:   2020-07-30 12:00:00
categories: etc
comments: true
image: http://tykimos.github.io/warehouse/2020-7-30-Update_book_using_colab_title.png
---

3년 전 "블록과 함께하는 파이썬 딥러닝 케라스"라는 책을 집필하였고, 그간 많은 변동사항이 있었습니다. 케라스가 고수준 API이라서 인터페이스 상에서 크게 변화는 없었으나 IT 분야 특히 인공지능 분야에서의 3년은 엄청 긴 시간이기에 예제 소스코드 여기저기서 오류가 발생하기 시작하였습니다. 

---
### 소스코드 수정

제 책의 예제코드는 깃헙으로 관리(https://github.com/tykimos/tykimos.github.io)하고 있었는데, 그 당시에는 별 이득이 없었지만, 그 덕분에 오늘 이렇게 편안해지는 날이 왔군요. 일단 3년 전에 작성한 주피터 노트북이 저장된 깃헙을 살펴볼까요?

![img](http://tykimos.github.io/warehouse/2020-7-30-Update_book_using_colab_1.png)

아마추어 같은 커밋 내용과 모든 책 예제소스가 3년 전이라고 박혀있네요. 깃헙에 있는 주피터 노트북 파일은 바로 코랩에서 열어볼 수 있습니다. 즉 주피터 노트북 소스코드와 아래와 같다면,

    https://github.com/tykimos/tykimos.github.io/blob/master/_writing/2017-1-27-Keras_Talk.ipynb

github.com 대신에 colab.research.google.com/github으로 수정하여 아래 링크로 들어가면 코랩이 바로 열립니다.

    https://colab.research.google.com/github/tykimos/tykimos.github.io/blob/master/_writing/2017-1-27-Keras_Talk.ipynb#scrollTo=SZiXv34BmD1H

아래는 문제되는 코드를 수정해서 제대로 동작되는 것을 확인한 것입니다. 3년동안 변화가 있었다지만 "acc"를 "accuracy"로 수정한 것 밖에는 없습니다.

![img](http://tykimos.github.io/warehouse/2020-7-30-Update_book_using_colab_3.png)

---
### 다시 깃헙에 저장

아래 메뉴를 보시면 깃헙에 사본 저장이라는 메뉴가 있습니다. 처음엔 이것이 다른 파일로 저장되는 줄 알고 사용하지 않았는데, 이 메뉴가 깃업에 커밋하고 푸쉬해주는 명령이네요.

![img](http://tykimos.github.io/warehouse/2020-7-30-Update_book_using_colab_4.png)

저장소와 브런치를 선택할 수 있고, Colaboratory 링크 추가 옵션까지 선택할 수 있네요.

![img](http://tykimos.github.io/warehouse/2020-7-30-Update_book_using_colab_5.png)

깃헙에서 해당 주피터 노트북을 열어보면 업데이트 된 소스코드를 볼 수 있으며, 코랩 링크 버튼까지 생겼네요~ 이제 이 버튼을 클릭하면 바로 코랩이 열리고 책 예제 소스코드를 실행할 수 있습니다.

![img](http://tykimos.github.io/warehouse/2020-7-30-Update_book_using_colab_6.png)

---
### 다음은...

주피터 노트북까지는 업데이트가 되었으나 제 블로그까지는 아직 업데이트가 되지 않았습니다. 이를 어떻게 동기화를 할 것인가는 또 다른 문제이기에 조금 더 고민 후에 적절한 방법을 찾으면 다시 글을 올리겠습니다.

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
