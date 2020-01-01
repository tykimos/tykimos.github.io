---
layout: post
title:  "파이썬 람다 맵(Lambda Map) 케미, 한줄 구구단 - 람다시리즈 3부"
author: 김태영
date:   2020-1-1 13:00:00
categories: python
comments: true
image: http://tykimos.github.io/warehouse/2020-1-1-Python_Lambda_Map_0.png
---

본격적으로 람다의 활용이 돋보이는 예제를 살펴보겠습니다. 아직 람다가 생소하신 분은 아래 링크를 먼저 봐주세요. 

* 이전보기
   * [파이썬 람다](https://tykimos.github.io/2019/12/25/Python_Lambda/)
   * [파이썬 람다함수](https://tykimos.github.io/2019/12/29/Python_Lambda_Function/)

람다와 맵이 케미를 이루면 아래와 같은 코드로 한 줄 구구단을 만들 수 있습니다.

![img](http://tykimos.github.io/warehouse/2020-1-1-Python_Lambda_Map_0.png)

람다자체가 함수라서 람다함수라고 부르기엔 애매합니다만 풀어서 말하면 람다를 반환하는 함수라고 하는 것이 맞겠네요.

```python
def sec2other(unit):
   return lambda sec: sec/unit

sec2min = sec2other(60)
sec2hour = sec2other(3600)

print(sec2min(180))
print(sec2hour(7200))
```

초를 분이나 시간으로 환산하는 예제인데, 기존 방식이라면 변환식마다 함수를 따로 만들겠지만, 람다함수를 이용하여 사용할 함수를 그때 그때 정의해서 사용하는 식입니다. 초를 일(day)로 바꿔야한다면 한 줄만 더 추가하면 되겠죠?

다른 예제를 살펴볼까요? 해외여행이나 해외출장가서 금액을 보면 이게 한국돈으로 얼마인지 궁금하잖아요. 만약 가격을 입력하면 한화로 바꾸는 프로그램을 짠다고 했을 때, 어느나라에 갈 지 모르니 전세계 통화에 해당하는
함수를 모두 만들수도 있겠지만, 우린 람다함수를 만들었으니 그때그때 필요할 때만 만들어서 쓰면 되겠죠?

```python
def other2won(unit):
   return lambda price: price*unit

usd2won = other2won(1160.50) #달러환율
eur2won = other2won(1293.90) #유로환율

print(usd2won(10)) #10달러는 한국돈으로 얼마?
print(eur2won(15)) #15유로는 한국돈으로 얼마?
```

두 예제 모두 일시적으로 사용했다기 보다는 함수 정의를 그때 그때 바꿀 수 있어 편리함과 확장성이 돋보이네요. 하지만 함수도 변수와 같이 객체처럼 다루고 있기때문에 일시적으로 사용한다는 감을 잡으실 수 있습니다. 변수라는 것도 계속 사용한다기 보다는 계산 상 혹은 잠시
값을 저장하는 용도로 일시적으로 사용하고 버리기는 경우가 대부분입니다. 다음에는 함수를 객체처럼 다뤄서 일시적으로 사용한 예제를 살펴보면서 람다의 파워를 느껴보시죠.

---

### 둘러보기

인공지능 및 머신러닝 관련된 커뮤니티입니다. 편하게 놀러오셔요~

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
