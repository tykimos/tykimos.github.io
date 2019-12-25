---
layout: post
title:  "파이썬 람다(Lambda)"
author: 김태영
date:   2019-12-25 13:00:00
categories: seminar
comments: true
image: http://tykimos.github.io/warehouse/2019-12-25-Python_Lambda.png
---

파이썬 입문자에게 람다(Lambda)가 나오면 잠시 명해질 때가 있습니다. 이해가 되는 것 같기도 하고 안되는 것 같기도 하고, 뭔가 묘한 녀석입니다. 뭔가 직관적으로 설명하는 그림이 있어 공유합니다.

![img](http://tykimos.github.io/warehouse/2019-12-25-Python_Lambda.png)

간단한 예제를 살펴보겠습니다.

```python
>>> f = lambda x: x + 2
>>> f(2)
4
```

조금 더 어려운 거 해볼까요?

```python
>>> f = lambda x,y: x + y
>>> f(1,2)
3
```

위 코드랑 같지만 아래와 같이 한 줄로 표현할 수 있습니다. 

```python
>>> (lambda x,y: x + y)(1,2)
3
```

뭐 이해는 되지만, 왜 이렇게 (어렵게) 하는 지 궁금하시죠? 일단 람다는 필요할 때 바로 정의해서 사용한 후 버리는 일시적인 함수라고 합니다(으잉? 점점 오리무중). 차차 알아보도록 하겠습니다.

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
