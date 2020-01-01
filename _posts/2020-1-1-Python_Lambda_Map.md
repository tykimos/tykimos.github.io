---
layout: post
title:  "파이썬 람다(lambda)랑 맵(map) 케미 - 람다시리즈 3부"
author: 김태영
date:   2020-1-1 13:00:00
categories: python
comments: true
image: http://tykimos.github.io/warehouse/2020-1-1-Python_Lambda_Map_1.png
---

본격적으로 람다의 활용이 돋보이는 예제를 살펴보겠습니다. 람다와 맵이 케미를 이루면 아래와 같은 코드로 한 줄 구구단을 만들 수 있습니다.

![img](http://tykimos.github.io/warehouse/2020-1-1-Python_Lambda_Map_1.png)

아직 람다가 생소하신 분은 아래 링크를 먼저 봐주세요. 

* 이전보기
   * [파이썬 람다](https://tykimos.github.io/2019/12/25/Python_Lambda/)
   * [파이썬 람다함수](https://tykimos.github.io/2019/12/29/Python_Lambda_Function/)

람다에 대해선 알아봤으니, 케미를 이룰 맵에 대해서 알아보겠습니다.

함수는 입력을 받아 어떤 처리를 합니다. 다른 입력으로 함수결과를 보려면 보통 입력을 바꿔서 함수를 다시 호출하는 식입니다. 만약 입력이 여러개라면 for문을 돌면서 함수를 여러번 호출을 할 수 있지만 맵(map)을 통하여 좀 더 쉽게 할 수 있습니다. 즉

```python
map(함수, 입력들)
```

이렇게 사용하면, 입력들(입력 리스트) 만큼 입력을 바꾸면서 함수를 호출할 수 있습니다. 1에서 5까지 제곱을 구하는 간단한 예제로 살펴보겠습니다. 일반적으로 다음과 같이 코드를 작성할 수 있습니다.

```python
def calc(x):
    return x*x

for i in range(1, 6):
    print(calc(i))
```

함수를 정의하고, for문을 돌면서 입력을 바꿔가며 함수를 호출하는 식입니다. 이를 map을 이용하여 입력 개수만큼 함수를 여러 번 호출할 수 있습니다. 아래와 같이 map의 첫번째 인자에 함수 이름을 넣고, 두번째 인자에는 입력들(입력 리스트)를 지정합니다.

```python
def calc(x):
    return x*x

list(map(calc, range(1,6)))
```

아직 람다를 까먹지 않으셨죠? 람다는 일시적으로 사용하고 버리는 함수라 map의 첫번째 인자에 그냥 람다로 지정해버리면 됩니다. 그럼 아래 코드와 같이 한 줄로 간단하게 됩니다.

```python
list(map(lambda x:x*x, range(1,6)))
```

하나의 입력에 두 개의 인자를 넘기고 싶을 때도 가능합니다. 아래 예제인 경우 두 리스트에서 입력값을 하나씩 가지고와서 함수를 호출하는 식입니다.

```python
in1 = [1, 3, 5, 7]
in2 = [2, 4, 6, 8]

list(map(lambda x,y:x+y, in1, in2))
```

간단하죠? 이 정도만 알면 코드리딩이나 적절한 곳에 사용하는 데에 있어서는 크게 어려움이 없을 겁니다. 재미삼아 몇 가지 예제를 더 살펴보겠습니다.

#### 구구단

한 줄로 구구단을 만들어볼까요? 나누기와 나머지를 이용해서 인자를 하나받아 계산하도록 하고 람다와 맵을 사용하여 한 줄로 만들어봤습니다.

```python
list(map(lambda x:(x//10)*(x%10), range(10,100)))
```

이를 화면에 이쁘게 표시하기 위한 장식을 조금하면 좀 더 긴~ 한 줄이 되겠네요.

```python
list(map(lambda x:str(x//10) + ' x ' + str(x%10) + ' = ' + str((x//10)*(x%10)), range(10,100)))
```
```
   ['1 x 0 = 0',
   '1 x 1 = 1',
   '1 x 2 = 2',
   '1 x 3 = 3',
   '1 x 4 = 4',
   '1 x 5 = 5',
   '1 x 6 = 6',
   '1 x 7 = 7',
   '1 x 8 = 8',
   '1 x 9 = 9',
   '2 x 0 = 0',
   '2 x 1 = 2',
   ...   
   '9 x 5 = 45',
   '9 x 6 = 54',
   '9 x 7 = 63',
   '9 x 8 = 72',
   '9 x 9 = 81']   
```

엄밀히 말하면 0이 포함되어 있어서 구구단은 아니지만 꽤 쓸만한 예제죠?

#### 369게임

369게임이라고 여럿이서 숫자를 높혀가며 부르다가 3과 6과 9 중 하나라도 나오면 숫자를 부르는 대신에 박수를 치는 게임입니다. 람수에는 조건문도 사용할 수 있으므로 아래 코드와 같이 작성할 수 있겠죠?

```python
list(map(lambda x: '짝' if x % 3 == 0 else x, range(1, 10)))
```
```
   [1, 2, '짝', 4, 5, '짝', 7, 8, '짝']
```

하지만 이건 369게임이 아니라 3의 배수에 박수를 치도록 하는 코드죠? 10이 넘어가면 사용할 수 없습니다. 조금 더 369게임에 맞게 코드를 만들어보면 다음과 같습니다.

```python
list(map(lambda x: '짝' if str(x).find('3') >= 0 or str(x).find('6') >= 0 or str(x).find('9') >= 0 else x, range(1, 20)))
```
```
   [1, 2, '짝', 4, 5, '짝', 7, 8, '짝', 10, 11, 12, '짝', 14, 15, '짝', 17, 18, '짝']
```

#### 정리하기

람다와 맵이 케미를 이루면 한 줄로 어마어마한 일들을 할 수 있다는 것을 알게 되았습니다. 다음에는 필터와 리듀스에 대해 알아볼텐데 벌써 기대되시죠?

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
