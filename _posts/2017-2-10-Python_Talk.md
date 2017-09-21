---
layout: post
title:  "파이썬 이야기"
author: 김태영
date:   2017-02-10 04:00:00
categories: Study
comments: true
---
케라스로 딥러닝 모델을 만들고, 데이터를 전처리하기 위해 필요한 최소한의 파이썬 내용을 다루고자 합니다. 

---

### 공부할 때 쓰기 좋은 개발환경 - 주피터 노트북

파이썬을 구동하려면 파이썬 개발환경이 필요합니다. 메모장이나 텍스트편집기를 이용해서 코드를 작성하고, 콘솔 창에서 실행해도 되지만 좀 더 편한 방법들이 많이 있습니다. 그 중 주석을 자유롭게 작성할 수 있고 특정 부분만 실행시킬 수 있는 주피터 노트북에 대해서 알아보겠습니다. 웹 브라우저에서 구동되기 때문에 주피터 노트북만 익혀놓으면, 운영체제와 상관없이 익숙한 환경에서 파이썬 코드를 작성할 수 있습니다.

### 파이썬 소개

파이썬은 간단하고 직관적인 언어이기 때문에, 간단한 문법만 익혀놓으면 남의 코드를 보거나 직접 코드를 작성할 때에도 쉽게 하실 수 있습니다. 그리고 막강한 라이브러리들이 제공하기 때문에 좀 전문적인 기능들은 이 라이브러리에서 쉽게 찾을 수 있습니다. 파이썬 버전은 2.X와 3.X가 있습니다. 3.X가 더 최신 버전이지만 아직 많은 사람들이 2.X을 사용하고 있기 때문에 배울 때 참고할 만한 코드들이 2.X로 되어 있는 것이 많습니다. 따라서 2.X 버전의 파이썬에 대해서 익혀보겠습니다. 3.X가 필요할 시점이 되었을 때, 2.X로 익히신 분이 3.X를 사용하시는 데는 크게 어렵지 않으실 겁니다. 

### 계산과 화면 출력

주피터 노트북에서는 현재 실행되는 셀의 마지막 명령의 반환 값이 있을 경우 그 값을 화면에 출력 해줍니다. 이 기능을 이용해서 간단한 계산 및 결과값을 출력해봅시다.


```python
350 + 27
```


```python
14 / 3 # 정수 나누기
```


```python
14.0 / 3.0 # 실수 나누기
```


```python
3 ** 2 # 제곱
```

### 기본형

#### 변수 선언


```python
a = 2
b = 1
x = 3
y = a * x + b

y
```

#### 정수 및 실수


```python
x = 5

x + 3
```


```python
x = x + 3

x
```


```python
x += 5.0

x
```


```python
x *= 2

x
```

#### 논리형


```python
a = True
b = False

a and b
```


```python
a or b
```


```python
not a
```


```python
a == b
```


```python
a != b
```

#### 출력하기


```python
i = 3

print(i)

print('%d' % i)

print(i + 2)

print(type(i))

j = 0.3

print(type(j))
```

#### 문자열


```python
h = 'hello'
k = "keras"

print(h)
print(len(h))
hk = h + ' ' + k
print(hk)
print('%s %s %d' % (h, k, 2017))
print(h + ' ' + k + ' ' + str(2017))

print(h, hk)
```


```python
msg = 'keras'

print(msg.capitalize())
print(msg.upper())
print(msg.replace('ras', 'RAS'))

msg = '  keras  '

print(msg.strip())
```

#### 자료구조

배열


```python
values = [2, 6, 3]

print(values)
print(values[1])
print(values[-1])

values[0] = 'first'

print(values)

values.append(5)
values.append('end')

print(values)

item = values.pop()

print(item, values)
```

#### 배열 잘라내기


```python
range(5)
```


```python
values = range(5)

print(values)
print(values[2:4])
print(values[2:])
print(values[:4])
print(values[:])
print(values[:-1])

values[2:4] = [8, 9]

print(values)
```

#### 반복문


```python
idx = 0

while idx < 10:
    print(idx)
    idx = idx + 1
```


```python
for idx in range(10):
    print(idx)
```


```python
people = ['kim', 'lee', 'choi']

for p in people:
    print(p)
```


```python
people = ['kim', 'lee', 'choi']

for i, p in enumerate(people):
    print('%d : %s' % (i, p))
```

#### 연습문제 구구단


```python
for i in range(9):
    for j in range(9):
        print("%d x %d = %d" % (i+1, j+1, (i+1) * (j+1)))
```

#### List comprehensions


```python
values = [0, 1, 2, 3, 4]
squares = []
for v in values:
    squares.append(v ** 2)
print squares
```


```python
values = [0, 1, 2, 3, 4]
squares = [v ** 2 for v in values]
print squares
```


```python
values = [0, 1, 2, 3, 4]
even_squares = [v ** 2 for v in values if v % 2 == 0]
print even_squares
```


```python
print([[(i+1)*(j+1) for i in range(9)] for j in range(9)])
```

#### 딕션러리

A dictionary stores (key, value) pairs, similar to a `Map` in Java or an object in Javascript. You can use it like this:


```python
dic = {}
dic['물리학과'] = '물리학의 각 분야에 걸친 이론과 응용방법을 심오하게 교수, 연구함으로써 독창적 능력을 함양하고 고도 산업사회를 선도해 갈 지도적 인재를 양성함을 목적으로 한다.'
dic['우주과학과'] = '우주과학과는 천체 및 우주에서 일어나는 제반 현상을 과학적으로 탐사하고 연구하는 학과이다. 본 학과는 인류의 우주진출이 더욱 활발해 지고 있는 이 시대에 그를 위한 지식과 기술의 개발과 보급을 목적으로 설립되었다. 현대 천문학에서부터 인공위성과 우주선의 활용에 이르는 기초와 응용의 병행 학습을 통하여 21세기 우주 시대가 요구하는 첨단분야에서 국제적인 경쟁력이 있는 인재를 양성하는 데에 우주과학과의 교육 목적이 있다.'
dic['우주탐사학과'] = '경희대학교 우주탐사학과(School of Space Research /KHU)는 교육과학기술부 제 1유형의 세계수준의 연구중심 대학 육성(WCU)사업에 달궤도 우주 탐사 연구 과제가 선정됨에 따라 설립된 대학원 학과로서 우리나라의 우주탐사를 위하여 본격적으로 전문인력 양성의 기틀을 마련하고자 한다. 한국 정부의 대학 교육 지원 과정의 세계 수준의 연구중심 대학(WCU) 육성사업을 통해 연구 역량이 높은 우수 해외 학자를 유치 활용하여, 국내 대학과 협력하여 핵심 성장 동력을 창출할 수 있는 분야의 연구를 활성화 하는데 그 목적이 있다. 다양한 프로젝트를 통해 국가적 발전을 선도하는 신 성장 동력을 창출하는 기술을 개발하는데 집중하고 있으며, 기초과학, 인문과학, 사회과학의 학제적 통합을 통해 학계 및 사회, 국가적 발전에 기여 할 수 있도록 정부에서 적극적으로 추진하고 있다.'

print(dic['물리학과'])

print(dic['우주과학과'])

print(dic['우주탐사학과'])


```

    물리학의 각 분야에 걸친 이론과 응용방법을 심오하게 교수, 연구함으로써 독창적 능력을 함양하고 고도 산업사회를 선도해 갈 지도적 인재를 양성함을 목적으로 한다.
    우주과학과는 천체 및 우주에서 일어나는 제반 현상을 과학적으로 탐사하고 연구하는 학과이다. 본 학과는 인류의 우주진출이 더욱 활발해 지고 있는 이 시대에 그를 위한 지식과 기술의 개발과 보급을 목적으로 설립되었다. 현대 천문학에서부터 인공위성과 우주선의 활용에 이르는 기초와 응용의 병행 학습을 통하여 21세기 우주 시대가 요구하는 첨단분야에서 국제적인 경쟁력이 있는 인재를 양성하는 데에 우주과학과의 교육 목적이 있다.
    경희대학교 우주탐사학과(School of Space Research /KHU)는 교육과학기술부 제 1유형의 세계수준의 연구중심 대학 육성(WCU)사업에 달궤도 우주 탐사 연구 과제가 선정됨에 따라 설립된 대학원 학과로서 우리나라의 우주탐사를 위하여 본격적으로 전문인력 양성의 기틀을 마련하고자 한다. 한국 정부의 대학 교육 지원 과정의 세계 수준의 연구중심 대학(WCU) 육성사업을 통해 연구 역량이 높은 우수 해외 학자를 유치 활용하여, 국내 대학과 협력하여 핵심 성장 동력을 창출할 수 있는 분야의 연구를 활성화 하는데 그 목적이 있다. 다양한 프로젝트를 통해 국가적 발전을 선도하는 신 성장 동력을 창출하는 기술을 개발하는데 집중하고 있으며, 기초과학, 인문과학, 사회과학의 학제적 통합을 통해 학계 및 사회, 국가적 발전에 기여 할 수 있도록 정부에서 적극적으로 추진하고 있다.



```python
# -*- coding: utf8 -*-
 
# 유니코드로 다루기 예제1
hoo = unicode('한글', 'utf-8')
print str(hoo.encode('utf-8'))
 
# 유니코드로 다루기 예제2
bar = '한글'.decode('utf-8')
print bar.encode('utf-8')
 
# 유니코드로 다루기 예제3
foo = u'한글'
print str(foo.encode('utf-8'))
```


```python
for item in dic.keys():
    print(item)
    
for item in dic.values():
    print(item)
```

    우주탐사학과
    물리학과
    우주과학과
    경희대학교 우주탐사학과(School of Space Research /KHU)는 교육과학기술부 제 1유형의 세계수준의 연구중심 대학 육성(WCU)사업에 달궤도 우주 탐사 연구 과제가 선정됨에 따라 설립된 대학원 학과로서 우리나라의 우주탐사를 위하여 본격적으로 전문인력 양성의 기틀을 마련하고자 한다. 한국 정부의 대학 교육 지원 과정의 세계 수준의 연구중심 대학(WCU) 육성사업을 통해 연구 역량이 높은 우수 해외 학자를 유치 활용하여, 국내 대학과 협력하여 핵심 성장 동력을 창출할 수 있는 분야의 연구를 활성화 하는데 그 목적이 있다. 다양한 프로젝트를 통해 국가적 발전을 선도하는 신 성장 동력을 창출하는 기술을 개발하는데 집중하고 있으며, 기초과학, 인문과학, 사회과학의 학제적 통합을 통해 학계 및 사회, 국가적 발전에 기여 할 수 있도록 정부에서 적극적으로 추진하고 있다.
    물리학의 각 분야에 걸친 이론과 응용방법을 심오하게 교수, 연구함으로써 독창적 능력을 함양하고 고도 산업사회를 선도해 갈 지도적 인재를 양성함을 목적으로 한다.
    우주과학과는 천체 및 우주에서 일어나는 제반 현상을 과학적으로 탐사하고 연구하는 학과이다. 본 학과는 인류의 우주진출이 더욱 활발해 지고 있는 이 시대에 그를 위한 지식과 기술의 개발과 보급을 목적으로 설립되었다. 현대 천문학에서부터 인공위성과 우주선의 활용에 이르는 기초와 응용의 병행 학습을 통하여 21세기 우주 시대가 요구하는 첨단분야에서 국제적인 경쟁력이 있는 인재를 양성하는 데에 우주과학과의 교육 목적이 있다.



```python
'철학과' in dic
```




    False




```python
'우주탐사학과' in dic
```




    True




```python
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print d['cat']       # Get an entry from a dictionary; prints "cute"
print 'cat' in d     # Check if a dictionary has a given key; prints "True"
```


```python

```


```python
d['fish'] = 'wet'    # Set an entry in a dictionary
print d['fish']      # Prints "wet"
```


```python
print d['monkey']  # KeyError: 'monkey' not a key of d
```


```python
print d.get('monkey', 'N/A')  # Get an element with a default; prints "N/A"
print d.get('fish', 'N/A')    # Get an element with a default; prints "wet"
```


```python
del d['fish']        # Remove an element from a dictionary
print d.get('fish', 'N/A') # "fish" is no longer a key; prints "N/A"
```

It is easy to iterate over the keys in a dictionary:


```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    print(animal)
```


```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
```


```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.iteritems():
    print('A %s has %d legs' % (animal, legs))
```


```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)
```

#### Sets


```python
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"

```


```python
animals.add('fish')      # Add an element to a set
print('fish' in animals)
print(len(animals))       # Number of elements in a set;
```


```python
animals.add('cat')       # Adding an element that is already in the set does nothing
print(len(animals))
animals.remove('cat')    # Remove an element from a set
print(len(animals))
```

_Loops_: Iterating over a set has the same syntax as iterating over a list; however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:


```python
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print ('#%d: %s' % (idx + 1, animal))
# Prints "#1: fish", "#2: dog", "#3: cat"
```

Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:


```python
from math import sqrt

nums = {int(sqrt(x)) for x in range(30)} 

print(nums)
```

#### 튜플

A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:


```python
a = 3
b = 7

temp = a
a = b
b = temp

print a, b
```


```python
a = 3
b = 7
a, b = b, a
print a, b
```


```python
t1 = (1, 3, 5)
t2 = 2, 4, 6

print(type(t1))
print(type(t2))

print(t1)
print(t2)
```


```python
t3 = ()

print(type(t3))
print(t3)
```


```python
t4 = 1,
t5 = (2)
t6 = (2,)

print(type(t4))
print(type(t5))
print(type(t6))
print(t4)
print(t5)
print(t6)
```


```python
p = (1, 2, 3)
print(p[:1])
print(p[2:])
q = p[:1] + (5,) + p[2:]
print(q)

r = p[:1], 5, p[2:]
print(r)
```


```python
t = (1, 2, 3)
print(t)
l = list(t)
print(l)
t2 = tuple(l)
print(t2)

```

 바로 기술적인 차이와 문화적인 차이다.
 
 둘 다 타입과 상관 없이 일련의 요소(element)를 갖을 수 있다. 두 타입 모두 요소의 순서를 관리한다. (세트(set)나 딕셔너리(dict)와 다르게 말이다.)

이제 차이점을 보자. 리스트와 튜플의 기술적 차이점은 불변성에 있다. 리스트는 가변적(mutable, 변경 가능)이며 튜플은 불변적(immutable, 변경 불가)이다. 이 특징이 파이썬 언어에서 둘을 구분하는 유일한 차이점이다.

이 특징은 리스트와 튜플을 구분하는 유일한 기술적 차이점이지만 이 특징이 나타나는 부분은 여럿 존재한다. 예를 들면 리스트에는 .append() 메소드를 사용해서 새로운 요소를 추가할 수 있지만 튜플은 불가능하다.

튜플은 .append() 메소드가 필요하지 않다. 튜플은 수정할 수 없기 때문이다.

문화적인 차이점을 살펴보자. 리스트와 튜플을 어떻게 사용하는지에 따른 차이점이 있다. 리스트는 단일 종류의 요소를 갖고 있고 그 일련의 요소가 몇 개나 들어 있는지 명확하지 않은 경우에 주로 사용한다. 튜플은 들어 있는 요소의 수를 사전에 정확히 알고 있을 경우에 사용한다. 동일한 요소가 들어있는 리스트와 달리 튜플에서는 각 요소의 위치가 큰 의미를 갖고 있기 때문이다.

디렉토리 내에 있는 파일 중 *.py로 끝나는 파일을 찾는 함수를 작성한다고 가정해보자. 이 함수를 사용했을 때는 파일을 몇 개나 찾게 될 지 알 수 없다. 그리고 동일한 규칙으로 찾은 파일이기 때문에 항목 하나 하나가 의미상 동일하다. 그러므로 이 함수는 리스트를 반환할 것이다.

>>> find_files("*.py")
["control.py", "config.py", "cmdline.py", "backward.py"]
다른 예를 확인한다. 기상 관측소의 5가지 정보, 식별번호, 도시, 주, 경도와 위도를 저장한다고 생각해보자. 이런 상황에서는 리스트보다 튜플을 사용하는 것이 적합하다.

>>> denver = (44, "Denver", "CO", 40, 105)
>>> denver[1]
'Denver'
(지금은 클래스를 사용하는 것에 대해서 이야기하지 않을 것이다.) 이 튜플에서 첫 요소는 식별번호, 두 번째는 도시… 순으로 작성했다. 튜플에서의 위치가 담긴 내용이 어떤 정보인지를 나타낸다.

C 언어에서 이 문화적 차이를 대입해보면 목록은 배열(array) 같고 튜플은 구조체(struct)와 비슷할 것이다.

때때로 기술적인 고려가 문화적 고려를 덮어쓰는 경우가 있다. 리스트를 딕셔너리에서 키로 사용할 수 없다. 불변 값만 해시를 만들 수 있기 때문에 키에 불변 값만 사용 가능하다. 대신 리스트를 키로 사용하고 싶다면 다음 예처럼 리스트를 튜플로 변경했을 때 사용할 수 있다.

>>> d = {}
>>> nums = [1, 2, 3]
>>> d[nums] = "hello"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unhashable type: 'list'
>>> d[tuple(nums)] = "hello"
>>> d
{(1, 2, 3): 'hello'}
기술과 문화가 충돌하는 또 다른 예가 있다. 파이썬에서도 리스트가 더 적합한 상황에서 튜플을 사용하는 경우가 있다. *args를 함수에서 정의했을 때, args로 전달되는 인자는 튜플을 사용한다. 함수를 호출할 때 사용한 인자의 순서가 크게 중요하지 않더라도 말이다. 튜플은 불변이고 전달된 값은 변경할 수 없기 때문에 이렇게 구현되었다고 말할 수 있겠지만 그건 문화적 차이보다 기술적 차이에 더 가치를 두고 설명하는 방식이라 볼 수 있다.

물론 *args에서 위치는 매우 큰 의미를 갖는다. 매개변수는 그 위치에 따라 의미가 크게 달라지기 때문이다. 하지만 함수는 *args를 전달 받고 다른 함수에 전달해준다고만 봤을 때 *args는 단순히 인자 목록이고 각 인자는 별 다른 의미적 차이가 없다고 할 수 있다. 그리고 각 함수에서 함수로 이동할 때마다 그 목록의 길이는 가변적인 것으로 볼 수 있다.

파이썬이 여기서 튜플을 사용하는 이유는 리스트에 비해서 조금 더 공간 효율적이기 때문이다. 리스트는 요소를 추가하는 동작을 빠르게 수행할 수 있도록 더 많은 공간을 저장해둔다. 이 특징은 파이썬의 실용주의적 측면을 나타낸다. 이런 상황처럼 *args를 두고 리스트인지 튜플인지 언급하기 어려운 애매할 때는 그냥 상황을 쉽게 설명할 수 있도록 자료 구조(data structure)라는 표현을 쓰면 될 것이다.

대부분의 경우에 리스트를 사용할지, 튜플을 사용할지는 문화적 차이에 기반해서 선택하게 될 것이다. 어떤 의미의 데이터인지 생각해보자. 만약 프로그램이 실제로 다루는 자료가 다른 길이의 데이터를 갖는다면 분명 리스트를 써야 할 것이다. 작성한 코드에서 세 번째 요소에 의미가 있는 경우라면 분명 튜플을 사용해야 할 상황이다.

반면 함수형 프로그래밍에서는 코드를 어렵게 만들 수 있는 부작용을 피하기 위해서 불변 데이터 구조를 사용하라고 강조한다. 만약 함수형 프로그래밍의 팬이라면 튜플이 제공하는 불변성 때문에라도 분명 튜플을 좋아하게 될 것이다.

자, 다시 질문해보자. 튜플을 써야 할까, 리스트를 사용해야 할까? 이 질문의 답변은 항상 간단하지 않다.


```python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print type(t)
print d[t]       
print d[(1, 2)]
```


```python
t[0] = 1
```

### 함수


```python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print sign(x)
```

We will often define functions to take optional keyword arguments, like this:


```python
def hello(name, loud=False):
    if loud:
        print 'HELLO, %s' % name.upper()
    else:
        print 'Hello, %s!' % name

hello('Bob')
hello('Fred', loud=True)
```


```python
today = '20170811'

def curr_order(idx):
    print(today + ' ' + str(idx))
    
curr_order(1)
curr_order(2)
```


```python
def sum(a, b):
    return a + b

print(sum(1,2))
```


```python
seed = 3

def updown(mind, guess):
    if mind < guess:
        return 'down'
    elif mind > guess:
        return 'up'
    else:
        return 'correct'

updown(seed, 3)
```


```python
def tuple_test(a, b, *c):
    print a, b, c

tuple_test(1, 2, 3, 4, 5)
```

### 클래스

The syntax for defining classes in Python is straightforward:


```python
class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```


```python
class USB:
    def poweron(self):
        pass
    
class FAN(USB):
    def poweron(self):
        print('wing~~')
        
class Cup(USB):
    def poweron(self):
        print('cool')
        
class Phone(USB):
    def poweron(self):
        print('charging...')
        
f = FAN()
f.poweron()

c = Cup()
c.poweron()

p = Phone()
p.poweron()
```

    wing~~
    cool
    charging...



```python
class USBhub(USB):
    
    def __init__(self):
        self.ports = []
        
    def add(self, USB):
        self.ports.append(USB)
    
    def poweron(self):
        for item in self.ports:
            item.poweron()

hub = USBhub()
hub.add(f)
hub.add(c)
hub.add(p)
hub.poweron()
```

    wing~~
    cool
    charging...



```python
class Man:
    def think(self):
        pass
    
class Android(Man, USB):
    def think(self):
        print('Who am I?')
        
    def poweron(self):
        print('Hello')

a = Android()
a.think()
a.poweron()
```

    Who am I?
    Hello



```python
hub.add(a)
hub.poweron()
```

    wing~~
    cool
    charging...
    Hello



```python
>>> class ParentOne:
    def func(self):
        print("ParentOne의 함수 호출!")
     
>>> class ParentTwo:
    def func(self):
        print("ParentTwo의 함수 호출!")
     
>>> class Child(ParentOne, ParentTwo):
    def childFunc(self):
        ParentOne.func(self)
        ParentTwo.func(self)
         
>>> objectChild = Child()
>>> objectChild.childFunc()
ParentOne의 함수 호출!
ParentTwo의 함수 호출!
>>> objectChild.func()
ParentOne의 함수 호출!


출처: http://blog.eairship.kr/286 [누구나가 다 이해할 수 있는 프로그래밍 첫걸음]
```

#### 모듈


```python
import 모듈 
from 모듈 import 변수
from 모듈 import 함수
```


```python
import 모듈

모듈.함수
```


```python
from 모듈 import *

함수

함수명이 동일할 때는 곤란
```


```python
import os

os.getcwd()
```




    '/Users/tykimos/Projects/Keras/_writing'




```python
os.listdir(os.getcwd())
```




    ['.DS_Store',
     '.ipynb_checkpoints',
     '2017-1-27-CNN_Layer_Talk.ipynb',
     '2017-1-27-Keras_Talk.ipynb',
     '2017-1-27-LossFuncion_Talk.ipynb',
     '2017-1-27-MLP_Layer_Talk.ipynb',
     '2017-1-27-Optimizer_Talk.ipynb',
     '2017-2-22-Integrating_Keras_and_TensorFlow.ipynb',
     '2017-2-4-AutoEncoder_Getting_Started.ipynb',
     '2017-2-4-BinaryClassification_Example.ipynb',
     '2017-2-4-ImageClassification_Example.ipynb',
     '2017-2-4-MLP_Getting_Started-Copy1.ipynb',
     '2017-2-4-MLP_Getting_Started.ipynb',
     '2017-2-4-MulticlassClassification_Example.ipynb',
     '2017-2-4-ObjectRecognition_Example.ipynb',
     '2017-2-4-Regression_Example.ipynb',
     '2017-2-4-RNN_Getting_Started.ipynb',
     '2017-2-4-TimeSeriesPrediction_Example.ipynb',
     '2017-2-6-First_Keras_Offline_Meeting.ipynb',
     '2017-3-11-To_Use_TensorBoard.ipynb',
     '2017-3-15-Keras_Offline_Install.ipynb',
     '2017-3-25-Dataset_and_Fit_Talk.ipynb',
     '2017-3-8-CNN_Data_Augmentation.ipynb',
     '2017-3-8-CNN_Getting_Started.ipynb',
     '2017-4-9-RNN_Getting_Started_2.ipynb',
     '2017-4-9-RNN_Layer_Talk.ipynb',
     '2017-5-20-LSTM_Example_Feeding_Regression-Copy1.ipynb',
     '2017-5-20-LSTM_Example_Feeding_Regression.ipynb',
     '2017-5-21-Conv_LSTM_Example.ipynb',
     '2017-5-22-Evaluation_Talk.ipynb',
     '2017-6-10-Model_Save_Load.ipynb',
     '2017-6-17-Relation_Network.ipynb',
     '2017-7-9-Early_Stopping.ipynb',
     '2017-7-9-Training_Monitoring.ipynb',
     '2017-8-10-Python_Package_Talk.ipynb',
     '2017-8-10-Python_Talk-Copy1.ipynb',
     '2017-8-10-Python_Talk.ipynb',
     '2017-8-4-RNN_Classification.ipynb',
     '2017-8-7-Keras_Install_on_Mac.ipynb',
     '2017-8-9-DeepBrick_Talk.ipynb',
     'abstract',
     'Animate.ipynb',
     'cosine_LSTM-Copy1.ipynb',
     'cosine_LSTM-Copy2.ipynb',
     'cosine_LSTM-Copy3.ipynb',
     'cosine_LSTM-Copy4.ipynb',
     'cosine_LSTM-flux.ipynb',
     'cosine_LSTM.ipynb',
     'Data_RNN.zip',
     'exAnimation.gif',
     'FeedPrediction_DeepStackedStatefulLSTM.ipynb',
     'Flare_Flux_Prediction.ipynb',
     'Flux Case 1.ipynb',
     'Flux Case 2.ipynb',
     'Flux_deep_stacked_stateful_LSTM_with_one_sample.ipynb',
     'Flux_Test-Copy1.ipynb',
     'Flux_Test.ipynb',
     'Flux_Test_Stateful.ipynb',
     'FullSizeRender.jpg',
     'graph',
     'HEPFluxPrediction_DeepStackedStatefulLSTM-Copy1.ipynb',
     'HEPFluxPrediction_DeepStackedStatefulLSTM.ipynb',
     'HEPFluxPrediction_DeepStackedStatefulLSTM_v200-Copy1.ipynb',
     'HEPFluxPrediction_DeepStackedStatefulLSTM_v200.ipynb',
     'image.png',
     'lecture.ipynb',
     'LSTM.py',
     'model.png',
     'object detector.ipynb',
     'sin_w40_u32_s2_e200.gif',
     'SPE_Prediction.ipynb',
     'stateful RNNs.ipynb',
     'tykimos.txt',
     'Untitled.ipynb',
     'w12_u64_s2_e300.gif',
     'w24_u128_s1_e100.gif',
     'w40_u128_s2_e200.gif',
     'w40_u128_s4_e1000.gif',
     'w40_u32_s2_e1.gif',
     'warehouse']




```python
os.rename('tykimos.txt', 'tykimos2.txt')
```


```python
os.listdir(os.getcwd())
```




    ['.DS_Store',
     '.ipynb_checkpoints',
     '2017-1-27-CNN_Layer_Talk.ipynb',
     '2017-1-27-Keras_Talk.ipynb',
     '2017-1-27-LossFuncion_Talk.ipynb',
     '2017-1-27-MLP_Layer_Talk.ipynb',
     '2017-1-27-Optimizer_Talk.ipynb',
     '2017-2-22-Integrating_Keras_and_TensorFlow.ipynb',
     '2017-2-4-AutoEncoder_Getting_Started.ipynb',
     '2017-2-4-BinaryClassification_Example.ipynb',
     '2017-2-4-ImageClassification_Example.ipynb',
     '2017-2-4-MLP_Getting_Started-Copy1.ipynb',
     '2017-2-4-MLP_Getting_Started.ipynb',
     '2017-2-4-MulticlassClassification_Example.ipynb',
     '2017-2-4-ObjectRecognition_Example.ipynb',
     '2017-2-4-Regression_Example.ipynb',
     '2017-2-4-RNN_Getting_Started.ipynb',
     '2017-2-4-TimeSeriesPrediction_Example.ipynb',
     '2017-2-6-First_Keras_Offline_Meeting.ipynb',
     '2017-3-11-To_Use_TensorBoard.ipynb',
     '2017-3-15-Keras_Offline_Install.ipynb',
     '2017-3-25-Dataset_and_Fit_Talk.ipynb',
     '2017-3-8-CNN_Data_Augmentation.ipynb',
     '2017-3-8-CNN_Getting_Started.ipynb',
     '2017-4-9-RNN_Getting_Started_2.ipynb',
     '2017-4-9-RNN_Layer_Talk.ipynb',
     '2017-5-20-LSTM_Example_Feeding_Regression-Copy1.ipynb',
     '2017-5-20-LSTM_Example_Feeding_Regression.ipynb',
     '2017-5-21-Conv_LSTM_Example.ipynb',
     '2017-5-22-Evaluation_Talk.ipynb',
     '2017-6-10-Model_Save_Load.ipynb',
     '2017-6-17-Relation_Network.ipynb',
     '2017-7-9-Early_Stopping.ipynb',
     '2017-7-9-Training_Monitoring.ipynb',
     '2017-8-10-Python_Package_Talk.ipynb',
     '2017-8-10-Python_Talk-Copy1.ipynb',
     '2017-8-10-Python_Talk.ipynb',
     '2017-8-4-RNN_Classification.ipynb',
     '2017-8-7-Keras_Install_on_Mac.ipynb',
     '2017-8-9-DeepBrick_Talk.ipynb',
     'abstract',
     'Animate.ipynb',
     'cosine_LSTM-Copy1.ipynb',
     'cosine_LSTM-Copy2.ipynb',
     'cosine_LSTM-Copy3.ipynb',
     'cosine_LSTM-Copy4.ipynb',
     'cosine_LSTM-flux.ipynb',
     'cosine_LSTM.ipynb',
     'Data_RNN.zip',
     'exAnimation.gif',
     'FeedPrediction_DeepStackedStatefulLSTM.ipynb',
     'Flare_Flux_Prediction.ipynb',
     'Flux Case 1.ipynb',
     'Flux Case 2.ipynb',
     'Flux_deep_stacked_stateful_LSTM_with_one_sample.ipynb',
     'Flux_Test-Copy1.ipynb',
     'Flux_Test.ipynb',
     'Flux_Test_Stateful.ipynb',
     'FullSizeRender.jpg',
     'graph',
     'HEPFluxPrediction_DeepStackedStatefulLSTM-Copy1.ipynb',
     'HEPFluxPrediction_DeepStackedStatefulLSTM.ipynb',
     'HEPFluxPrediction_DeepStackedStatefulLSTM_v200-Copy1.ipynb',
     'HEPFluxPrediction_DeepStackedStatefulLSTM_v200.ipynb',
     'image.png',
     'lecture.ipynb',
     'LSTM.py',
     'model.png',
     'object detector.ipynb',
     'sin_w40_u32_s2_e200.gif',
     'SPE_Prediction.ipynb',
     'stateful RNNs.ipynb',
     'tykimos2.txt',
     'Untitled.ipynb',
     'w12_u64_s2_e300.gif',
     'w24_u128_s1_e100.gif',
     'w40_u128_s2_e200.gif',
     'w40_u128_s4_e1000.gif',
     'w40_u32_s2_e1.gif',
     'warehouse']




```python
import webbrowser
url = 'http://www.google.com'
webbrowser.open(url)
```




    True




```python
### 랜덤

import random

random.random()
```




    0.5919179034090589




```python
random.randrange(1, 7)
```




    4




```python
range(1, 7)
```




    [1, 2, 3, 4, 5, 6]




```python
abc = ['a', 'b', 'c', 'd', 'e']

random.shuffle(abc)

abc
```




    ['b', 'e', 'c', 'd', 'a']




```python
random.choice(abc)
```




    'b'




```python
random.choice([True, False])
```




    True



#### 파일


```python
lines = ['1. first\n', '2. second\n', '3. third\n']

f = open('text.txt', 'w')
f.writelines(lines)
f.close()
```


```python
f = open('text.txt')
print(f.readline())
print(f.readline())
print(f.readline())
f.close()
```

    1. first
    
    2. second
    
    3. third
    



```python
f = open('text.txt')
print(f.readlines())
f.close()
```

    ['1. first\n', '2. second\n', '3. third\n']



```python
f = open('text.txt')
lines = f.readlines()

import sys
sys.stdout.writelines(lines)
```

    1. first
    2. second
    3. third



```python

```
