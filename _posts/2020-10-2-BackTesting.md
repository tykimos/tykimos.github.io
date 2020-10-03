---
layout: post
title:  "백테스팅"
author: 김태영 
date:   2020-10-2 12:00:00
categories: etc
comments: true
image: http://tykimos.github.io/warehouse/2020-10-2-BackTesting_title3.png
---

주식 투자에 대해 공부를 하다보면, 여러가지 투자전략에 대해서 배우게 됩니다. 자신의 투자 경향, 관심 종목, 접근할 수 있는 정보 권한 등으로 자신만의 투자전략을 세우게 됩니다. 투자 전략에 있어서 주로 차트와 시장 가격을 분석하는 기술적 분석과 재무재표, 경영 등 가격 결정의 원인을 분석하는 기본적 분석이 있습니다. 이번에는 기술적 분석에 국한하여 몇 가지 전략을 적용해보겠습니다. 투자전략을 세웠다면 이 투자전략으로 바로 실전에 투입하기 전에 과거 데이터로 한 번 검증을 해봐야합니다. 이렇게 과거데이터를 이용해서 투자전략을 테스트해보는 것을 백테스팅이라고 부릅니다. 백테스팅 툴은 퀀트들에게 전략 및 지표 분석에 집중할 수 있도록 백테스팅 및 트래이딩 환경을 제공합니다.

    퀀트란 인공지능(혹은 수학 및 통계정보)를 활용하여 투자 전략을 세우는 사람을 의미합니다.

백테스팅을 위한 소프트웨어나 패키지들이 있는데요, 주요 가능은 다음과 같습니다.

* 초기 투자 금액 설정
* 시작일, 종료일 설정
* 매매, 수익 정보 제공

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_title3.png)

백테스팅 기능이 지원되는 여러가지 툴이 있지만 이 중 백트레이더(Backtrader)를 알아보도록 하겠습니다. 자 그럼 내가 짠 투자전략 알고리즘이 과거데이터로 얼마나 수익을 낼 수 있는 지 확인해볼까요?

---
### 백트레이더

백트레이더 홈페이지 가면 문서와 함께 샘플 예제가 보입니다. 

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_2.png)

---
### 백트레이더 기본예제 구동하기

백트레이더에서 제공되는 기본 예제를 구동해보겠습니다. 소스코드가 그리 길지 않습니다. 자신만의 투자전략을 파이썬 프로그램으로 표현만 할 수 있다면 과거 데이터로 쉽게 백테스팅을 할 수 있다니 벌써 기대가 되네요.

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_2.png)

실습은 코랩에서 해보도록 하겠습니다. 코랩에서 파이썬 패키지를 설치를 pip로 하기 위해서는 터미널 명령임을 알리기 위해서 느낌표(!)를 명령 앞에 붙여서 셀을 실행시킵니다.

```python
# 백트레이더 설치
!pip install backtrader
```

    Collecting backtrader
    Downloading https://files.pythonhosted.org/packages/1a/bf/78aadd993e2719d6764603465fde163ba6ec15cf0e81f13e39ca13451348/backtrader-1.9.76.123-py2.py3-none-any.whl (410kB)
        |████████████████████████████████| 419kB 2.8MB/s 
    Installing collected packages: backtrader
    Successfully installed backtrader-1.9.76.123

예제코드는 크게 투자전략 클래스 정의, 백테스팅 설정, 실행 및 결과확인으로 되어 있습니다. 예제에서는 SMA(Simple Moving Average, 단순이동평균)을 이용해서 교차점을 구하고 이때 매매가 일어나도록 전략을 구사했네요. SMA에 대해서는 잠시 후에 설명 드리겠습니다.

```python
from datetime import datetime
import backtrader as bt

# Create a subclass of Strategy to define the indicators and logic

class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position


cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

# Create a data feed
data = bt.feeds.YahooFinanceData(dataname='MSFT',
                                 fromdate=datetime(2011, 1, 1),
                                 todate=datetime(2012, 12, 31))

cerebro.adddata(data)  # Add the data feed

cerebro.addstrategy(SmaCross)  # Add the trading strategy
cerebro.run()  # run it all
cerebro.plot()  # and plot it with a single command
```
    [[<Figure size 432x288 with 5 Axes>]]

차트 결과는 코랩에서 바로 출력이 되지 않아서 아래처럼 이미지 파일로 저장한 다음 이미지 파일을 화면에 표시하도록 하였습니다.

```python
# 차트를 코랩에서 바로 출력
from IPython.display import display, Image
cerebro.plot()[0][0].savefig('plot.png', dpi=100)
display(Image(filename='plot.png'))
```

예제 코드 차트 결과는 아래와 같습니다.

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_5.png)

코랩에서 바로 실습을 해보시려면 아래 링크로 접속하세요.
* [코랩 소스코드 - 백테스팅_1_기본예제](https://colab.research.google.com/drive/1XCdegVNsa451ecv91HDpu640xhnblrKI?authuser=1#scrollTo=yM4DaIOA2nPL)

일단 예제 코드가 제대로 동작됨을 확인해봤으니 내가 원하는 종목으로 백테스팅을 해보겠습니다.

---
### 국내종목으로 바꿔보기

국내종목으로 바꿔보기 위해서는 먼저 국내종목코드를 알아야합니다. 검색엔진에서 "회사명" + "주가" 혹은 "종목"으로 검색하면 쉽게 종목코드를 확인할 수 있습니다.

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_3.png)

백트레이더에서 제공하는 야후 금융 기능을 사용할 예정입니다. 위에서 검색한 종목이 야후 금융에도 동일하게 등록되어 있는 지 확인합니다. 본 예제에서는 엔씨소프트로 예시들어 보겠습니다.

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_4.png)

```python
# 백트레이더 설치
!pip install backtrader
```
    Collecting backtrader
    Downloading https://files.pythonhosted.org/packages/1a/bf/78aadd993e2719d6764603465fde163ba6ec15cf0e81f13e39ca13451348/backtrader-1.9.76.123-py2.py3-none-any.whl (410kB)
        |████████████████████████████████| 419kB 2.5MB/s 
    Installing collected packages: backtrader
    Successfully installed backtrader-1.9.76.123

```python
# 0. 필요 패키지 가져오기

from datetime import datetime
import backtrader as bt
from IPython.display import display, Image

```

기본 예제에서 적용된 전략은 단순 이동 평균(SMA) 10일과 30일짜리 두 개를 이용하여, 서로 교차되는 지점에서 매매 타이밍을 잡는 방법입니다.

```python
# 1. 전략 클래스 정의

class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position
```

bt.feeds.YahooFinanceData() 함수의 dataname 인자에서 위에서 확인한 국내종목코드를 입력합니다. 그리고 추가로 브로커(broker) 겍체의 setcash() 함수를 이용하여 현재 투자금액을 설정합니다. 한국 통화(원)은 단위가 높으므로 백트레이더의 기본 설정으로는 국내 주식 구매가 힘들기 때문에 백테스팅 수행 전에 투자금액 설정과 매매 단위 설정부분은 확인합니다.

```python
# 2. 세레브로(백트레이더의 엔진) 설정

# 세레브로 가져오기
cerebro = bt.Cerebro()

# 야후 금융 데이터 불러오기
data = bt.feeds.YahooFinanceData(dataname='036570.KS', # 엔씨소프트
                                 fromdate=datetime(2017, 1, 1), # 시작일
                                 todate=datetime(2020, 10, 1)) # 종료일

# 데이터 추가하기
cerebro.adddata(data)

# 전략 추가하기
cerebro.addstrategy(SmaCross)  # Add the trading strategy

# 브로거 설정
cerebro.broker.setcash(10000000)

# 매매 단위 설정하기
cerebro.addsizer(bt.sizers.SizerFix, stake=30) # 한번에 30주 설정
```

```python
# 3. 세레브로 실행하기

# 초기 투자금 가져오기
init_cash = cerebro.broker.getvalue()

# 세레브로 실행하기
cerebro.run()

# 최종 금액 가져오기
final_cash = cerebro.broker.getvalue()

print("최종금액 : ", final_cash, "원")
print("수익률 : ", float(final_cash - init_cash)/float(init_cash) * 100., "%")

# 차트 출력하기
cerebro.plot()[0][0].savefig('plot.png', dpi=100)
display(Image(filename='plot.png'))
```
    최종금액 :  11431054.299999999 원
    수익률 :  14.31054299999999 %

간단한 SMA 전략으로 14프로 이상의 수익률이 나왔습니다.

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_6.png)

코랩에서 바로 실습을 해보시려면 아래 링크로 접속하세요.
* [코랩 소스코드 - 백테스팅_2_국내종목으로 바꿔보기](https://colab.research.google.com/drive/1UZNEuJ7zPH-5-tr-DyWBxwF_wh9FjUtP?authuser=1#scrollTo=-E5ZGN3E9bUG)

다음으로 넘어가기 전에 SMA 크로스 전략에 대해서 좀 더 살펴보겠습니다.

#### 단순 이동 평균(Simple Moving Average, SMA)

먼저 단순 이동 평균(이하 SMA)은 며칠 동안의 종가를 모두 합한 후 평균을 낸 것입니다. SMA(10)는 10일 동안 종가를 평균 낸 것이고, SMA(30)은 30일 동안 종가를 평균 낸 것입니다. 기간이 길면 길수록 가격변동을 부드럽게 보여주며, 추세를 보기 쉽습니다. 

#### SMA 크로스 전략

이러한 SMA를 이용해서 상승장과 하락장을 유추해볼 수 있습니다. 먼저 단기 SMA와 장기 SMA를 구한 뒤에 교차되는 지점을 구합니다. 이 예제에서는 단기 SMA를 10일 즉 SMA(10), 장기 SMA를 30일 즉 SMA(30)으로 설정했네요. 교차는 두가지로 나누어집니다.

* 골든크로스오버: 단기 SMA가 장기 SMA를 돌파하고 상승할 경우 상승장으로 판단
* 데쓰크로스오버: 장기 SMA가 단기 SMA를 돌차하고 상승할 경우 하락장으로 판단

즉 단기평균이 장기평균보다 높아지면(골든크로스오버) 오를 것으로 예측하고, 단기평균이 장기평균보다 낮아지면(데쓰크로스오버) 떨어질 것이라고 예측하는 겁니다. 따라서,

* 매수 시점: 골든크로스오버
* 매도 시점: 데쓰크로스오버

가 됩니다. SMA는 최신 정보에 대한 반영이 느릴 수 있는 데, 이러한 단점을 보완하기 위해서 가중평균이동평균(WMA), 지수이동평균(EMA) 등도 있으니 살펴보시기 바랍니다.

---
### 투자전략 바꿔보기

투자전략 클래스를 상속받아 필요한 부분만 구현하면 쉽게 백테스팅 기능을 활용할 수 있습니다. 이번에는 "상대적 강도 지수(RSI)" 젼략을 클래스로 만들고 이를 백테스팅 수행해보겠습니다. 

```python
# 백트레이더 설치
!pip install backtrader
```

    Collecting backtrader
    Downloading https://files.pythonhosted.org/packages/1a/bf/78aadd993e2719d6764603465fde163ba6ec15cf0e81f13e39ca13451348/backtrader-1.9.76.123-py2.py3-none-any.whl (410kB)
        |████████████████████████████████| 419kB 2.5MB/s 
    Installing collected packages: backtrader
    Successfully installed backtrader-1.9.76.123

```python
# 0. 필요 패키지 가져오기

from datetime import datetime
import backtrader as bt
from IPython.display import display, Image

```

```python
# 1.1 단순 이동 평균(SMA) 크로스 전략 클래스 정의

class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position
```
전략 추가하기 부분에서 쉽게 전략 클랙스를 교체할 수 있습니다.

```python
# 1.2 상대적 강도 지수(RSI) 전략 클래스 정의

class RSI(bt.Strategy):

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close)

    def next(self):
        if not self.position: # 아직 주식을 사지 않았다면
            if self.rsi < 30:
                self.order = self.buy()

        elif self.rsi > 70:
            self.order = self.sell()
```

이번에 정의한 RSI에 대해서 살펴보겠습니다.

#### 상대 강도 지수(Relative Strength Index, RSI)

RSI는 과매수, 과매도를 판단하기 위한 지표입니다. 과매수는 주식 상승으로 인해 많이 매수된 상태를 말하고, 과매도는 주식 하락으로 인해 많이 매도된 상태를 말합니다. 과매수 상태를 천정으로 유추하고 과매도를 바닥으로 유추한다. 이 RSI지수는 최소값이 0이고, 최대값이 100입니다. 직관적으로 추세의 강도를 의미하며, 높을수록 과매수 상태이고, 낮을수록 과매도 상태입니다. 계산법은 아래와 같습니다.

    RSI = 일정기간 동안의 상승폭 합계 / (일정기간 동안의 상승폭 합계 + 일정기간 동안의 하락폭 합계)

#### RSI 전략

* 과매도 : RSI가 30보다 떨어질 때
* 과매수 : RSI가 70보다 올라갈 때

즉 과매도의 의미는 일정기간 동안 상승한 폭보다 하락한 폭이 훨씬 크다는 것을 의미하고, 과매수의 의미는 일정기간 동안 하락한 폭보다 상승한 폭이 훨씬 크다는 것을 의미합니다. 과매도이면 바닥이라고 보고 매수 시점을 의미하고, 과매수이면 천정이라고 보고 매도 시점을 의미합니다.

* 매수 시점: 과매도
* 매도 시점: 과매수

RSI에서도 여러가지 파라미터가 있을 수 있는데, "일정기간"이 7일, 14일 등으로 설정할 수 있고, 과매도 및 과매수 임계값을 70, 30이 아니라, 80, 20으로 설정할 수 있습니다. 또한 매매 시점을 임계값을 초과할 때 바로 보는 것이 아니라 유지되다가 다시 탈피할 때라던지 등 여러가지 시점으로 볼 수 있습니다.

```python
# 2. 세레브로(백트레이더의 엔진) 설정

# 세레브로 가져오기
cerebro = bt.Cerebro()

# 야후 금융 데이터 불러오기
data = bt.feeds.YahooFinanceData(dataname='036570.KS', # 엔씨소프트
                                 fromdate=datetime(2017, 1, 1), # 시작일
                                 todate=datetime(2020, 10, 1)) # 종료일

# 데이터 추가하기
cerebro.adddata(data)

# 전략 추가하기
#cerebro.addstrategy(SmaCross) 
cerebro.addstrategy(RSI) 

# 브로거 설정
cerebro.broker.setcash(10000000)

# 매매 단위 설정하기
cerebro.addsizer(bt.sizers.SizerFix, stake=30) # 한번에 30주 설정
```

```python
# 3. 세레브로 실행하기

# 초기 투자금 가져오기
init_cash = cerebro.broker.getvalue()

# 세레브로 실행하기
cerebro.run()

# 최종 금액 가져오기
final_cash = cerebro.broker.getvalue()

print("최종금액 : ", final_cash, "원")
print("수익률 : ", float(final_cash - init_cash)/float(init_cash) * 100., "%")

# 차트 출력하기
cerebro.plot()[0][0].savefig('plot.png', dpi=100)
display(Image(filename='plot.png'))
```
    최종금액 :  12925000.0 원
    수익률 :  29.25 %

수익률이 30프로 가까이 나왔습니다. 투자전략에 따라 수익률 차이가 생겨남을 확인했습니다.

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_7.png)

코랩에서 바로 실습을 해보시려면 아래 링크로 접속하세요.
* [코랩 소스코드 - 백테스팅_3_투자전략 바꿔보기](https://colab.research.google.com/drive/1CoKnva5KrtFQwlw_dMz2OQRm2UqHtN0d?usp=sharing)

---
### 마무리

이번에는 자기가 만든 투자전략을 검증을 하기 위해서 과거데이터를 활용하여 백스팅하는 툴인 백트레이더 소개하였고, 기본 예제에서 국내 종목을 바꾸거나 투자 전략을 바꾸어서 수익률이 얼마나 나는 지 확인하였습니다.

---
### 자료

* [발표자료](https://docs.google.com/presentation/d/1h8WreG-PPdF2iDy0iJbGjRvxDdrg2cPgUKQ4kW0AUF8/edit?usp=sharing)

---
### 참고

* 파이썬 증권 데이터 분석, 김황후 지음, 한빛미디어
* [금융에 딥러닝 적용해보기 - 시스템 트레이딩, 황준원(이스트소프트)](http://aifactory.space/dld/task/detail.do?taskId=T000052)
* [기술적 분석 기초(1. Intro + SMA/EMA)](https://medium.com/@icehongssii/%EA%B8%B0%EC%88%A0%EC%A0%81-%EB%B6%84%EC%84%9D-%EA%B8%B0%EC%B4%88-1-intro-sma-ema-54368ac687db)

#### AIFactory

* [AI팩토리 머신러닝 경연대회](http://aifactory.space)

#### 인공지능 퀀트 코리아 커뮤니티

* [인공지능 퀀트 코리아 페북](https://www.facebook.com/groups/KerasKorea/)
* [인공지능 퀀트 코리아 단톡방](https://www.facebook.com/groups/KerasKorea/)

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

