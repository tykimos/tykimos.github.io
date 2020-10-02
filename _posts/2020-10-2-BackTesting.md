---
layout: post
title:  "백테스팅"
author: 김태영 
date:   2020-10-2 12:00:00
categories: etc
comments: true
image: http://tykimos.github.io/warehouse/2020-10-2-BackTesting_title2.png
---

투자 투자전략을 투자전략을 세웠다면 이 투자전략으로 바로 실전에 투입하기 전에 과거 데이터로 한 번 검증을 해봐야합니다. 이렇게 과거데이터를 이용해서 투자전략을 테스트해보는 것을 백테스팅이라고 부릅니다. 백테스팅을 위한 소프트웨어나 패키지들이 있는데요, 주요 가능은 다음과 같습니다.

* 초기 투자 금액 설정
* 시작일, 종료일 설정
* 매매, 수익 정보 제공

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_title1.png)

여러 패키지 중 백트레이더(Backtrader)를 알아보도록 하겠습니다.

---
### 백트레이더

백트레이더 홈페이지 가면 문서와 함께 샘플 예제가 보입니다.

![img](http://tykimos.github.io/warehouse/2020-10-2-BackTesting_2.png)

---
### 백트레이더 기본예제 구동하기

백트레이더에서 제공되는 기본 예제를 구동해보겠습니다.

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

예제코드는 크게 투자전략 클래스 정의, 백테스팅 설정, 실행 및 결과확인으로 되어 있습니다.

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

---
### 투자전략 바꿔보기

투자전략 클래스를 상속받아 필요한 부분만 구현하면 쉽게 백테스팅 기능을 활용할 수 있습니다. 아래 예제에서는 "상대적 강도 지수(RSI)" 젼략을 클래스로 만들고 이를 백테스팅 수행한 것입니다.

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
``
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
### 참고

* [발표자료](https://docs.google.com/presentation/d/1h8WreG-PPdF2iDy0iJbGjRvxDdrg2cPgUKQ4kW0AUF8/edit?usp=sharing)
* 파이썬 증권 데이터 분석, 김황후 지음, 한빛미디어
* [금융에 딥러닝 적용해보기 - 시스템 트레이딩, 황준원(이스트소프트)](http://aifactory.space/dld/task/detail.do?taskId=T000052)

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

