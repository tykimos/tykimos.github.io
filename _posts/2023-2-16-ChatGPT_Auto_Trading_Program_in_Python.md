---
layout: post
title:  "챗GPT로 비트코인 자동매매 프로그램 개발시키기"
author: 김태영
date:   2023-2-16 00:00:00
categories: tech
comments: true
image: https://tykimos.github.io/warehouse/2023/2023-2-16-ChatGPT_Auto_Trading_Program_in_Python_title1.png
---

이번 글에서는 비트코인 자동매매 프로그램 개발에 대해 살펴보고, 챗GPT를 활용해 매매 전략을 짜보도록 하겠습니다. 먼저 비트코인 자동매매 프로그램과 챗GPT에 대해서 간단하게 알아보겠습니다.

<iframe width="100%" height="400" src="https://www.youtube.com/embed/Tgngw6-IH98" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

(참고로 위 영상을 생성하는 과정에서도 인공지능 도움을 많이 받았는데요. 글 일부도 챗GPT가 작성해줬고, 음성은 타입캐스트로 생성했으며, 얼굴 영상은 D-ID 서비스를 이용했습니다.)

## 소개

### 비트코인 자동매매 프로그램

비트코인은 가상화폐 중에서 가장 대표적인 종목입니다. 매매시기를 결정하는 것은 매우 어려운데요. 이때 자동매매 프로그램은 시장을 분석하고 자동으로 매매를 결정해주는 데에 도움을 받을 수 있습니다. 대부분 이러한 프로그램은 다양한 지표들을 종합하여 매매 시기를 결정하고, 자동으로 주문을 처리합니다.  대부분의 자동매매 프로그램은 다음과 같은 기본 전략을 사용합니다.

* 추세 판단: 가격 변동 추세를 파악합니다.
* 매수/매도 포인트 결정: 추세에 따라 매수/매도 포인트를 결정합니다.
* 주문 처리: 결정된 매수/매도 포인트를 바탕으로 주문을 처리합니다.
* 수익/손실 관리: 매매 이후 수익과 손실을 관리합니다.

이때, 챗GPT 모델은 추세를 파악하고, 매수/매도 포인트를 예측하는 데에 활용됩니다. 예를 들어, 챗GPT 모델을 통해 "현재 추세는 상승세이고, 이후에는 더 올라갈 가능성이 높습니다."라는 정보를 얻었다면, 이를 바탕으로 매수 포인트를 결정합니다.

### 챗GPT 란
챗GPT는 OpenAI에서 개발한 자연어 처리(Natural Language Processing) 모델입니다. 챗GPT는 대용량의 데이터를 학습하여, 인간의 언어처리 능력을 모방하는 기술을 사용합니다. 이 모델은 대화형 인공지능, 기계 번역, 텍스트 생성 등 다양한 분야에서 활용됩니다.

## 만들기 순서

챗GPT를 활용하여 비트코인 자동매매 프로그램 순서는 아래와 같습니다. 
1. 매매 전략을 제외한 비트코인 자동매매 프로그램 작성하기
2. 매매 전략을 챗GPT에게 작성 부탁하기
3. 챗GPT가 작성한 매매 전략에 대해 검증하기
4. 실제로 시장에서 자동매매 시켜보기

위 순서대로 하나씩 살펴보겠습니다. 

### (1) 매매 전략을 제외한 비트코인 자동매매 프로그램 작성하기
챗GPT에게 비트코인 자동매매 프로그램 작성을 부탁하면, 매매 전략을 포함한 전체 코드를 작성해주긴 하지만, 매번 다른 코드를 만들기도 하고, 오류가 포함된 코드도 만들기 때문에 검증을 마친 코드를 사용하는 것을 권장드립니다. 이번 예제에서 사용된 코드는 아래와 같습니다. 가상화폐 거래소 중 하나인 빗썸에서 제공하는 파이썬 API를 이용하여 만들었습니다.
이 코드에 대한 전제조건은 아래와 같습니다. 

* 매수신호체크 (check_buy_signal) 함수 반환값이 참(True)이면 매수합니다.
* 매도신호체크 (check_sell_signal) 함수 반환값이 참(True)이면 매도합니다.
* 매수 시 전체 잔고로 매수합니다.
* 매도 시 전체 보유량을 매도합니다.

0.2초 간격으로 현재가를 정보를 수신하여 매매 조건이 되는 지 체크합니다. 종목을 보유하고 있지 않다면 매수신호를, 종목을 보유하고 있다면 매도 신호를 기다립니다.매수 후 하루가 지난 시점에도 매도 신호가 없다면 전량 매도합니다.con_key와 sec_key 는 빗썸 사이트에서 API 사용 신청 후 발급 받은 정보를 기입합니다.

```python
import pybithumb
import datetime
import time

con_key = "발급된 키 정보 삽입"
sec_key = "발급된 키 정보 삽입"

bithumb = pybithumb.Bithumb(con_key, sec_key)

ticker = "BTC"

def check_buy_signal(ticker, curr_price):
    
    return False
    
def check_sell_signal(ticker, curr_price, buy_price):
    
    return False

def buy_all(ticker):  
    krw = bithumb.get_balance(ticker)[2]
    orderbook = pybithumb.get_orderbook(ticker)
    sell_price = orderbook['asks'][0]['price']
    unit = krw / float(sell_price)
    bithumb.buy_market_order(ticker, unit)
    return unit

def sell_all(ticker):
    unit = bithumb.get_balance(ticker)[0]
    bithumb.sell_market_order(ticker, unit)
    return unit

holding = False

while True:
    
    curr_price = pybithumb.get_current_price(ticker)
    curr_time = datetime.datetime.now()
    
    print(curr_time, '|', curr_price)
    
    if holding == False:
        if check_buy_signal(ticker, curr_price):
            unit = buy_all(ticker)
            buy_price = curr_price
            buy_time = curr_time
            print("[buy] unit:", unit, ", buy_price:", curr_price)
            holding = True
    else:
        if check_sell_signal(ticker, curr_price, buy_price):
            unit = sell_all(ticker)
            print("[sell by signal] unit:", unit, ", sell_price:", curr_price)
            holding = False
        elif (curr_time - buy_time).total_seconds() > 86400:
            unit = sell_all(ticker)
            print("[sell over 24hour] unit:", unit, ", sell_price:", curr_price)
            holding = False
            
    time.sleep(0.2)
```

### (2) 매매 전략을 챗GPT에게 작성 부탁하기

먼저 챗GPT에게 위에서 구현한 매매 전략을 제외한 비트코인 자동매매 프로그램을 이해시킨 후, 챗GPT에게 높은 수익의 매매 전략을 check_buy_signal와 check_sell_signal 함수에 작성해달라고 합니다. 

임계값(threshold) 기반의 간단한 알고리즘을 작성해주네요. 이 전략외에도 챗GPT가 작성한 다양한 전략은 뒤에서 살펴보겠습니다. 

```python
# Constants to be used in the trading strategy
BUY_THRESHOLD = 0.05  # percentage change in price to trigger a buy signal
SELL_THRESHOLD = -0.03  # percentage change in price to trigger a sell signal
STOP_LOSS_THRESHOLD = -0.1  # percentage change in price to trigger a stop-loss sell

def check_buy_signal(ticker, prev_price, curr_price):
    """
    Check if a buy signal should be triggered based on the current price of the asset.
    """
    price_change = (curr_price - prev_price) / prev_price  # calculate price change percentage

    if price_change > BUY_THRESHOLD:
        return True
    else:
        return False

def check_sell_signal(ticker, curr_price, buy_price):
    """
    Check if a sell signal should be triggered based on the current price of the asset and the buy price.
    """
    price_change = (curr_price - buy_price) / buy_price  # calculate price change percentage

    if price_change < SELL_THRESHOLD:
        return True  # sell if the price has dropped by the sell threshold percentage
    elif price_change < STOP_LOSS_THRESHOLD:
        return True  # sell if the price has dropped by the stop loss threshold percentage
    else:
        return False
```

### (3) 챗GPT가 작성한 매매 전략에 대해 검증하기

매매 전략을 검증하기 위해 가장 많이 사용하는 것이 백테스팅입니다. 백테스팅 코드를 챗GPT에게 부탁해도 작성해주긴 하지만, 사용자마다 검증 지표가 다르고 방식도 여러가지 이기 때문에 나만의 검증 방법에 대해 직접 테스트해보시는 것을 추천드립니다. 아래 코드는 기간수익률(HPR, Holding Period Return), 최대손실낙폭(MDD, Maximum Drawdown) 등을 리포팅해주는 백테스팅 클래스입니다. 

```python
class BackTesting():
    
    def __init__(self, start_cash):
        self.start_cash = start_cash # 시작 자산
        self.ror_list = []
        self.buy_count = 0  # 매수횟수
        self.sell_count = 0 # 매도 횟수
        self.sell_count_by_timeout = 0 # 타임아웃에 의한 매도 횟수
        self.win_count = 0 # 승리 횟수
        self.lose_count = 0 # 패배 횟수
        self.draw_count = 0 # 무승부 횟수
        self.curr_price = 0        
        self.highest_price = -float('inf') # 최고가
        self.lowest_price = float('inf') # 최저가
        self.holding = False # 보유여부
        self.buy_price = 0

    def curr_recode(self, curr_price):
        self.curr_price = curr_price
        
        if self.curr_price > self.highest_price:
            self.highest_price = self.curr_price
        if self.curr_price < self.lowest_price:
            self.lowest_price = self.curr_price
        
    def buy_record(self, curr_price):
        self.buy_count += 1
        self.holding = True
        self.buy_price = curr_price
        
    def sell_record(self, timeout):
        self.sell_count += 1
        if timeout == True:
            self.sell_count_by_timeout += 1

        # Rate of Return 회차별 수익률, 회차는 전량 매수 후 전량 매도까지를 의미
        ror = float(self.curr_price) / float(self.buy_price)
        self.ror_list.append(ror)  

    def print_report(self):
        
        print("count buy sell timeout_sell : ", self.buy_count, self.sell_count, self.sell_count_by_timeout)
        print('price curr high low : ', self.curr_price, self.highest_price, self.lowest_price)
                
        current_balance = self.start_cash
        highest_balance = self.start_cash
        lowest_balance = self.start_cash

        # Holding Period Return 기간수익률 : 수익률을 모두 곱한 것
        hpr = 1
        
        for ror in self.ror_list:
            if ror > 1.0 :
                self.win_count += 1
            elif ror == 1.0 :
                self.draw_count += 1
            else:
                self.lose_count += 1

            hpr *= ror
            current_balance *= ror
            
            if current_balance > highest_balance:
                highest_balance = current_balance
            if current_balance < lowest_balance:
                lowest_balance = current_balance        

        print('count win draw lose :', self.win_count, self.draw_count, self.lose_count)
        print('winning rate : {:.4f}'.format(self.win_count / (self.win_count + self.draw_count + self.lose_count)))

        print('start cash :', self.start_cash)
        print('balance curr high low : {0:.4f} {1:.4f} {2:.4f}'.format(current_balance, highest_balance, lowest_balance))
        print('Maximum Drawdown : {:.4f}'.format((highest_balance - lowest_balance) / highest_balance))
        print('Holding period return : {:.4f}'.format(hpr))
```

백테스팅 클래스를 이용하여 전략에 대해서 검증합니다. 현재가를 얻었을 때와 매매가 발생했을 때 이 클래스 객체를 호출하고 마지막에 리포트를 출력시킵니다.

* curr_record 함수는 현재 가격을 처리하는 함수
* buy_record 함수는 매수 시에 호출하여 기록하는 함수입니다.
* sell_record 함수는 매도 시에 호출하여 기록하는 함수입니다.
* print_report 함수는 지금까지 백테스팅한 결과를 계산하여 출력하는 함수입니다.

백테스팅 객체를 생성한 후 백테스팅을 위해서 현재 가격을 가지고 오는 것이 아니라 빗썸 API를 이용해서 과거 데이터를 가지고 옵니다. 과거 데이터 처리 후에 보유한 종목이 있다면, 마지막으로 전량 매도로 가정하고 sell_record 함수를 호출합니다.

```python
bt = BackTesting(100)

import pybithumb
import pandas as pd

ticker = "BTC"

# Constants to be used in the trading strategy
BUY_THRESHOLD = 0.05  # percentage change in price to trigger a buy signal
SELL_THRESHOLD = -0.03  # percentage change in price to trigger a sell signal
STOP_LOSS_THRESHOLD = -0.1  # percentage change in price to trigger a stop-loss sell

def check_buy_signal(ticker, prev_price, curr_price):
    #previous_price = pybithumb.get_ohlcv(ticker)['close'][-2]  # get previous closing price
    price_change = (curr_price - prev_price) / prev_price  # calculate price change percentage

    if price_change > BUY_THRESHOLD:
        return True
    else:
        return False

def check_sell_signal(ticker, curr_price, buy_price):
    price_change = (curr_price - buy_price) / buy_price  # calculate price change percentage

    if price_change < SELL_THRESHOLD:
        return True  # sell if the price has dropped by the sell threshold percentage
    elif price_change < STOP_LOSS_THRESHOLD:
        return True  # sell if the price has dropped by the stop loss threshold percentage
    else:
        return False
    
# retrieve historical data
df = pybithumb.get_ohlcv(ticker, interval='day')

# initialize variables
holding = False
buy_price = 0
buy_time = None

# iterate over the historical data
for i in range(1, len(df)):
    curr_price = df.iloc[i]['close']
    prev_price = df.iloc[i-1]['close']
    curr_time = df.index[i]

    bt.curr_recode(curr_price)
    
    if holding == False:
        if check_buy_signal(ticker, prev_price, curr_price):
            bt.buy_record(curr_price)
            buy_price = curr_price
            buy_time = curr_time
            holding = True
    else:
        if check_sell_signal(ticker, curr_price, buy_price):
            holding = False
            bt.sell_record(False)
        elif (curr_time - buy_time).total_seconds() > 86400*2:
            holding = False
            bt.sell_record(True)

# calculate final holding period return
if holding:
    bt.sell_record(True)
    
bt.print_report()  
```

아래 화면은 백테스팅 결과를 출력한 것입니다. 매매 횟수, 현재/최고/최저 가격승률, 시작 자본, 평가 잔고 그리고 MDD와 HPR이 표시됩니다.

https://tykimos.github.io/warehouse/2023/2023-2-16-ChatGPT_Auto_Trading_Program_in_Python_1.png

### (4) 실전에 자동매매 시켜보기

검증까지 마쳤다면 실전으로 돌려봅니다. 매매 전략을 챗GPT 모델로 자동으로 생성하고, 검증을 통해 효과적인 전략임을 확인하였다면 실제로 시장에서 자동매매를 시켜봅니다. 실전에서는 잔고 상황과 시장 상황에 따라 매매가 정상적으로 이뤄지지 않을 수 있기 때문에 적은 금액으로 먼저 안정성 테스트를 수행합니다. 또한 매매 결과를 실시간으로 모니터링하면서 매매 전략이 제대로 작동하는 지와 효과적인 지 확인하며 자동매매 프로그램과 전략을 개선시켜나가야 합니다.

```python
import pybithumb
import datetime
import time

con_key = "발급된 키 정보 삽입"
sec_key = "발급된 키 정보 삽입"

bithumb = pybithumb.Bithumb(con_key, sec_key)

ticker = "BTC"

# Constants to be used in the trading strategy
BUY_THRESHOLD = 0.05  # percentage change in price to trigger a buy signal
SELL_THRESHOLD = -0.03  # percentage change in price to trigger a sell signal
STOP_LOSS_THRESHOLD = -0.1  # percentage change in price to trigger a stop-loss sell

def check_buy_signal(ticker, prev_price, curr_price):
    """
    Check if a buy signal should be triggered based on the current price of the asset.
    """
    price_change = (curr_price - prev_price) / prev_price  # calculate price change percentage

    if price_change > BUY_THRESHOLD:
        return True
    else:
        return False

def check_sell_signal(ticker, curr_price, buy_price):
    """
    Check if a sell signal should be triggered based on the current price of the asset and the buy price.
    """
    price_change = (curr_price - buy_price) / buy_price  # calculate price change percentage

    if price_change < SELL_THRESHOLD:
        return True  # sell if the price has dropped by the sell threshold percentage
    elif price_change < STOP_LOSS_THRESHOLD:
        return True  # sell if the price has dropped by the stop loss threshold percentage
    else:
        return False

def buy_all(ticker):  
    krw = bithumb.get_balance(ticker)[2]
    orderbook = pybithumb.get_orderbook(ticker)
    sell_price = orderbook['asks'][0]['price']
    unit = krw / float(sell_price)
    bithumb.buy_market_order(ticker, unit)
    return unit

def sell_all(ticker):
    unit = bithumb.get_balance(ticker)[0]
    bithumb.sell_market_order(ticker, unit)
    return unit

holding = False

while True:
    
    try:
        curr_price = pybithumb.get_current_price(ticker)
        curr_time = datetime.datetime.now()
        
        if holding == False:
            
            prev_price = pybithumb.get_ohlcv(ticker)['close'][-2]  # get previous closing price
            
            if check_buy_signal(ticker, prev_price, curr_price):
                unit = buy_all(ticker)
                buy_price = curr_price
                buy_time = curr_time
                print("[buy] unit:", unit, ", buy_price:", curr_price)
                holding = True
        else:
            if check_sell_signal(ticker, curr_price, buy_price):
                unit = sell_all(ticker)
                print("[sell by signal] unit:", unit, ", sell_price:", curr_price)
                holding = False
            elif (curr_time - buy_time).total_seconds() > 86400:
                unit = sell_all(ticker)
                print("[sell over 24hour] unit:", unit, ", sell_price:", curr_price)
                holding = False
    except:
        print("error")
            
    time.sleep(0.2)
```

## 챗GPT가 생성한 다양한 전략들

위 예제에서는 임계값을 이용한 간단한 전략을 예시로 들었지만, 원하는 지표에 대해 챗GPT에게 좀 더 상세하게 주문하면 다양한 전략을 구사해줍니다. 챗GPT가 생성한 몇 가지 예시를 보도록 하겠습니다.

### MACD와 RSI 조합 전략

MACD와 RSI 조합 전략에 대해서 요청해봤습니다.

    write a buy_signal and sell_signal function for a highly profitable bitcoin automatic trading suggest using several indicators such as Moving Average Convergence Divergence (MACD) or Relative Strength Index (RSI) and combination of strategies such as Volatility Breakout or SMA

위 주문에 대한 챗GPT의 답변입니다.

Sure, here's an example of the check_buy_signal() and check_sell_signal() functions that can be used as a high-yield trading strategy for Bitcoin. These functions use the moving average convergence divergence (MACD) indicator and relative strength index (RSI) as technical indicators to generate buy and sell signals. The check_buy_signal() function generates a buy signal when the MACD line crosses the signal line from below and the RSI is below 30. The check_sell_signal() function generates a sell signal when the MACD line crosses the signal line from above or when the RSI is above 70.

talib이라는 패키지를 사용하여 MACD와 RSI 지표를 사용하는 전략을 알려주네요. 매도 시에도 동일 지표를 사용하여 조건을 체크합니다.

```python
import talib

# Define MACD and RSI parameters
fast_period = 12
slow_period = 26
signal_period = 9
rsi_period = 14

def check_buy_signal(ticker, curr_price):
    # Get historical data for MACD and RSI calculation
    historical_data = pybithumb.get_ohlcv(ticker)
    close_prices = historical_data['close']
    
    # Calculate MACD and signal line
    macd, signal, _ = talib.MACD(close_prices, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    
    # Calculate RSI
    rsi = talib.RSI(close_prices, timeperiod=rsi_period)
    
    # Check if MACD line crosses signal line from below and RSI is below 30
    if macd[-1] > signal[-1] and macd[-2] < signal[-2] and rsi[-1] < 30:
        return True
    else:
        return False

def check_sell_signal(ticker, curr_price, buy_price):
    # Get historical data for MACD calculation
    historical_data = pybithumb.get_ohlcv(ticker)
    close_prices = historical_data['close']
    
    # Calculate MACD and signal line
    macd, signal, _ = talib.MACD(close_prices, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    
    # Calculate RSI
    rsi = talib.RSI(close_prices, timeperiod=rsi_period)
    
    # Check if MACD line crosses signal line from above or RSI is above 70
    if macd[-1] < signal[-1] and macd[-2] > signal[-2] or rsi[-1] > 70:
        return True
    else:
        return False

```

### MACD와 BollingerBands 조합 전략

챗GPT는 동일한 요청을 하더라도 실행시킬 때 마다 다른 답변을 해줍니다.

    write a buy_signal and sell_signal function for a highly profitable bitcoin automatic trading suggest using several indicators such as Moving Average Convergence Divergence (MACD) or Relative Strength Index (RSI) and combination of strategies such as Volatility Breakout or SMA

위 주문에 대한 챗GPT의 답변입니다.

Note that this is just an example of a trading strategy and there are many other strategies that can be used. It's also important to note that trading involves risk and it's important to understand the risks and do your own research before implementing any trading strategy.

이번에는 MACD와 볼린저밴드 조합 전략을 알려주네요. ta라는 패키지를 통해 MACD와 볼린저밴드 지표를 가지고 와서 활용합니다. 매도 시에도 이 두 지표와 인자로 넣어준 매수 가격을 이용하여 조건을 판단합니다.

```python
from ta.trend import MACD
from ta.volatility import BollingerBands

def buy_signal(ticker, curr_price):
    # Get the MACD and Bollinger Bands indicators for the specified ticker
    df = pybithumb.get_ohlcv(ticker)
    macd = MACD(df['close']).macd()[-1]
    upper_band = BollingerBands(df['close']).bollinger_hband()[-1]
    middle_band = BollingerBands(df['close']).bollinger_mavg()[-1]
    lower_band = BollingerBands(df['close']).bollinger_lband()[-1]

    # Check if the MACD is positive and the current price is below the upper Bollinger Band
    if macd > 0 and curr_price < upper_band:
        return True
    else:
        return False

def sell_signal(ticker, curr_price, buy_price):
    # Get the MACD and Bollinger Bands indicators for the specified ticker
    df = pybithumb.get_ohlcv(ticker)
    macd = MACD(df['close']).macd()[-1]
    upper_band = BollingerBands(df['close']).bollinger_hband()[-1]
    lower_band = BollingerBands(df['close']).bollinger_lband()[-1]

    # Check if the MACD is negative or the current price is above the upper Bollinger Band or below the lower Bollinger Band
    if macd < 0 or curr_price > upper_band or curr_price < lower_band:
        return True
    # Check if the current price has dropped below the buy price by a certain percentage
    elif (curr_price / buy_price) < 0.9:
        return True
    else:
        return False

```
This updated buy_signal function takes the ticker and curr_price parameters, and uses the ticker to retrieve the historical price data needed to calculate the indicators, as before. The function then checks if the MACD is positive and the current price is below the upper Bollinger Band, using the curr_price parameter.

### SMA 전략

다음은 별다른 조건없이 간단하게 다시 주문한 것입니다.

    compelete buy_signal and sell_signal function for profitable bitcoin automatic trading program.

위 주문에 대한 챗GPT의 답변입니다. 매수 시에는 직관적이고 간단하게 사용할 수 있는 SMA 지표를 사용하는 전략이 나왔네요. 매도 시에는 손실과 타임아웃을 사용하여 조건을 판단합니다. 

As an AI language model, I cannot guarantee a profitable strategy, but I can provide general guidelines and examples of buy_signal and sell_signal functions.

Here is an example of a buy_signal function that uses a simple moving average (SMA) crossover strategy:

```python
def buy_signal(curr_price):
    # get historical prices for the last 20 minutes (100 data points with 12-second interval)
    historical_prices = pybithumb.get_ohlcv("BTC", interval="12s", count=100)
    # calculate the 5-period and 20-period simple moving averages
    sma5 = historical_prices['close'].rolling(window=5).mean()
    sma20 = historical_prices['close'].rolling(window=20).mean()
    # check if the 5-period SMA has crossed above the 20-period SMA
    if sma5[-1] > sma20[-1] and sma5[-2] <= sma20[-2]:
        return True
    else:
        return False
```

This function checks whether the 5-period SMA has crossed above the 20-period SMA, which is a bullish signal that indicates a potential trend reversal.

```python
def sell_signal(curr_price):
    # get the buy price and holding time
    global buy_price, buy_time
    # calculate the stop loss level as 3% below the buy price
    stop_loss = buy_price * 0.97
    # check if the current price has dropped below the stop loss level
    if curr_price <= stop_loss:
        return True
    # check if 24 hours have passed since the buy time
    elif time.time() - buy_time >= 86400:
        return True
    else:
        return False
```

This function checks whether the current price has dropped below a 3% stop loss level or whether 24 hours have passed since the buy time. If either of these conditions is met, the function returns True, indicating a sell signal. Note that the buy_price and buy_time variables are assumed to be set when the buy signal is triggered.

전략 내에 코드는 전략 외의 매매 프로그램 구성에 영향을 미치기 때문에 챗GPT에게 요청할 때는 명확하게 요구사항을 주문하는 것이 중요합니다.

## 결론

이번 글에서는 비트코인 자동매매 프로그램 개발에 대해 알아보았습니다. 챗GPT는 자연어 처리 모델이지만, 그 기능을 활용하여 비트코인 자동매매 프로그램 개발에도 사용해봤습니다. 다양한 전략을 공부하거나 직접 프로그래밍할 때도 도움을 받을 수 있을 것 같습니다. 다만 본 예제를 포함하여 챗GPT에서 생성한 매매 전략 사용에 대한 책임은 사용자에게 있기 때문에 실전에 사용할 때는 신중하게 접근하시기 바랍니다. 실전에 적용한 결과는 테스트 해보고 추후에 다시 올리겠습니다.
