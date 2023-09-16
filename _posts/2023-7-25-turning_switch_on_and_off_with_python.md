---
layout: post
title:  "파이썬으로 방 스위치 켜고 끄기"
author: 김태영
date:   2023-7-24 00:00:00
categories: ai
comments: true
image: https://tykimos.github.io/warehouse/2023/2023-7-25-turning_switch_on_and_off_with_python_title.png
---

### 스위치봇(SwitchBot) API를 이용해 디바이스 제어하기

스위치봇은 스마트 홈 디바이스를 제어하는데 사용할 수 있는 인기 있는 플랫폼입니다. 이 포스트에서는 Python과 SwitchBot API를 이용하여 디바이스를 제어하는 방법을 소개하겠습니다. 특히, 스위치를 켜고 끄는 방법을 중점적으로 다룰 예정입니다. 아래 영상은 본 소스코드의 함수를 실행하여 스위치봇 켜고 끄기를 테스트해본 영상입니다.

<iframe width="100%" height="400" src="https://www.youtube.com/embed/_Ei98Pa6sf4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### 시작하기 전에 필요한 라이브러리

먼저, 이 스크립트를 실행하기 위해 필요한 Python 라이브러리들을 설치해야 합니다. 다음과 같이 설치할 수 있습니다.

```bash
pip install requests
```

### SwitchBot API 이해하기

SwitchBot API를 사용하려면, 먼저 API 요청에 필요한 헤더를 설정해야 합니다. 헤더에는 인증 정보와 컨텐트 타입 등이 포함됩니다.

```python
apiHeader = {}
token = 'your-token'
secret = 'your-secret'
...
apiHeader['Authorization']=token
apiHeader['Content-Type']='application/json'
apiHeader['charset']='utf8'
apiHeader['t']=str(t)
apiHeader['sign']=str(sign, 'utf-8')
apiHeader['nonce']=str(nonce)
```

**참고:** 실제 코드에 토큰과 시크릿을 직접 삽입하는 것은 안전하지 않습니다. 환경 변수 또는 별도의 보안이 적용된 설정 파일을 사용하는 것이 좋습니다.

이후 SwitchBot API를 사용하여, 디바이스의 상태를 조회하거나 특정 명령을 보낼 수 있습니다. 상세한 SwitchBot API 문서는 아래를 참고하세요.

* SwitchBot API : https://github.com/OpenWonderLabs/SwitchBotAPI

### 디바이스 상태 조회하기

스위치봇 API를 사용하여 특정 디바이스의 상태를 확인할 수 있습니다.

```python
hub_id = 'your-hub-id'
response = requests.get(f'https://api.switch-bot.com/v1.1/devices/{hub_id}/status', headers=apiHeader)
```

## 스위치 켜고 끄기

스위치봇의 특정 디바이스에 'turnOn' 또는 'turnOff' 명령을 보내어 스위치를 켜거나 끌 수 있습니다.

```python
def turn_on_switch():
    bot_id = 'your-bot-id'
    command_data = {
        "command": "turnOn",
        "parameter": "default",
        "commandType": "command"
    }
    response = requests.post(f'https://api.switch-bot.com/v1.1/devices/{bot_id}/commands', headers=apiHeader, json=command_data)

def turn_off_switch():
    bot_id = 'your-bot-id'
    command_data = {
        "command": "turnOff",
        "parameter": "default",
        "commandType": "command"
    }
    response = requests.post(f'https://api.switch-bot.com/v1.1/devices/{bot_id}/commands', headers=apiHeader, json=command_data)
```

이 코드를 이용하면, SwitchBot API를 사용하여 스마트 홈 디바이스를 원격으로 제어하는 방법에 대한 기본적인 이해를 얻을 수 있을 것입니다.

### 마무리

이번 포스트에서는 Python과 SwitchBot API를 활용하여 스마트 홈 디바이스를 원격으로 제어하는 방법에 대해 살펴보았습니다. 위의 코드를 통해 간단히 스위치를 켜고 끄는 작업을 수행할 수 있지만, SwitchBot API는 이 외에도 다양한 기능을 제공합니다. 이를 활용하여 자신만의 스마트 홈 시스템을 구축해 보세요. API를 이용하는 것은 처음에는 복잡하게 느껴질 수 있지만, 이해하고 나면 원격으로 디바이스를 제어하고, 자동화된 작업을 수행하는 등 많은 장점을 가질 수 있습니다. 

다음에는 SwitchBot API과 챗GPT를 연동하여 챗GPT로 스위치를 조작해보겠습니다.
