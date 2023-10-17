---
layout: post
title: "스위치봇 API 시작하기"
author: Taeyoung Kim
date: 2023-10-17 00:00:00
categories: chatgpt, IoT, switchbot
comments: true
---

```python
import hashlib
import hmac
import base64
import uuid
import time
import openai
import requests

def get_current_temperature(param=''):
    apiHeader = {}
    token = 'token'
    secret = 'secret'
    nonce = uuid.uuid4()
    t = int(round(time.time() * 1000))
    string_to_sign = '{}{}{}'.format(token, t, nonce)

    string_to_sign = bytes(string_to_sign, 'utf-8')
    secret = bytes(secret, 'utf-8')

    sign = base64.b64encode(hmac.new(secret, msg=string_to_sign, digestmod=hashlib.sha256).digest())

    apiHeader['Authorization']=token
    apiHeader['Content-Type']='application/json'
    apiHeader['charset']='utf8'
    apiHeader['t']=str(t)
    apiHeader['sign']=str(sign, 'utf-8')
    apiHeader['nonce']=str(nonce)

    # Make the GET request
    response = requests.get('https://api.switch-bot.com/v1.1/devices', headers=apiHeader)

    # Make the GET request for the hub status
    hub_id = 'F5BAA8DC2AFE'
    response = requests.get(f'https://api.switch-bot.com/v1.1/devices/{hub_id}/status', headers=apiHeader)

    # Print the status code and returned data
    print(response.status_code)
    print(response.json())

    return_message = ''

    if response.status_code != 200:
        return_message = 'error'
    else:
        response_data = response.json()
        temperature = response_data['body']['temperature']
        return_message = str(temperature)

    return return_message

def get_current_humidity(param=''):
    apiHeader = {}
    token = 'token'
    secret = 'secret'
    nonce = uuid.uuid4()
    t = int(round(time.time() * 1000))
    string_to_sign = '{}{}{}'.format(token, t, nonce)

    string_to_sign = bytes(string_to_sign, 'utf-8')
    secret = bytes(secret, 'utf-8')

    sign = base64.b64encode(hmac.new(secret, msg=string_to_sign, digestmod=hashlib.sha256).digest())

    apiHeader['Authorization']=token
    apiHeader['Content-Type']='application/json'
    apiHeader['charset']='utf8'
    apiHeader['t']=str(t)
    apiHeader['sign']=str(sign, 'utf-8')
    apiHeader['nonce']=str(nonce)

    # Make the GET request
    response = requests.get('https://api.switch-bot.com/v1.1/devices', headers=apiHeader)

    # Make the GET request for the hub status
    hub_id = 'F5BAA8DC2AFE'
    response = requests.get(f'https://api.switch-bot.com/v1.1/devices/{hub_id}/status', headers=apiHeader)

    # Print the status code and returned data
    print(response.status_code)
    print(response.json())

    return_message = ''

    if response.status_code != 200:
        return_message = 'error'
    else:
        response_data = response.json()
        temperature = response_data['body']['humidity']
        return_message = str(temperature)

    return return_message

def turn_on_light(param=''):
    apiHeader = {}
    token = 'token'
    secret = 'secret'
    nonce = uuid.uuid4()
    t = int(round(time.time() * 1000))
    string_to_sign = '{}{}{}'.format(token, t, nonce)

    string_to_sign = bytes(string_to_sign, 'utf-8')
    secret = bytes(secret, 'utf-8')

    sign = base64.b64encode(hmac.new(secret, msg=string_to_sign, digestmod=hashlib.sha256).digest())

    apiHeader['Authorization']=token
    apiHeader['Content-Type']='application/json'
    apiHeader['charset']='utf8'
    apiHeader['t']=str(t)
    apiHeader['sign']=str(sign, 'utf-8')
    apiHeader['nonce']=str(nonce)

    # Make the GET request
    response = requests.get('https://api.switch-bot.com/v1.1/devices', headers=apiHeader)

    # Make the POST request to turn on the bot
    bot_id = 'D23533341257'
    command_data = {
        "command": "turnOn",
        "parameter": "default",
        "commandType": "command"
    }

    response = requests.post(f'https://api.switch-bot.com/v1.1/devices/{bot_id}/commands', headers=apiHeader, json=command_data)

    # Print the status code and returned data
    print(response.status_code)
    print(response.json())

    return_message = ''

    if response.status_code != 200:
        return_message = 'error'
    else:
        return_message = 'success'

    return return_message

def turn_off_light(param=''):
    apiHeader = {}
    token = 'token'
    secret = 'secret'
    nonce = uuid.uuid4()
    t = int(round(time.time() * 1000))
    string_to_sign = '{}{}{}'.format(token, t, nonce)

    string_to_sign = bytes(string_to_sign, 'utf-8')
    secret = bytes(secret, 'utf-8')

    sign = base64.b64encode(hmac.new(secret, msg=string_to_sign, digestmod=hashlib.sha256).digest())

    apiHeader['Authorization']=token
    apiHeader['Content-Type']='application/json'
    apiHeader['charset']='utf8'
    apiHeader['t']=str(t)
    apiHeader['sign']=str(sign, 'utf-8')
    apiHeader['nonce']=str(nonce)

    # Make the GET request
    response = requests.get('https://api.switch-bot.com/v1.1/devices', headers=apiHeader)

    # Make the POST request to turn on the bot
    bot_id = 'D23533341257'
    command_data = {
        "command": "turnOff",
        "parameter": "default",
        "commandType": "command"
    }
    response = requests.post(f'https://api.switch-bot.com/v1.1/devices/{bot_id}/commands', headers=apiHeader, json=command_data)

    return_message = ''

    if response.status_code != 200:
        return_message = 'error'
    else:
        return_message = 'success'

    return return_message
```


