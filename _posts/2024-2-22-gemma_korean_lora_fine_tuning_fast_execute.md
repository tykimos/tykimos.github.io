---
layout: post
title: "Gemma 한국어 LoRA 파인튜닝 빠른실행"
author: 김태영
date: 2024-2-22 01:00:00
categories: llm
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_korean_lora_fine_tuning_fast_execute_title_3.png
---
  
![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_korean_lora_fine_tuning_fast_execute_title_3.png)

이번에는 Gemma를 "databricks-dolly-15k-ko.jsonl" 데이터셋을 이용해서 한국어 LoRA 파인튜닝을 해보도록 하겠습니다.

### 함께보기

* 1편 - Gemma 시작하기 빠른실행 (추후 공개)
* [2편 - Gemma LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_lora_fine_tuning_fast_execute/)
* [3편 - Gemma 한국어 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_korean_lora_fine_tuning_fast_execute/)
* [4편 - Gemma 영한번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_en2ko_lora_fine_tuning_fast_execute/)
* [5편 - Gemma 한영번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_ko2en_lora_fine_tuning_fast_execute/)
* [6편 - Gemma 한국어 SQL챗봇 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/23/gemma_ko2sql_lora_fine_tuning_fast_execute/)
* [7편 - Gemma 온디바이스 탑재 - 웹브라우저편 빠른실행](https://tykimos.github.io/2024/04/02/gemma_ondevice_webbrowser_fast_execute/)
* [8편 - Gemma 온디바이스 탑재 - 아이폰(iOS)편 빠른실행](https://tykimos.github.io/2024/04/05/local_llm_installed_on_my_iphone_gemma_2b/)

### databricks-dolly-15k-ko 데이터셋

databricks-dolly-15k-ko 데이터셋은 허깅페이스에서 다운로드 받을 수 있으며, NLP & AI - Korea University에서 databricks-dolly-15k를 DeepL API를 이용해서 한국어 번역을 수행한 파일입니다.

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_korean_lora_fine_tuning_fast_execute_1.png)

* [링크](https://huggingface.co/datasets/nlpai-lab/databricks-dolly-15k-ko)

#### 수행결과 1

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_korean_lora_fine_tuning_fast_execute_2.png)

```
Instruction:
유럽 여행에서 무엇을 해야 할까요?

Response:
여행하는 동안 멋진 관람객 경험을 하려면 다음을 고려해야 합니다.
- 멋진 호텔
- 멋진 레스토랑
- 멋진 관광지

여행을 하다가 멋진 관람객 경험을 하려면 다음을 고려해야 합니다.
- 멋진 숙소에서 멋진 레스토랑으로 즐거운 경험을 하려면 멋진 숙소를 선택해야 합니다.
```

#### 수행결과 2

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_korean_lora_fine_tuning_fast_execute_3.png)

```
Instruction:
광합성 과정을 5살 아이도 이해할 수 있게 설명해주세요.

Response:
광합성 과정은 식물과 녹조류에서 일차 생산을 위해 필요한 과정입니다. 이 과정은 광합성과 탄산화를 통해 이루어집니다. 식물은 광합성으로 이루어진 빛과 수분이 풍부한 습기가 있는 환경을 선호하고 탄산가스를 필요로 합니다. 그래서 이 과정을 위해서는 빛, 수분 그리고 탄산가스가 필요합니다. 식물의 광합성은 빛이 필요한 과정이기 때문에, 식용 가능한 식물이 빛을 필요로 합니다. 그러나 식용 가능한 식물은 광합성 과정을 통해서 탄소를 흡수하고, 이 탄소를 식물의 탄수화물로 변화시킵니다. 그래서 식물의 탄수화물이 식용 가능한 식물의 식물로 변화하는 것은 광합성과 탄산화가 필요한 것
```

### 추가문의

* 작성자 : 김태영
* 이메일 : tykim@aifactory.page

### 더보기

LoRA 파인튜닝 공식 예제는 다음과 같습니다.

* [https://ai.google.dev/gemma/docs/lora_tuning](https://ai.google.dev/gemma/docs/lora_tuning)

* 1편 - Gemma 시작하기 빠른실행 (추후 공개)
* [2편 - Gemma LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_lora_fine_tuning_fast_execute/)
* [3편 - Gemma 한국어 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_korean_lora_fine_tuning_fast_execute/)
* [4편 - Gemma 영한번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_en2ko_lora_fine_tuning_fast_execute/)
* [5편 - Gemma 한영번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_ko2en_lora_fine_tuning_fast_execute/)
* [6편 - Gemma 한국어 SQL챗봇 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/23/gemma_ko2sql_lora_fine_tuning_fast_execute/)
* [7편 - Gemma 온디바이스 탑재 - 웹브라우저편 빠른실행](https://tykimos.github.io/2024/04/02/gemma_ondevice_webbrowser_fast_execute/)
* [8편 - Gemma 온디바이스 탑재 - 아이폰(iOS)편 빠른실행](https://tykimos.github.io/2024/04/05/local_llm_installed_on_my_iphone_gemma_2b/)