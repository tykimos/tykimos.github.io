---
layout: post
title: "Gemma 영한번역 LoRA 파인튜닝 빠른실행"
author: 김태영
date: 2024-2-22 03:00:00
categories: llm
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_en2ko_lora_fine_tuning_fast_execute_title_3.png
---
 
![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_en2ko_lora_fine_tuning_fast_execute_title_3.png)

이번에는 Gemma를 "databricks-dolly-15k.jsonl"과 "databricks-dolly-15k-ko.jsonl" 데이터셋을 이용해서 영한번역 LoRA 파인튜닝을 해보도록 하겠습니다.

### 함께보기

* 1편 - Gemma 시작하기 빠른실행 (추후 공개)
* [2편 - Gemma LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_lora_fine_tuning_fast_execute/)
* [3편 - Gemma 한국어 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_korean_lora_fine_tuning_fast_execute/)
* [4편 - Gemma 영한번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_en2ko_lora_fine_tuning_fast_execute/)
* [5편 - Gemma 한영번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_ko2en_lora_fine_tuning_fast_execute/)
* [6편 - Gemma 한국어 SQL챗봇 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/23/gemma_ko2sql_lora_fine_tuning_fast_execute/)
* [7편 - Gemma 온디바이스 탑재 - 웹브라우저편 빠른실행](https://tykimos.github.io/2024/04/02/gemma_ondevice_webbrowser_fast_execute/)
* [8편 - Gemma 온디바이스 탑재 - 아이폰(iOS)편 빠른실행](https://tykimos.github.io/2024/04/05/local_llm_installed_on_my_iphone_gemma_2b/)
* 9편 - Gemma 온디바이스 탑재 - 안드로이드편 빠른실행 (작업중)
* [10편 - RLHF 튜닝으로 향상된 Gemma 1.1 2B IT 공개](https://tykimos.github.io/2024/04/08/rlhf_tuning_enhanced_gemma_1.1_2b_it_release/)
* [11편 - 소스코드 생성 전용 - CodeGemma 시작하기](https://tykimos.github.io/2024/04/10/getting_started_with_codegemma/)

### databricks-dolly-15k 데이터셋

databricks-dolly-15k는 2023년 3월과 4월에 Databricks의 5,000명 이상의 직원이 작성한 15,000개의 고품질 인간 생성 프롬프트/응답 쌍을 포함하는 데이터셋입니다. 이 데이터셋은 큰 언어 모델의 지시 튜닝을 위해 특별히 설계되었으며, 훈련 레코드는 자연스럽고 표현력이 풍부하여 브레인스토밍 및 콘텐츠 생성부터 정보 추출 및 요약에 이르기까지 다양한 행동을 대표하도록 설계되었습니다

### databricks-dolly-15k-ko 데이터셋

databricks-dolly-15k-ko 데이터셋은 허깅페이스에서 다운로드 받을 수 있으며, NLP & AI - Korea University에서 databricks-dolly-15k를 DeepL API를 이용해서 한국어 번역을 수행한 파일입니다.

* [링크](https://huggingface.co/datasets/nlpai-lab/databricks-dolly-15k-ko)

#### 학습시간

LoRA 랭크 4로 1 에포크 시에 23.4분이 소요되었습니다. 사양은 구글코랩 T4입니다.

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_en2ko_lora_fine_tuning_fast_execute_1.png)

#### 수행결과 1

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_en2ko_lora_fine_tuning_fast_execute_2.png)

```
Instruction:
To quickly acquire skills, it is advisable to learn through rapid execution, in-depth analysis, and practical application in that order.

Response:
기술을 빨리 습득하기 위해서는 빠른 실행, 깊은 분석, 실제 적용 순으로 학습을 권한다.
```

#### 수행결과 2

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_en2ko_lora_fine_tuning_fast_execute_3.png)

```
Instruction:
What should I do on a trip to Europe?

Response:
유럽 여행에서 무엇을 할 수 있나요?
```

#### 수행결과 3

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_en2ko_lora_fine_tuning_fast_execute_4.png)

```
Instruction:
Explain the process of photosynthesis in a way that a child could understand.

Response:
어린이가 이해할 수 있는 방식으로 광합성 과정을 설명합니다.
```

### 더보기

LoRA 파인튜닝 공식 예제는 다음과 같습니다.

* [https://ai.google.dev/gemma/docs/lora_tuning](https://ai.google.dev/gemma/docs/lora_tuning)

### 추가문의

* 작성자 : 김태영
* 이메일 : tykim@aifactory.page

### 함께보기

* 1편 - Gemma 시작하기 빠른실행 (추후 공개)
* [2편 - Gemma LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_lora_fine_tuning_fast_execute/)
* [3편 - Gemma 한국어 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_korean_lora_fine_tuning_fast_execute/)
* [4편 - Gemma 영한번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_en2ko_lora_fine_tuning_fast_execute/)
* [5편 - Gemma 한영번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_ko2en_lora_fine_tuning_fast_execute/)
* [6편 - Gemma 한국어 SQL챗봇 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/23/gemma_ko2sql_lora_fine_tuning_fast_execute/)
* [7편 - Gemma 온디바이스 탑재 - 웹브라우저편 빠른실행](https://tykimos.github.io/2024/04/02/gemma_ondevice_webbrowser_fast_execute/)
* [8편 - Gemma 온디바이스 탑재 - 아이폰(iOS)편 빠른실행](https://tykimos.github.io/2024/04/05/local_llm_installed_on_my_iphone_gemma_2b/)
* 9편 - Gemma 온디바이스 탑재 - 안드로이드편 빠른실행 (작업중)
* [10편 - RLHF 튜닝으로 향상된 Gemma 1.1 2B IT 공개](https://tykimos.github.io/2024/04/08/rlhf_tuning_enhanced_gemma_1.1_2b_it_release/)
* [11편 - 소스코드 생성 전용 - CodeGemma 시작하기](https://tykimos.github.io/2024/04/10/getting_started_with_codegemma/)