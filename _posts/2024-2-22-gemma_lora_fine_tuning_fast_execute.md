---
layout: post
title: "Gemma LoRA 파인튜닝 빠른실행"
author: 김태영
date: 2024-2-22 00:00:00
categories: llm
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_lora_fine_tuning_fast_execute_title_5.png
---

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_lora_fine_tuning_fast_execute_title_5.png)

구글 Gemma가 공개되고 파인튜닝 예제 또한 제공되어 이를 간단하게 테스트 해봤습니다. 아래는 테스트 결과입니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_lora_fine_tuning_fast_execute_1.png)

"What should I do on a trip to Seoul?"이라고 물었을 때, "If you want to go to the most famous tourist destinations in Seoul, you should start with the Seoul Tower, which has the largest tower in Seoul. If you are looking for a good restaurant, you should go to Gwangjang Market, where you can taste Korean food."라고 답변을 해주세요.

### 함께보기

* 1편 - Gemma 시작하기 빠른실행 (추후 공개)
* [2편 - Gemma LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_korean_lora_fine_tuning_fast_execute/)
* [3편 - Gemma 한국어 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_korean_lora_fine_tuning_fast_execute/)
* [4편 - Gemma 영한번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_en2ko_lora_fine_tuning_fast_execute/)
* [5편 - Gemma 한영번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_ko2en_lora_fine_tuning_fast_execute/)

### 프리프로세서 및 토크나이저

- 프리프로세서: "gemma_causal_lm_preprocessor"는 입력 데이터를 모델이 처리할 수 있는 형태로 변환하는 데 사용됩니다.
- 토크나이저: GemmaTokenizer는 256,000개의 어휘를 포함하며, 텍스트를 모델이 이해할 수 있는 토큰으로 변환하는 역할을 합니다.

### 모델 아키텍처

- 입력 레이어: 모델은 padding_mask와 token_ids 두 가지 입력을 받습니다. 이들은 모델이 입력 텍스트를 올바르게 처리할 수 있도록 정보를 제공합니다.
- GemmaBackbone: 모델의 핵심으로, 2,507,536,384개의 매개변수를 가지고 있으며, 각 입력 토큰에 대해 2048차원의 표현을 생성합니다.
- Token Embedding: ReversibleEmbedding 레이어는 GemmaBackbone의 출력을 사용하여 256,000개의 어휘 크기에 해당하는 임베딩을 생성합니다. 이는 모델의 예측을 실제 단어로 변환하는 데 필요합니다.

### 파인튜닝

![img](http://tykimos.github.io/warehouse/2024/2024-2-22-gemma_lora_fine_tuning_fast_execute_2.png)

모델로부터 더 나은 응답을 얻기 위해, Databricks Dolly 15k 데이터셋을 사용하여 모델을 Low Rank Adaptation (LoRA)으로 파인 튜닝합니다. LoRA 랭크는 LLM의 원래 가중치에 추가되는 학습 가능한 행렬의 차원성을 결정합니다. 이것은 파인 튜닝 조정의 표현력과 정밀도를 제어할 수 있습니다. 

- 높은 랭크는 더 세부적인 변경이 가능함을 의미하지만, 또한 더 많은 학습 가능한 매개변수를 의미합니다.
- 낮은 랭크는 더 적은 계산 오버헤드를 의미하지만, 잠재적으로 덜 정확한 적응을 의미할 수 있습니다.

제공된 소스에서는 LoRA 랭크로 4로 설정한 사항은 아래와 같습니다.

- 총 매개변수 수: 2,508,603,394개로, 이 중 9.35GB가 모델의 크기를 나타냅니다.
- 훈련 가능한 매개변수: 533,504개 (약 2.04MB), 모델을 훈련하면서 조정될 수 있는 매개변수의 수입니다.
- 훈련 불가능한 매개변수: 2,507,002,880개 (약 9.34GB), 훈련 과정에서 고정되어 변경되지 않는 매개변수의 수입니다.
- 최적화 매개변수: 1,067,010개 (약 4.07MB), 모델 훈련을 최적화하는 데 사용되는 매개변수입니다.
드라이 런: 어떤 작업을 실제로 수행하기 전에, 그 작업을 수행할 준비가 되었는지 확인하기 위해 실행하는 테스트 과정입니다.

#### Databricks Dolly 15k 데이터셋이란

Databricks Dolly 15k는 2023년 3월과 4월에 Databricks의 5,000명 이상의 직원이 작성한 15,000개의 고품질 인간 생성 프롬프트/응답 쌍을 포함하는 데이터셋입니다. 이 데이터셋은 큰 언어 모델의 지시 튜닝을 위해 특별히 설계되었으며, 훈련 레코드는 자연스럽고 표현력이 풍부하여 브레인스토밍 및 콘텐츠 생성부터 정보 추출 및 요약에 이르기까지 다양한 행동을 대표하도록 설계되었습니다

#### LoRA 기법인란?

LoRA (Low-Rank Adaptation)는 대규모 언어 모델(Large Language Models, LLM)을 효율적으로 미세 조정하기 위한 기술입니다. 이 방법은 모델의 원본 가중치에 저차원의 학습 가능한 행렬을 추가함으로써 작동합니다. LoRA는 추가적인 추론 시간 없이 제공되며, 이는 생산 환경에서 사용할 때 W' = W + BA를 명시적으로 계산하여 결과를 저장함으로써, 평소처럼 추론을 수행할 수 있음을 보장합니다. 이 기법은 태스크 전환을 더 쉽고 빠르게 할 수 있게 해주며, 다양한 태스크에 대해 맞춤화된 모델을 쉽게 생성하고 교체할 수 있도록 합니다. LoRA는 기존의 효율적인 미세 조정 기술을 상당한 차이로 일반적으로 능가하며, 전체 미세 조정과 비교할 때 비슷하거나 더 나은 성능을 제공하는 것으로 나타났습니다.

### 링크

LoRA 파인튜닝 공식 예제는 다음과 같습니다.

* [https://ai.google.dev/gemma/docs/lora_tuning](https://ai.google.dev/gemma/docs/lora_tuning)

### 추가문의

* 작성자 : 김태영
* 이메일 : tykim@aifactory.page