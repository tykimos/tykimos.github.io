---
layout: post
title: "소스코드 생성 전용 - CodeGemma 시작하기"
author: Taeyoung Kim
date: 2024-4-10 14:56:03
categories: llm gemma codegemma
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-4-10-getting_started_with_codegemma_title.jpg
---

![img](http://tykimos.github.io/warehouse/2024/2024-4-10-getting_started_with_codegemma_title.jpg)

CodeGemma는 Gemini 모델을 기반으로 제작되어, 경량화되었으며 기기 내에서도 작동 가능한 모델입니다. 주로 코드 관련 5000억 개 이상의 토큰으로 훈련되었으며, Gemma 모델 계열과 동일한 아키텍처를 사용합니다. 이에 따라, CodeGemma 모델은 코드 완성 및 생성 작업 모두에서 최신 기술의 성능을 달성하면서도, 다양한 규모에서 강력한 이해력과 추론 능력을 유지한다고 합니다.

CodeGemma는 다음과 같은 3가지 버전으로 제공됩니다:

- 7B 코드 사전 훈련 모델
- 7B 지시 기반 튜닝 코드 모델
- 코드 채우기 및 개방형 생성을 위해 특별히 훈련된 2B 모델.

이 가이드에서는 KerasNLP를 활용하여 CodeGemma 2B 모델로 코드 완성 작업을 수행하는 방법에 대해 안내합니다. 추가로 구구단 예제를 테스트하고 모델의 출력에서 완성된 코드를 추출하는 것까지 해보겠습니다.

본 게시물은 아래 링크를 기반으로 작성되었습니다. (Copyright 2024 Google LLC.)

- https://ai.google.dev/gemma/docs/codegemma/keras_quickstart

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


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### CodeGemma 접근권한 얻기

먼저 Gemma 설정에서 설정 지침을 완료해야 합니다. Gemma 설정 지침은 다음을 수행하는 방법을 보여줍니다:

- [캐글](kaggle.com)을 가입합니다.
- https://www.kaggle.com/models/google/gemma 에 접속하여 권한 요청을 합니다.
- Kaggle 사용자 이름과 API 키를 생성하고 구성하기 위해 Kaggle 사용자 프로필의 계정 탭으로 이동하여 새 토큰 생성을 선택합니다.
- API 자격 증명이 포함된 kaggle.json 파일을 다운로드 할 수 있습니다.


### 코랩 환경 설정

CodeGemma 2B 모델을 실행할 수 있는 충분한 리소스를 갖춘 Colab 런타임이 필요합니다. 이 경우 T4 GPU를 사용할 수 있습니다:

- **메뉴** - **런타임** - **런타임 유형 변경**을 선택합니다.
- 하드웨어 가속기에서 **T4 GPU**를 선택합니다.

다음은 API 키를 구성합니다.

- 다운로드 한 kaggle.json을 열어보면, Kaggle 사용자 이름과 Kaggle API 정보를 담고 있습니다.
- 코랩에서 왼쪽 패널의 보안비밀(🔑)을 선택하고 키를 추가합니다.
- Kaggle 사용자 이름은 KAGGLE_USERNAME으로 저장합니다.
- Kaggle API는 KAGGLE_KEY로 저장합니다.

### 환경변수 설정하기

위에서 저장한 `KAGGLE_USERNAME`과 `KAGGLE_KEY`을 환경변수로 설정합니다.


```
import os
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

### 필요한 패키지 설치


```
!pip install -q -U keras-nlp
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m508.4/508.4 kB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m950.8/950.8 kB[0m [31m42.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.2/5.2 MB[0m [31m23.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m589.8/589.8 MB[0m [31m2.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.8/4.8 MB[0m [31m98.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m88.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.5/5.5 MB[0m [31m96.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m67.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m311.2/311.2 kB[0m [31m30.6 MB/s[0m eta [36m0:00:00[0m
    [?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tf-keras 2.15.1 requires tensorflow<2.16,>=2.15, but you have tensorflow 2.16.1 which is incompatible.[0m[31m
    [0m

### 백엔드 선택

케라스는 TensorFlow, JAX, 또는 PyTorch 중 하나를 백엔드로 선택하여 워크플로우를 실행할 수 있습니다. 이번 예제에서는 TensorFlow를 백엔드로 구성합니다.


```
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch".
```

### 패키지 가져오기

필요한 패키지는 Keras와 KerasNLP 입니다.


```
import keras_nlp
import keras

# Run at half precision.
keras.config.set_floatx("bfloat16")
```

### 모델 로드

인과적 언어 모델(causal language modeling)을 위한 엔드 투 엔드 Gamma 모델인 `GemmaCausalLM`을 사용합니다. `from_preset`을 이용해서 모델을 생성합니다. `from_preset` 함수는 사전 설정된 아키텍처와 가중치를 기반으로 모델을 인스턴스화합니다. 위 코드에서 문자열 code_gemma_2b_en은 사전 설정 아키텍처를 지정합니다. 20억 개의 매개변수를 가진 CodeGemma 모델입니다. 70억 개의 매개변수를 가진 CodeGemma 7B 모델도 사용할 수 있으나 Colab에서 더 큰 모델을 실행하려면 유료 플랜에서 제공하는 프리미엄 GPU에 대한 접근이 필요합니다.


```
# code_gemma_2b_en 모델을 불러옵니다.

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("code_gemma_2b_en")
gemma_lm.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Preprocessor: "gemma_causal_lm_preprocessor_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Tokenizer (type)                                   </span>┃<span style="font-weight: bold">                                             Vocab # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ gemma_tokenizer (<span style="color: #0087ff; text-decoration-color: #0087ff">GemmaTokenizer</span>)                   │                                             <span style="color: #00af00; text-decoration-color: #00af00">256,000</span> │
└────────────────────────────────────────────────────┴─────────────────────────────────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "gemma_causal_lm_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                  </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">         Param # </span>┃<span style="font-weight: bold"> Connected to               </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)              │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_ids (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)              │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gemma_backbone                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)        │   <span style="color: #00af00; text-decoration-color: #00af00">2,506,172,416</span> │ padding_mask[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],        │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GemmaBackbone</span>)               │                           │                 │ token_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ token_embedding               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256000</span>)      │     <span style="color: #00af00; text-decoration-color: #00af00">524,288,000</span> │ gemma_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">ReversibleEmbedding</span>)         │                           │                 │                            │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,506,172,416</span> (4.67 GB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,506,172,416</span> (4.67 GB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



### 중간 채우기 코드 완성 태스크

본 예제는 CodeGemma의 중간 채우기(Fill-in-the-Middle, FIM) 기능을 사용하여 주변 컨텍스트를 기반으로 코드를 완성합니다. 이는 코드 에디터 애플리케이션에서 커서 전후의 코드를 바탕으로 텍스트 커서 위치에 코드를 삽입하는 데 특히 유용합니다. CodeGemma는 4개의 사용자 정의 토큰을 사용하도록 허용합니다 - FIM을 위한 3개와 다중 파일 컨텍스트 지원을 위한 토큰입니다.


```
BEFORE_CURSOR = "<|fim_prefix|>"
AFTER_CURSOR = "<|fim_suffix|>"
AT_CURSOR = "<|fim_middle|>"
FILE_SEPARATOR = "<|file_separator|>"
```

모델을 위한 정지 토큰을 정의합니다.



```
END_TOKEN = gemma_lm.preprocessor.tokenizer.end_token

stop_tokens = (BEFORE_CURSOR, AFTER_CURSOR, AT_CURSOR, FILE_SEPARATOR, END_TOKEN)

stop_token_ids = tuple(gemma_lm.preprocessor.tokenizer.token_to_id(x) for x in stop_tokens)
```

코드 완성을 위한 프롬프트 포맷을 아래 가이드에 따라 정의합니다.

- FIM 토큰과 접두사 및 접미사 사이에는 공백이 없어야 합니다.
- 모델이 채우기를 계속하도록 유도하기 위해 FIM 중간 토큰은 끝에 있어야 합니다.
- 현재 파일에서 커서의 위치나 모델에 제공하고자 하는 컨텍스트의 양에 따라 접두사나 접미사는 비어 있을 수 있습니다.

프롬프트 구성을 위한 함수를 정의합니다. 이 함수는 before 코드와 after 코드를 입력해주면, 모델에게 전달할 프롬프트를 구성해줍니다.


```
def format_completion_prompt(before, after):
    return f"{BEFORE_CURSOR}{before}{AFTER_CURSOR}{after}{AT_CURSOR}"

before = "import "
after = """if __name__ == "__main__":\n    sys.exit(0)"""
prompt = format_completion_prompt(before, after)
print(prompt)
```

    <|fim_prefix|>import <|fim_suffix|>if __name__ == "__main__":
        sys.exit(0)<|fim_middle|>


즉 아래처럼 프롬프트를 구성합니다.

- fim_prefix
- 이전 코드
- fim_suffix
- 이후 코드
- fim_middle

Run the prompt. It is recommended to stream response tokens. Stop streaming upon encountering any of the user-defined or end of turn/senetence tokens to get the resulting code completion.

그 다음은 프롬프트를 gemma 모델을 이용하여 실행시킵니다. stop_token_ids을 정의하여 정지 토큰이나 문장의 끝을 감지하면 해당 부분에서 생성을 중단하도록 합니다.


```
gemma_lm.generate(prompt, stop_token_ids=stop_token_ids, max_length=128)
```




    '<|fim_prefix|>import <|fim_suffix|>if __name__ == "__main__":\n    sys.exit(0)<|fim_middle|>sys\n<|file_separator|>'



모델이 코드 완성을 위해 `sys`을 제안해줍니다.

### 추가 예제

구구단 코드를 완성하는 예제를 테스트해보겠습니다. 구구단 일부 코드를 입력하여 중간 코드를 생성합니다.


```
before = """
def print_multiplication_table():
    for i in range(1, 10):
"""

after = """
if __name__ == "__main__":
    print_multiplication_table()
"""
prompt = format_completion_prompt(before, after)
response = gemma_lm.generate(prompt, stop_token_ids=stop_token_ids, max_length=128)

print(response)
```

    <|fim_prefix|>
    def print_multiplication_table():
        for i in range(1, 10):
    <|fim_suffix|>
    if __name__ == "__main__":
        print_multiplication_table()
    <|fim_middle|>        for j in range(1, 10):
                print(f"{i} x {j} = {i * j}")
            print()<|file_separator|>


모델의 출력에서 중간 코드를 추출하는 함수를 정의합니다.


```
def extract_inner_code(code, start_token =  "<|fim_middle|>", end_token = "<|file_separator|>"):
    """
    주어진 코드에서 시작 토큰과 종료 토큰 사이의 내부 코드를 추출합니다.

    Args:
    - code (str): 분석할 전체 코드 문자열
    - start_token (str): 내부 코드의 시작을 나타내는 토큰
    - end_token (str): 내부 코드의 종료를 나타내는 토큰

    Returns:
    - str: 시작 토큰과 종료 토큰 사이에 위치한 내부 코드 문자열
    """
    start_index = code.find(start_token) + len(start_token)
    end_index = code.find(end_token, start_index)
    inner_code = code[start_index:end_index]

    return inner_code
```

extract_inner_code() 함수를 이용하여 중간코드를 추출합니다.


```
inner_code = extract_inner_code(response)
```


```
print(inner_code)
```

            for j in range(1, 10):
                print(f"{i} x {j} = {i * j}")
            print()


전체 코드를 구성하여 추출합니다.


```
full_code = before + inner_code + after
```


```
print(full_code)
```

    
    def print_multiplication_table():
        for i in range(1, 10):
            for j in range(1, 10):
                print(f"{i} x {j} = {i * j}")
            print()
    if __name__ == "__main__":
        print_multiplication_table()
    


### 마무리

CodeGemma를 이용하여 주변 컨텍스트를 기반으로 코드를 채우는 방법에 대해 살펴봤습니다. 다음 링크를 함께 보시는 것을 추천드립니다.

- [AI Assisted Programming with CodeGemma and KerasNLP notebook](https://ai.google.dev/gemma/docs/codegemma/code_assist_keras)
- [CodeGemma model card](https://ai.google.dev/gemma/docs/codegemma/model_card)

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