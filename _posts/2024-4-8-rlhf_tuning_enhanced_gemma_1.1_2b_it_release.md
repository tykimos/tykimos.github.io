---
layout: post
title: "RLHF 튜닝으로 향상된 Gemma 1.1 2B IT 공개"
author: Taeyoung Kim
date: 2024-4-8 01:53:12
categories: llm, gemma
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-4-8-rlhf_tuning_enhanced_gemma_1.1_2b_it_release_title.jpg
---

![img](http://tykimos.github.io/warehouse/2024/2024-4-8-rlhf_tuning_enhanced_gemma_1.1_2b_it_release_title.jpg)

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

이 문서는 https://huggingface.co/google/gemma-1.1-2b-it 기반으로 제작되었습니다.

## 필요한 패키지 설치

아래 3개의 패키지를 설치합니다. 만약에 경고 메시지에 런타임 재시작해야한다는 메시지가 나온다면 [메뉴] > [런타임] > [세션 다시 시작]을 클릭하여 런타임을 재시작합니다.


```python
!pip install torch          # PyTorch, 딥러닝 프레임워크를 설치합니다.
!pip install transformers   # Hugging Face의 Transformers 라이브러리를 설치합니다.
!pip install accelerate     # Hugging Face의 Accelerate 라이브러리를 설치합니다.
```

    Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.2.1+cu121)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch) (3.13.3)
    Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch) (4.10.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.3)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2023.6.0)
    Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch)
      Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.7/23.7 MB[0m [31m38.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-runtime-cu12==12.1.105 (from torch)
      Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m823.6/823.6 kB[0m [31m58.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-cupti-cu12==12.1.105 (from torch)
      Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.1/14.1 MB[0m [31m48.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cudnn-cu12==8.9.2.26 (from torch)
      Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m731.7/731.7 MB[0m [31m1.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cublas-cu12==12.1.3.1 (from torch)
      Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m410.6/410.6 MB[0m [31m2.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cufft-cu12==11.0.2.54 (from torch)
      Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m121.6/121.6 MB[0m [31m5.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-curand-cu12==10.3.2.106 (from torch)
      Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.5/56.5 MB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusolver-cu12==11.4.5.107 (from torch)
      Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.2/124.2 MB[0m [31m7.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cusparse-cu12==12.1.0.106 (from torch)
      Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m196.0/196.0 MB[0m [31m2.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nccl-cu12==2.19.3 (from torch)
      Downloading nvidia_nccl_cu12-2.19.3-py3-none-manylinux1_x86_64.whl (166.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m166.0/166.0 MB[0m [31m5.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nvtx-cu12==12.1.105 (from torch)
      Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m99.1/99.1 kB[0m [31m14.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: triton==2.2.0 in /usr/local/lib/python3.10/dist-packages (from torch) (2.2.0)
    Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch)
      Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m18.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.5)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)
    Installing collected packages: nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12
    Successfully installed nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.19.3 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.1.105
    Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.38.2)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.13.3)
    Requirement already satisfied: huggingface-hub<1.0,>=0.19.3 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.25.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2023.12.25)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.31.0)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.15.2)
    Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.2)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.2)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers) (2023.6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers) (4.10.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.2.2)
    Collecting accelerate
      Downloading accelerate-0.29.1-py3-none-any.whl (297 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m297.3/297.3 kB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from accelerate) (1.25.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (24.0)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate) (5.9.5)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from accelerate) (6.0.1)
    Requirement already satisfied: torch>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (2.2.1+cu121)
    Requirement already satisfied: huggingface-hub in /usr/local/lib/python3.10/dist-packages (from accelerate) (0.20.3)
    Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from accelerate) (0.4.2)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.13.3)
    Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (4.10.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.1.3)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2023.6.0)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.105)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.105)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.105)
    Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (8.9.2.26)
    Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.3.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (11.0.2.54)
    Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (10.3.2.106)
    Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (11.4.5.107)
    Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.0.106)
    Requirement already satisfied: nvidia-nccl-cu12==2.19.3 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2.19.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (12.1.105)
    Requirement already satisfied: triton==2.2.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2.2.0)
    Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.10/dist-packages (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.10.0->accelerate) (12.4.127)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->accelerate) (2.31.0)
    Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->accelerate) (4.66.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.10.0->accelerate) (2.1.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->accelerate) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->accelerate) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->accelerate) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->accelerate) (2024.2.2)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.10.0->accelerate) (1.3.0)
    Installing collected packages: accelerate
    Successfully installed accelerate-0.29.1
    Collecting qrcode
      Downloading qrcode-7.4.2-py3-none-any.whl (46 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46.2/46.2 kB[0m [31m895.7 kB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from qrcode) (4.10.0)
    Collecting pypng (from qrcode)
      Downloading pypng-0.20220715.0-py3-none-any.whl (58 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.1/58.1 kB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: pypng, qrcode
    Successfully installed pypng-0.20220715.0 qrcode-7.4.2


## 허깅페이스 접근 토큰 발행

Hugging Face 인증을 위한 환경 변수를 설정합니다. 'HF_TOKEN'은 Hugging Face 토큰을 저장하고 있는 것으로 가정한 사용자 특정 환경 변수입니다.

![colab user data key setting](http://tykimos.github.io/warehouse/2024/2024-4-8-rlhf_tuning_enhanced_gemma_1.1_2b_it_release_3.png)

## 허깅페이스 접근 토큰 설정

허깅페이스 접근 토큰을 발행하였다면 'HF_TOKEN'이름으로 환경 변수로 등록합니다. 코랩에서 좀 더 쉽게 사용할 수 있도록 코랩의 "보안 비밀" 기능을 사용하여 키 값을 가지고 옵니다. 이렇게 하면 소스코드 공유 시에 키를 공유하지 않아도 되며, 다른 코랩 소스코드 사용할 때도 재사용이 용이합니다.

![colab user data key setting](http://tykimos.github.io/warehouse/2024/2024-4-8-rlhf_tuning_enhanced_gemma_1.1_2b_it_release_2.png)

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
```

## 모델 준비하기

"google/gemma-1.1-2b-it"에 대한 토크나이저와 모델을 준비합니다. 아래와 같이 필요한 파일을 다운로드 받습니다. 가장 큰 파일이 4.95G 정도 되네요.

![downloading from huggingface](http://tykimos.github.io/warehouse/2024/2024-4-8-rlhf_tuning_enhanced_gemma_1.1_2b_it_release_1.png)


```python
# Hugging Face의 Transformers 라이브러리에서 토크나이저와 모델 클래스를 임포트합니다.
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Hugging Face 모델 저장소에서 토크나이저와 모델을 로드합니다.
# 모델 식별자 'google/gemma-1.1-2b-it'는 로드할 모델을 지정합니다.
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",   # 모델 식별자
    device_map="auto",          # 모델을 사용 가능한 장치(GPU/CPU)에 자동으로 매핑합니다.
    torch_dtype=torch.float16,  # 메모리 사용을 줄이고 빠르게 계산하기 위해 데이터 타입을 float16으로 설정합니다.
    revision="float16",         # float16을 지원하는 특정 모델 리비전을 사용합니다.
)
```


    tokenizer_config.json:   0%|          | 0.00/40.6k [00:00<?, ?B/s]
    tokenizer.model:   0%|          | 0.00/4.24M [00:00<?, ?B/s]
    tokenizer.json:   0%|          | 0.00/17.5M [00:00<?, ?B/s]
    special_tokens_map.json:   0%|          | 0.00/636 [00:00<?, ?B/s]
    config.json:   0%|          | 0.00/618 [00:00<?, ?B/s]
    model.safetensors.index.json:   0%|          | 0.00/13.5k [00:00<?, ?B/s]
    Downloading shards:   0%|          | 0/2 [00:00<?, ?it/s]
    model-00001-of-00002.safetensors:   0%|          | 0.00/4.95G [00:00<?, ?B/s]
    model-00002-of-00002.safetensors:   0%|          | 0.00/67.1M [00:00<?, ?B/s]
    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
    generation_config.json:   0%|          | 0.00/132 [00:00<?, ?B/s]

```python
# 모델이 머신 러닝에 관한 시를 작성하도록 입력 텍스트를 정의합니다.
input_text = "Write me a poem about Machine Learning."

# 입력 텍스트를 토크나이즈하고 텐서로 준비합니다.
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda") # PyTorch 텐서로 변환하고 CUDA(GPU)로 보냅니다.

# input_ids를 기반으로 모델에서 토큰 시퀀스를 생성합니다.
outputs = model.generate(**input_ids)

# 생성된 토큰을 문자열로 디코딩하고 출력합니다.
print(tokenizer.decode(outputs[0]))
```

    /usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py:1178: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
      warnings.warn(


    <bos>Write me a poem about Machine Learning.
    In circuits of logic, a mind unseen,


위 코드 실행 결과가 아래 처럼 출력되다가 중단되어 나옵니다.

```
<bos>Write me a poem about Machine Learning.
In circuits of logic, a mind unseen,
```

그 이유는 model.generate() 함수에서 max_length의 기본값이 20으로 설정되어 있기 때문입니다.

- max_length란 (입력 텍스트의 길이 + 출력 텍스트의 길이)의 합의 최대 토큰 수를 정의합니다. 즉 입력과 출력을 모두 더해서 max_length을 넘기지 않도록 생성됩니다.

모델의 결과가 충분히 작성될 수 있도록 max_length를 512로 설정하여 다시 호출해봅니다.


```python
input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# input_ids를 기반으로 모델에서 토큰 시퀀스를 생성합니다.
# 생성된 시퀀스는 입력 길이도 포함하여 최대 512 토큰 길이를 가집니다.
outputs = model.generate(**input_ids, max_length=512)
print(tokenizer.decode(outputs[0]))
```

    <bos>Write me a poem about Machine Learning.
    
    In circuits of logic, a mind unseen,
    A tapestry of data, a world serene.
    Algorithms dance, a symphony of code,
    Learning from experience, a journey bold.
    
    From images that speak, to numbers that bind,
    Machines learn, their knowledge entwined.
    Predictive models, with foresight so bright,
    Insights hidden, in the digital night.
    
    In self-driving cars, their sensors keen,
    Predicting paths, a seamless scene.
    Natural language processing, a bridge so wide,
    Understanding words, their meaning inside.
    
    But with power comes responsibility,
    Bias lurking, in the algorithms' abyss.
    Transparency elusive, a challenge to face,
    As machines learn, their consequences embrace.
    
    Yet, in this labyrinth of digital dreams,
    A future emerges, both wondrous and streams.
    A world where machines and humans unite,
    Creating a harmony, a technological delight.<eos>


정상적으로 결과가 출력되는 것을 확인할 수 있습니다. `<eos`> 토큰이 마지막에 출력되는데요. 이 특수토큰에 대한 설명은 아래와 같습니다.

`<bos`>와 `<eos`>는 특수한 토큰(token)으로, 각각 "beginning of sequence"와 "end of sequence"를 의미합니다. 이 토큰들은 자연어 처리에서 문장이나 텍스트의 시작과 끝을 나타내는 데 사용됩니다. 모델이 텍스트를 생성하거나 해석할 때 이 토큰들을 기준으로 문장의 시작과 끝을 구분하게 됩니다.

- `<bos`>: 문장이나 텍스트 입력의 시작 부분에 사용되며, 모델이 새로운 텍스트를 생성하기 시작할 때의 시작점으로 사용됩니다.
- `<eos`>: 문장이나 텍스트 입력의 끝 부분에 사용되며, 모델이 텍스트 생성을 중단해야 할 시점을 나타냅니다.

이러한 토큰들은 모델이 주어진 입력에 대해 문맥을 더 잘 이해하고, 적절한 길이의 출력을 생성하는 데 도움을 줍니다. 예를 들어, 텍스트를 생성하는 언어 모델에서 `<eos`> 토큰이 등장하면, 모델은 그 시점에서 문장 생성을 멈추고 그것을 출력의 끝으로 간주합니다.

### 코드 작성 테스트

문자열에서 각 단어의 빈도수를 세는 파이썬 코드를 작성해달라고 요청해보겠습니다. 문자열은 앞서 생성했던 시를 예시로 들어보겠습니다.


```python
text = """
In circuits of logic, a mind unseen,
A tapestry of data, a world serene.
Algorithms dance, a symphony of code,
Learning from experience, a journey bold.
"""

input_text = f"Write a Python code to count the frequency of each word in the {text} and print the words in descending order of frequency."

input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_length=512)
print(tokenizer.decode(outputs[0]))
```

    <bos>Write a Python code to count the frequency of each word in the 
    In circuits of logic, a mind unseen,
    A tapestry of data, a world serene.
    Algorithms dance, a symphony of code,
    Learning from experience, a journey bold.
     and print the words in descending order of frequency.
    
    ```python
    from collections import Counter
    
    text = """
    In circuits of logic, a mind unseen,
    A tapestry of data, a world serene.
    Algorithms dance, a symphony of code,
    Learning from experience, a journey bold.
    """
    
    # Count the frequency of each word
    word_counts = Counter(text.split())
    
    # Sort the words in descending order of frequency
    sorted_words = word_counts.most_common(10)
    
    # Print the words
    for word, count in sorted_words:
        print(f"{word}: {count}")
    ```
    
    Output:
    
    ```
    Algorithms: 3
    code: 2
    data: 2
    journey: 2
    mind: 2
    process: 1
    sequence: 1
    tapestries: 1
    """
    ```<eos>


파이썬 코드블록을 사용하여 파이썬 코드가 생성되었습니다. 그럼 생성된 코드가 제대로 동작하는 지 테스트해보겠습니다.




```python
from collections import Counter

text = """
In circuits of logic, a mind unseen,
A tapestry of data, a world serene.
Algorithms dance, a symphony of code,
Learning from experience, a journey bold.
"""

# Count the frequency of each word
word_counts = Counter(text.split())

# Sort the words in descending order of frequency
sorted_words = word_counts.most_common(10)

# Print the words
for word, count in sorted_words:
    print(f"{word}: {count}")
```

    a: 4
    of: 3
    In: 1
    circuits: 1
    logic,: 1
    mind: 1
    unseen,: 1
    A: 1
    tapestry: 1
    data,: 1


정상적으로 작동됩니다. 그럼 이제 전체 시를 입력하고 테스트 해보겠습니다.


```python
from collections import Counter

text = """
In circuits of logic, a mind unseen,
A tapestry of data, a world serene.
Algorithms dance, a symphony of code,
Learning from experience, a journey bold.

From images that speak, to numbers that bind,
Machines learn, their knowledge entwined.
Predictive models, with foresight so bright,
Insights hidden, in the digital night.

In self-driving cars, their sensors keen,
Predicting paths, a seamless scene.
Natural language processing, a bridge so wide,
Understanding words, their meaning inside.

But with power comes responsibility,
Bias lurking, in the algorithms' abyss.
Transparency elusive, a challenge to face,
As machines learn, their consequences embrace.

Yet, in this labyrinth of digital dreams,
A future emerges, both wondrous and streams.
A world where machines and humans unite,
Creating a harmony, a technological delight.
"""

# Count the frequency of each word
word_counts = Counter(text.split())

# Sort the words in descending order of frequency
sorted_words = word_counts.most_common(10)

# Print the words
for word, count in sorted_words:
    print(f"{word}: {count}")
```

    a: 9
    of: 4
    their: 4
    A: 3
    in: 3
    In: 2
    world: 2
    that: 2
    to: 2
    learn,: 2


### 마무리

이번에 릴리즈된 gemma 1.1 2b it 모델을 테스트 해봤습니다. 온디바이스나 간단한 태스크 처리에 2B 모델의 활약이 기대되는 만큼 앞으로 여러가지 테스트를 해보면서 활용 사례를 늘려가보겠습니다.

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