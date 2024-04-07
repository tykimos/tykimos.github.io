---
layout: post
title: "RLHF 튜닝으로 향상된 Gemma 1.1 2B IT 공개"
author: Taeyoung Kim
date: 2024-4-7 17:33:21
categories: llm, gemma
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-4-7-rlhf_tuning_enhanced_gemma_1.1_2b_it_release_title.jpg
---

본 내용은 (어시+랭체인)에 의해 자동으로 작성된 글입니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-7-rlhf_tuning_enhanced_gemma_1.1_2b_it_release_title.jpg)
== start ==
안녕하세요, 여러분. 오늘은 Gemma의 새로운 릴리즈에 대해 소개할 예정입니다. 지난번 버전에서 발견된 몇 가지 문제를 해결하고, 훨씬 더 개선된 기능을 제공하기 위해 Gemma 1.1을 출시하게 되었습니다. 이번 업데이트를 통해 여러분은 RLHF(Reinforcement Learning with Human Feedback) 방법을 사용하여 훈련된 Gemma를 체험할 수 있습니다. 이 방법은 품질, 코딩 능력, 사실성, 지시 사항 따르기 및 멀티턴 대화 품질에 상당한 향상을 이끌어 냈습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-4-todayassi___blog_writing_assi_	itle.jpeg)

## 릴리즈 내용

Gemma 1.1은 새로운 RLHF 방법을 사용하여 훈련되었습니다. 이는 품질, 코딩 능력, 사실성, 지시 사항 따르기, 그리고 멀티턴 대화 품질에 상당한 향상을 이끌어 냈습니다. 또한, 이전 버전에서 발견된 몇 가지 버그를 수정하였으며, 모델 응답이 항상 "Sure,"로 시작하지 않도록 개선하였습니다.

## 성능 비교

Gemma 1.1은 이전 버전보다 훨씬 더 향상된 성능을 보여줍니다. RLHF 방법을 사용하여 훈련된 Gemma 1.1은 다양한 상황에서의 적응력과 반응성을 향상시켰습니다. 이로 인해 Gemma 1.1은 사용자의 요구에 더욱 정확하게 응답할 수 있게 되었습니다.

## 테스트 결과

우리는 이번 릴리즈가 대부분의 사용 사례에서 개선을 나타낼 것으로 믿지만, 사용자가 자신의 특정 응용 프로그램에서 테스트할 것을 권장합니다. 이전 모델은 동일한 리포지토리에서 계속 사용할 수 있습니다.

```python
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

이번 업데이트에는 사용자들의 피드백이 큰 도움이 되었습니다. Gemma의 열정적인 채택을 높이 평가하며, 커뮤니티로부터의 모든 피드백을 계속 환영합니다. 여러분의 소중한 의견을 계속 받고, 더 나은 Gemma를 만들기 위해 노력하겠습니다.

감사합니다.
== end ==
