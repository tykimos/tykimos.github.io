---
layout: post
title: "ChatGPT+DALL·E로 이미지 편집"
author: Taeyoung Kim
date: 2024-4-3 20:20:07
categories: llm, chatgpt, dalle
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-4-3-image_editing_with_chatgpt_dall_e_title.jpeg
---

드디어 ChatGPT 서비스에서 DALL·E의 이미지 편집 기능이 가능해졌습니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-4-3-image_editing_with_chatgpt_dall_e_title.jpeg)

<iframe width="100%" height="400" src="https://youtube.com/embed/NvvS0qaYXTw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen=""></iframe>

### DALL·E API

DALL·E 2 API에서는 텍스트 to 이미지와 이미지 to 이미지 그리고 이미지 편집 기능을 제공하고 있었는데요. DALL·E 3는 텍스트 to 이미지를 제외한 API를 제공하고 있지 않아서 API 릴리즈 되기를 많이 기다렸습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-3-image_editing_with_chatgpt_dall_e_1.jpeg)

- 출처 : https://platform.openai.com/docs/guides/images/usage?context=node

### DALL·E API와 ChatGPT의 결합

이제 DALL·E API에서 제공하는 이미지 편집 기능이 ChatGPT에 드디어 탑재되었습니다. 이번에 ChatGPT가 업데이트되면서 생성된 이미지에 대해서 마스크를 지정하고 프롬프트로 해당 마스크에 대해서 이미지를 수정할 수 있게 되었습니다. 

### 새로운 이미지 편집 기능의 사용법

이 기능을 이용하면, 생성된 이미지를 클릭 한 후 편집 아이콘을 클릭하면, 수정하고자 하는 영역을 선택할 수 있습니다. 그 다음 선택된 영역에 대해 변경하고 싶은 내용을 프롬프트로 입력하면, 반영하여 새로운 이미지가 만들어집니다. 

이러한 과정은 다음과 같은 단계로 이루어집니다:

1. ChatGPT 서비스에 연동된 DALL·E를 통해 원하는 이미지를 생성합니다.
2. 생성된 이미지를 클릭한 후, 우측 상단의 편집 아이콘을 클릭합니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-3-image_editing_with_chatgpt_dall_e_2.jpeg)

3. 편집하고자 하는 영역을 선택합니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-3-image_editing_with_chatgpt_dall_e_3.jpeg)

4. 선택된 영역에 대해 변경하고 싶은 내용을 프롬프트로 입력합니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-3-image_editing_with_chatgpt_dall_e_4.jpeg)

5. 이를 ChatGPT에 전달하면, DALL·E API가 입력을 받아 새로운 이미지를 생성합니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-3-image_editing_with_chatgpt_dall_e_5.jpeg)

이제 사용자는 단순히 이미지를 생성하는 것 뿐만 아니라, 그 이미지를 자신의 의도에 맞게 편집하는 것도 가능해졌습니다.다만 이미지 생성 품질로 본다면 이미지 편집 모델은 DALL·E 3가 아니라 DALL·E 2인 것 같다는 예상은 조심스레 해봅니다.  

### 마무리

ChatGPT와 DALL·E의 결합은 단순히 텍스트에서 이미지를 생성하는 것을 넘어서, 사용자가 원하는 대로 이미지를 수정하고 개선할 수 있는 기능까지 제공하게 되었습니다. 앞으로 더 어떤 기능이 제공될 지 기대됩니다.

### 함께보기

- [BIC(Beyond Imagination Creations) Gallery 페이스북 그룹](https://www.facebook.com/groups/1366046607340589)