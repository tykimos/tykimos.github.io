---
layout: post
title: "ChatGPT+DALL·E로 이미지 편집"
author: Taeyoung Kim
date: 2024-4-3 20:20:07
categories: llm, chatgpt, dalle
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-4-3-image_editing_with_chatgpt_dall_e_title.jpeg
---

본 내용은 (어시+랭체인)에 의해 자동으로 작성된 글입니다.

![img](http://tykimos.github.io/warehouse/2024/2024-4-3-image_editing_with_chatgpt_dall_e_title.jpeg)
# DALL·E API와 ChatGPT가 만나다: 새로운 이미지 편집 기능 탑재 소식

기술 세계에서 가장 흥미진진한 것 중 하나는 두 가지 강력한 기술이 결합하여 더욱 획기적인 기능을 제공하는 것입니다. 이번에는 인공지능 기반의 이미지 생성 API인 DALL·E와 대화형 자연어 처리 모델인 ChatGPT가 결합하여 사용자가 원하는 대로 이미지를 편집할 수 있는 새로운 기능이 탑재되었습니다.

## 이미지 편집 기능이란?

이미지 편집 기능은 생성된 이미지를 사용자의 요구에 따라 수정하는 기능입니다. 이를 통해 사용자는 이미지의 특정 영역을 선택하고, 그 영역을 변경하여 새로운 이미지를 만들 수 있습니다. 이 기능은 원래 이미지에 추가, 삭제, 변경 등 다양한 편집 작업을 수행할 수 있습니다.

```python
# DALL·E API와 ChatGPT를 이용한 이미지 편집
# 예제 코드

# 1. 이미지 생성
image = DALLE.create_image(prompt="apple")

# 2. 이미지 편집
edited_image = image.edit(prompt="change the color to red")

# 3. 이미지 확인
edited_image.show()
```

## DALL·E API의 이미지 편집 기능 활용

이제 DALL·E API에서 제공하는 이미지 편집 기능이 ChatGPT에 탑재되어 사용자는 생성된 이미지를 클릭 한 후 편집 아이콘을 클릭하여 원하는 영역을 선택하고, 그 다음 프롬프트로 변경하고 싶은 내용을 입력하여 새로운 이미지를 만들 수 있습니다.

```python
# DALL·E API와 ChatGPT를 이용한 이미지 편집
# 예제 코드

# 1. 이미지 생성
image = DALLE.create_image(prompt="apple")

# 2. 이미지 선택 및 편집 아이콘 클릭
selected_area = image.select_area()

# 3. 프롬프트 입력 및 이미지 편집
edited_image = selected_area.edit(prompt="change the color to red")

# 4. 이미지 확인
edited_image.show()
```

## 마치며

이렇게 DALL·E API와 ChatGPT가 결합하여 새롭게 탑재된 이미지 편집 기능은 사용자가 원하는 대로 이미지를 수정하고 새롭게 생성하는데 매우 유용한 기능입니다. 이 기능을 활용하면 사용자는 자신만의 독특한 이미지를 쉽게 만들어낼 수 있을 것입니다. 앞으로 이 두 기술이 어떤 신기한 기능을 만들어낼지 기대해봅니다.
