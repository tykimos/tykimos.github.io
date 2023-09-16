---
layout: post
title: "AI로 흉기 검출 - Grounding Dino 활용"
author: 김태영
date: 2023-8-4 00:00:00
categories: ai
comments: true
image: https://tykimos.github.io/warehouse/2023/2023-8-4-weapon_detection_with_grounding_dino_title.png
---

### Grounding Dino를 이용한 흉기 검출

충격적인 서현역 흉기난동 사건이 발생했습니다. 이런 사건을 방지하거나 빨리 인지하려면 어떻게 해야할까 고민 중에, Grounding Dino를 활용하여 흉기를 검출하는 모델을 만들어봤습니다. Grounding Dino는 객체를 검출하고 그 객체에 대한 설명을 생성하는 작업을 수행하는 AI 모델입니다. 이번 프로젝트에서는 Grounding Dino를 사용하여 동영상의 각 프레임에서 흉기 객체를 검출하고, 그 결과를 분석하는 작업을 수행하였습니다.

흉기가 검출된 프레임만 모아서 움직이는 GIF로 만들어봤습니다. 

![img](https://tykimos.github.io/warehouse/2023/2023-8-4-weapon_detection_with_grounding_dino_1.gif)

전체 영상에서 검출 결과를 보시려면 아래 동영상을 참고하세요.

<iframe width="100%" height="400" src="https://www.youtube.com/embed/8JsofBOSuEk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Grounding Dino 소개

Grounding DINO는 (이미지, 텍스트) 쌍을 입력으로 받는 AI 모델입니다. 이 모델은 이미지 내의 객체를 인식하고, 해당 객체에 대한 설명을 생성하는 작업을 수행합니다. 기본적으로 Grounding DINO는 이미지 내의 900개의 객체 상자를 출력합니다. 각 상자는 입력된 모든 단어에 대한 유사성 점수를 가지고 있습니다. 특정 임계값(box_threshold)보다 높은 최대 유사성을 가진 상자만을 선택하는 방식으로 작동합니다. 또한, 특정 임계값(text_threshold) 이상의 유사성을 가진 단어들을 예측 레이블로 추출합니다. 예를 들어, "두 마리의 개가 막대기를 가지고 있다"라는 문장에서 '개'에 해당하는 객체를 얻고 싶다면, '개'와 가장 높은 텍스트 유사성을 가진 상자를 최종 출력으로 선택할 수 있습니다.

* [공식깃헙](https://github.com/IDEA-Research/GroundingDINO)

### 흉기 검출 처리 과정

공식깃헙에서 제공하는 소스코드를 활용하여 아래 코드를 추가한 후 흉기 검출을 처리했습니다.

```python
# 이미지 처리 함수
def process_image(image_path, model, text_prompt):
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=text_prompt, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    # Apply Gaussian blur to the image_source
    blurred_image = cv2.GaussianBlur(image_source, (25, 25), 0)
    
    annotated_frame = annotate(image_source=blurred_image, boxes=boxes, logits=logits, phrases=phrases)

    return annotated_frame, phrases

import os
import cv2
from tqdm import tqdm
import supervision as sv
import tempfile

text_prompt = "A person with a knife in their hand"
input_video_path = '/content/source_video.mp4'
output_video_path = '/content/output_video.mp4'
output_knife_path = '/content/output_knife'
output_no_knife_path = '/content/output'

# Create output directories if they don't exist
os.makedirs(output_knife_path, exist_ok=True)
os.makedirs(output_no_knife_path, exist_ok=True)

# 동영상 파일 읽기
cap = cv2.VideoCapture(input_video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the video width/height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 동영상 파일 만들기
video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# 각 프레임에 대해
for _ in tqdm(range(frame_count)):

    # Read the frame
    ret, frame = cap.read()

    # Save the frame to a temporary file
    temp_filename = tempfile.mktemp(suffix='.jpg')
    cv2.imwrite(temp_filename, frame)

    # Process the image
    processed_frame, phrases = process_image(temp_filename, model, text_prompt)

    # Remove the temporary file
    os.remove(temp_filename)

    # 칼이 포함된 프레임 이미지
    if 'knife' in phrases:
        save_path = os.path.join(output_knife_path, f"frame_{_}.jpg")
    # 칼이 포함되어 있지 않은 이미지
    else:
        save_path = os.path.join(output_no_knife_path, f"frame_{_}.jpg")
    
    # 이미지 저장
    cv2.imwrite(save_path, processed_frame)

    # 동영상에 프레임 추가
    video.write(processed_frame)

# 동영상 파일 닫기
video.release()
cap.release()
```

### 전체 소스 코드

전체 소스 코드는 아래에서 확인할 수 있습니다.

* [전체 소스 코드](https://aifactory.space/forum/discussion/569)
