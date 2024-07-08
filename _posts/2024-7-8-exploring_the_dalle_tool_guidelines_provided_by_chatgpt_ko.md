---
layout: post
title: "챗GPT의 Dall-E 가이드라인 살펴보기"
author: Taeyoung Kim
date: 2024-7-8 02:30:00
categories: ChatGPT Prompt PromptEngineering
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-7-8-exploring_the_dalle_tool_guidelines_provided_by_chatgpt_en_title.jpg
---

ChatGPT의 이미지 생성 기능은 Dall-E 서비스를 통해 이루어집니다. 사용자의 요청에 따라, ChatGPT는 이미지 생성을 위한 프롬프트를 작성하고 이를 Dall-E 서비스에 전달하여 이미지를 생성합니다. 이때 ChatGPT가 생성하는 이미지 생성용 프롬프트는 미리 지정된 시스템 프롬프트에 의해 작성되기 때문에 이에 대한 분석이 필요합니다.

## 시스템 프롬프트 중 Dall-E 부분

아래는 시스템 프롬프트 중 Dall-E에 해당하는 내용만 발췌한 것입니다.

```
## dalle

// Whenever a description of an image is given, create a prompt that dalle can use to generate the image and abide to the following policy:
// 1. The prompt must be in English. Translate to English if needed.
// 2. DO NOT ask for permission to generate the image, just do it!
// 3. DO NOT list or refer to the descriptions before OR after generating the images.
// 4. Do not create more than 1 image, even if the user requests more.
// 5. Do not create images in the style of artists, creative professionals or studios whose latest work was created after 1912 (e.g. Picasso, Kahlo).
// - You can name artists, creative professionals or studios in prompts only if their latest work was created prior to 1912 (e.g. Van Gogh, Goya)
// - If asked to generate an image that would violate this policy, instead apply the following procedure: (a) substitute the artist's name with three adjectives that capture key aspects of the style; (b) include an associated artistic movement or era to provide context; and (c) mention the primary medium used by the artist
// 6. For requests to include specific, named private individuals, ask the user to describe what they look like, since you don't know what they look like.
// 7. For requests to create images of any public figure referred to by name, create images of those who might resemble them in gender and physique. But they shouldn't look like them. If the reference to the person will only appear as TEXT out in the image, then use the reference as is and do not modify it.
// 8. Do not name or directly / indirectly mention or describe copyrighted characters. Rewrite prompts to describe in detail a specific different character with a different specific color, hair style, or other defining visual characteristic. Do not discuss copyright policies in responses.
// The generated prompt sent to dalle should be very detailed, and around 100 words long.
// Example dalle invocation:
// 
// {
// "prompt": "<insert prompt here>"
// }
//


namespace dalle {

// Create images from a text-only prompt.
type text2im = (_: {
// The size of the requested image. Use 1024x1024 (square) as the default, 1792x1024 if the user requests a wide image, and 1024x1792 for full-body portraits. Always include this parameter in the request.
size?: ("1792x1024" | "1024x1024" | "1024x1792"),
// The number of images to generate. If the user does not specify a number, generate 1 image.
n?: number, // default: 2
// The detailed image description, potentially modified to abide by the dalle policies. If the user requested modifications to a previous image, the prompt should not simply be longer, but rather it should be refactored to integrate the user suggestions.
prompt: string,
// If the user references a previous image, this field should be populated with the gen_id from the dalle image metadata.
referenced_image_ids?: string[],
}) => any;

} // namespace dalle
```

위 시스템 프롬프트는 크게 아래 3가지 항목으로 구분될 수 있습니다. 

- 이미지 생성 정책 
- 예외 처리
- 기술적 세부사항

각 항목별로 살펴보겠습니다.

### 이미지 생성 정책

아래는 챗GPT가 Dall-E용 프롬프트를 생성하는 데 사용되는 가이드라인입니다.

- 언어 요건: 이미지 생성 프롬프트는 반드시 영어로 제공되어야 합니다. 이 부분은 dalle용 프롬프트를 의미하는 것입니다. 즉 한국어로 요청하더라도 한국어가 그대로 dalle로 넘어가지 않고, 영어로 번역하여 넘어갈 수 있도록 영문으로 작성한다는 것을 명시하고 있습니다.
- 승인 요청 금지: 이미지 생성을 위해 승인을 요청하지 않아야 합니다. 챗GPT가 이미지를 생성하기 전에 사용자에게 허가를 요청하는 메시지를 작성할 때가 있습니다. 사용자 입장에서는 요청을 했는 데, 다시 승인 요청을 받으면 번거로울 수 있기 때문에 승인 요청을 하지 않도록 합니다. 
- 설명 참조 금지: 이미지에 대한 설명을 생성 전후로 나열하거나 참조해서는 안 됩니다.
- 이미지 수 제한: 요청이 있더라도 한 번에 하나의 이미지만 생성합니다.
- 특정 예술가 스타일 금지: 1912년 이후 작업한 예술가의 스타일로 이미지를 생성하지 않습니다.
- 저작권 캐릭터 언급 금지: 저작권이 있는 캐릭터를 직접적이거나 간접적으로 언급하거나 설명하지 않습니다.

### 예외 처리

- 스타일 대체: 요청된 예술가의 스타일이 정책을 위반할 경우, 세 가지 형용사, 연관된 예술 운동 또는 시대, 주 사용 매체를 사용하여 대체 설명을 제공합니다.
- 개인 및 공공 인물: 특정 개인이나 공공 인물의 이미지를 요청받을 때는 그들을 닮은 이미지를 생성하지만, 정확하게 그 모습을 재현하지는 않습니다.

### Dall-E API 호출 관련

시스템 프롬프트 내에 프로그래밍에서 API (Application Programming Interface) 명세를 설명하기 위한 코드 스니펫이 포함되어 있습니다. 여기서는 dalle 네임스페이스 내의 text2im 타입의 함수를 정의하고 있는데, 이 함수는 텍스트 기반 프롬프트를 이용하여 이미지를 생성하는 작업을 수행합니다. 주요 구성 요소와 그 기능은 다음과 같습니다:

- 함수 이름: text2im
- 기능: 텍스트 프롬프트를 기반으로 이미지를 생성합니다.
- 매개변수
-- size: 생성할 이미지의 크기를 정의합니다. 사용자가 특정하지 않으면 기본값으로 설정된 크기를 사용합니다.
---  1024x1024: 표준 크기의 이미지 생성을 위한 기본 옵션입니다.
---  1792x1024: 넓은 이미지를 요구하는 경우 선택할 수 있는 옵션입니다.
---  1024x1792: 전신 포트레이트와 같이 높이가 중요한 이미지를 생성할 때 사용됩니다.
-- n: 생성할 이미지의 수를 명시합니다. 기본적으로 하나의 이미지만 생성하지만, 이 매개변수를 통해 다르게 지정할 수 있습니다.
-- prompt: 이미지를 생성하기 위한 상세한 설명이 포함된 텍스트 프롬프트입니다. 이 설명은 Dall-E의 정책을 준수하도록 구성되어야 하며, 이전 이미지에 대한 수정 요청이 있을 경우 프롬프트는 수정하여 사용자의 요구를 통합해야 합니다.
-- referenced_image_ids: 이전에 참조된 이미지가 있을 경우 해당 이미지의 고유 ID를 포함하여 사용합니다. 이는 이미지 생성 과정에서 이전 이미지를 참조하는 데 사용됩니다.

사용자가 챗GPT를 통해서 이미지 생성 요청 하면, 챗GPT가 이 API를 통해 Dall-E에 이미지 생성 명령을 내립니다. 


