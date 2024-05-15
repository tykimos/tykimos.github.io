---
layout: post
title: "Image Descriptions with GPT-4o and LCEL"
author: Taeyoung Kim
date: 2024-5-15 11:40:20
categories: LCEL LangChain
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-5-15-image_descriptions_with_gpt_4o_and_lcel_title.jpg
---

![img](http://tykimos.github.io/warehouse/2024/2024-5-15-image_descriptions_with_gpt_4o_and_lcel_title.jpg)
This code demonstrates two methods for working with images in the context of generating and invoking prompts with LangChain and OpenAI's GPT-4o model. The two primary methods are:

1. Using an Image URL: Directly passing the URL of an image to the OpenAI API for processing.
1. Using an Image File: Encoding a locally saved image file to a base64 string and then using it in a prompt for the OpenAI API.

# Using an Image URL


```python
!pip install langchain_openai
```

    Collecting langchain_openai
      Downloading langchain_openai-0.1.6-py3-none-any.whl (34 kB)
    Collecting langchain-core<0.2.0,>=0.1.46 (from langchain_openai)
      Downloading langchain_core-0.1.52-py3-none-any.whl (302 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m302.9/302.9 kB[0m [31m3.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting openai<2.0.0,>=1.24.0 (from langchain_openai)
      Downloading openai-1.30.1-py3-none-any.whl (320 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m320.6/320.6 kB[0m [31m6.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tiktoken<1,>=0.5.2 (from langchain_openai)
      Downloading tiktoken-0.7.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m8.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.10/dist-packages (from langchain-core<0.2.0,>=0.1.46->langchain_openai) (6.0.1)
    Collecting jsonpatch<2.0,>=1.33 (from langchain-core<0.2.0,>=0.1.46->langchain_openai)
      Downloading jsonpatch-1.33-py2.py3-none-any.whl (12 kB)
    Collecting langsmith<0.2.0,>=0.1.0 (from langchain-core<0.2.0,>=0.1.46->langchain_openai)
      Downloading langsmith-0.1.57-py3-none-any.whl (121 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m121.0/121.0 kB[0m [31m7.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting packaging<24.0,>=23.2 (from langchain-core<0.2.0,>=0.1.46->langchain_openai)
      Downloading packaging-23.2-py3-none-any.whl (53 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.0/53.0 kB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pydantic<3,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain-core<0.2.0,>=0.1.46->langchain_openai) (2.7.1)
    Requirement already satisfied: tenacity<9.0.0,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from langchain-core<0.2.0,>=0.1.46->langchain_openai) (8.3.0)
    Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.10/dist-packages (from openai<2.0.0,>=1.24.0->langchain_openai) (3.7.1)
    Requirement already satisfied: distro<2,>=1.7.0 in /usr/lib/python3/dist-packages (from openai<2.0.0,>=1.24.0->langchain_openai) (1.7.0)
    Collecting httpx<1,>=0.23.0 (from openai<2.0.0,>=1.24.0->langchain_openai)
      Downloading httpx-0.27.0-py3-none-any.whl (75 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.6/75.6 kB[0m [31m4.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from openai<2.0.0,>=1.24.0->langchain_openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.10/dist-packages (from openai<2.0.0,>=1.24.0->langchain_openai) (4.66.4)
    Requirement already satisfied: typing-extensions<5,>=4.7 in /usr/local/lib/python3.10/dist-packages (from openai<2.0.0,>=1.24.0->langchain_openai) (4.11.0)
    Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken<1,>=0.5.2->langchain_openai) (2023.12.25)
    Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from tiktoken<1,>=0.5.2->langchain_openai) (2.31.0)
    Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai<2.0.0,>=1.24.0->langchain_openai) (3.7)
    Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai<2.0.0,>=1.24.0->langchain_openai) (1.2.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai<2.0.0,>=1.24.0->langchain_openai) (2024.2.2)
    Collecting httpcore==1.* (from httpx<1,>=0.23.0->openai<2.0.0,>=1.24.0->langchain_openai)
      Downloading httpcore-1.0.5-py3-none-any.whl (77 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m2.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting h11<0.15,>=0.13 (from httpcore==1.*->httpx<1,>=0.23.0->openai<2.0.0,>=1.24.0->langchain_openai)
      Downloading h11-0.14.0-py3-none-any.whl (58 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m5.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting jsonpointer>=1.9 (from jsonpatch<2.0,>=1.33->langchain-core<0.2.0,>=0.1.46->langchain_openai)
      Downloading jsonpointer-2.4-py2.py3-none-any.whl (7.8 kB)
    Collecting orjson<4.0.0,>=3.9.14 (from langsmith<0.2.0,>=0.1.0->langchain-core<0.2.0,>=0.1.46->langchain_openai)
      Downloading orjson-3.10.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m142.5/142.5 kB[0m [31m1.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: annotated-types>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1->langchain-core<0.2.0,>=0.1.46->langchain_openai) (0.6.0)
    Requirement already satisfied: pydantic-core==2.18.2 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1->langchain-core<0.2.0,>=0.1.46->langchain_openai) (2.18.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken<1,>=0.5.2->langchain_openai) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken<1,>=0.5.2->langchain_openai) (2.0.7)
    Installing collected packages: packaging, orjson, jsonpointer, h11, tiktoken, jsonpatch, httpcore, langsmith, httpx, openai, langchain-core, langchain_openai
      Attempting uninstall: packaging
        Found existing installation: packaging 24.0
        Uninstalling packaging-24.0:
          Successfully uninstalled packaging-24.0
    Successfully installed h11-0.14.0 httpcore-1.0.5 httpx-0.27.0 jsonpatch-1.33 jsonpointer-2.4 langchain-core-0.1.52 langchain_openai-0.1.6 langsmith-0.1.57 openai-1.30.1 orjson-3.10.3 packaging-23.2 tiktoken-0.7.0
    Requirement already satisfied: pillow in /usr/local/lib/python3.10/dist-packages (9.4.0)


### Setting Up OpenAI API Key in Google Colab


```python
import os

from google.colab import userdata

# Retrieve the OpenAI API key from user data and set it as an environment variable
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
```

### Initializing the ChatOpenAI Model with GPT-4o


```python
from langchain_openai import ChatOpenAI

# Create an instance of ChatOpenAI with the model set to "gpt-4o"
chat_model = ChatOpenAI(model="gpt-4o")
```

### Generating a Prompt with System and Human Multimodal Messages


```python
from langchain_core.messages import HumanMessage, SystemMessage

# Function to generate a prompt based on given parameters
def gen_prompt(param_dict):

    # Define the system message content
    system_message = "You are a helpful assistant that kindly explains images and answers questions provided by the user."

    # Define the human messages content
    human_messages = [
        {
            "type" : "text",
            "text" : f"{param_dict['question']}",
        },
        {
            "type" : "image_url",
            "image_url" : {
                "url" : f"{param_dict['image_url']}",
            }
        }

    ]

    return [SystemMessage(content=system_message), HumanMessage(content=human_messages)]
```

# Creating and Invoking a Chain with LCEL


```python
from langchain_core.output_parsers import StrOutputParser

# Create a chain by combining the prompt generation, chat model, and output parser
chain = gen_prompt | chat_model | StrOutputParser()

# Invoke the chain with the provided question and image URL
response = chain.invoke({
    "question":"Please describe this person.",
    "image_url": "http://tyritarot.github.io/warehouse/2024/2024-4-7-shining_in_the_cherry_blossoms_and_just_me_title.jpg"
    }
)
```


```python
print(response)
```

    The image depicts an illustrated character of a young woman standing outdoors, surrounded by a picturesque setting with blooming cherry blossom trees. She has long, flowing blonde hair and is wearing a stylish, floral-patterned dress with lace trim. She carries a handbag on her shoulder and wears sunglasses perched on her head. The character has a bright smile, giving off a cheerful and lively vibe. The background is vibrant with pink cherry blossoms and greenery, adding to the overall pleasant and serene atmosphere of the scene.


# Using an Image File


```python
!pip install pillow
```

    Requirement already satisfied: pillow in /usr/local/lib/python3.10/dist-packages (9.4.0)


### Prepare an Image


```python
import requests

# URL of the image
url = "http://tyritarot.github.io/warehouse/2024/2024-4-7-shining_in_the_cherry_blossoms_and_just_me_title.jpg"

# Send a GET request to the URL
response = requests.get(url)

# Save the image content to a file
with open("sample.jpg", "wb") as file:
    file.write(response.content)

print("Image downloaded and saved as sample.jpg")
```

    Image downloaded and saved as sample.jpg


### Encoding an Image to Base64


```python
from PIL import Image
import base64
from io import BytesIO

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "sample.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)
```

### Creating and Invoking a Chat Chain with Base64 Encoded Image in LangChain


```python
response = chain.invoke({"question":"Please describe this person." , "image_url": f"data:image/jpeg;base64,{base64_image}"})
```


```python
print(response)
```

    The image depicts an animated or illustrated character of a young woman. She has long, flowing blonde hair and is smiling warmly. She is wearing a floral dress with a white lace collar and has a pair of sunglasses resting on top of her head. Additionally, she is accessorized with a necklace and dangling earrings. The background shows a scenic outdoor setting with blooming pink cherry blossoms and a group of people in the distance, suggesting a pleasant and serene atmosphere.


