---
layout: post
title: "Getting Started with Gemma Tokenizer including Multilingual Testing"
author: 김태영
date: 2024-2-23 00:00:00
categories: llm
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-2-27-getting_started_with_gemma_tokenizer_including_multilingual_testing_title_1.png
---
 
![img](http://tykimos.github.io/warehouse/2024/2024-2-27-getting_started_with_gemma_tokenizer_including_multilingual_testing_title_1.png)

### See Also

* Part 1 - Quick Start with Gemma (To be published)
* Part 2 - Getting Started with Gemma Tokenizer including Multilingual Testing
* Part 3 - [Quick Start to Gemma LoRA Fine-Tuning](https://tykimos.github.io/2024/02/22/gemma_lora_fine_tuning_fast_execute/)
* Part 4 - [Quick Start to Gemma Korean LoRA Fine-Tuning](https://tykimos.github.io/2024/02/22/gemma_korean_lora_fine_tuning_fast_execute/)
* Part 5 - [Quick Start to Gemma English-Korean Translation LoRA Fine-Tuning](https://tykimos.github.io/2024/02/22/gemma_en2ko_lora_fine_tuning_fast_execute/)
* Part 6 - [Quick Start to Gemma Korean-English Translation LoRA Fine-Tuning](https://tykimos.github.io/2024/02/22/gemma_ko2en_lora_fine_tuning_fast_execute/)
* Part 7 - [Quick Start to Gemma Korean SQL Chatbot LoRA Fine-Tuning](https://tykimos.github.io/2024/02/23/gemma_ko2sql_lora_fine_tuning_fast_execute/)

Below is an introductory guide for developers on how to get started with the Gemma Tokenizer using the keras-nlp library. It covers installation, configuration, and basic usage examples across multiple languages to demonstrate the versatility of the Gemma Tokenizer.

## Installation

First, ensure that you have the latest versions of the necessary libraries. This includes installing keras-nlp for NLP utilities and updating keras to its latest version to ensure compatibility.


```
# Install the required keras-nlp library
!pip install -q -U keras-nlp
# Update keras to the latest version
!pip install -q -U keras>=3
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m465.2/465.2 kB[0m [31m8.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m950.8/950.8 kB[0m [31m15.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.2/5.2 MB[0m [31m30.6 MB/s[0m eta [36m0:00:00[0m
    [?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow 2.15.0 requires keras<2.16,>=2.15.0, but you have keras 3.0.5 which is incompatible.[0m[31m
    [0m

## Configuration

Before using the Gemma Tokenizer, you need to configure your environment. This involves setting up Kaggle API credentials (if you're planning to use datasets or models hosted on Kaggle) and specifying the backend for Keras. In this guide, we're using JAX for its performance benefits.

### Setting Up Kaggle API Credentials


```
# Import necessary libraries for environment setup
import os
from google.colab import userdata

# Set Kaggle API credentials in the environment variables
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

### Specifying the Backend


```
# Set the KERAS_BACKEND to 'jax'. You can also use 'torch' or 'tensorflow'.
os.environ["KERAS_BACKEND"] = "jax"  # JAX backend is preferred for its performance.

# Avoid memory fragmentation when using the JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
```

## Initialization

Now, let's import the necessary libraries and initialize the Gemma Tokenizer with a preset configuration. This example uses an English model, but Gemma supports multiple languages.


```
# Import keras and keras_nlp for tokenizer functionality
import keras
import keras_nlp

# Initialize the GemmaTokenizer with a preset configuration
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_2b_en")
```

    Attaching 'tokenizer.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
    Attaching 'tokenizer.json' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...
    Attaching 'assets/tokenizer/vocabulary.spm' from model 'keras/gemma/keras/gemma_2b_en/2' to your Colab notebook...


    [  4521 235269  15674 235265]
    tf.Tensor(b'Hello, Korea.', shape=(), dtype=string)


## Basic Usage

Here's how to tokenize and detokenize a sample text. Additionally, we'll define a function to convert token IDs to strings and print tokens with sequential colors for better visualization.

### Tokenizing and Detokenizing Text


```
tokens = tokenizer("Hello, Korea! I'm Gemma.")
print(tokens)

strings = tokenizer.detokenize(tokens)
print(strings)
```

    [  4521 235269  15674 235341    590 235303 235262 137061 235265]
    tf.Tensor(b"Hello, Korea! I'm Gemma.", shape=(), dtype=string)


## Visualizing Tokens with Colors

We define a helper function to apply colors to tokens sequentially and print them. This enhances readability and helps in analyzing the tokenization output visually.


```
# Import itertools for later use in color cycling
import itertools

# Define a function to convert tokens to strings
def tokens2string(tokens):
    # Initialize a list to hold the token strings
    token_string = []

    # Convert each token ID to a string using the tokenizer
    for token_id in tokens:
        # Convert the token ID to a numpy array and get the actual integer ID
        token_id_numpy = jnp.array(token_id).item()
        # Detokenize the single token ID to get the raw string
        raw_string = tokenizer.detokenize([token_id_numpy])
        # Convert the raw string to bytes
        byte_string = raw_string.numpy()
        # Decode the byte string to UTF-8 to get the decoded string
        decoded_string = byte_string.decode('utf-8')
        # Append the decoded string to the token_string list
        token_string.append(decoded_string)

    # Return the list of token strings
    return token_string
```


```
# Color definitions using ANSI 24-bit color codes
colors = {
    "pale_purple": "\033[48;2;202;191;234m",
    "light_green": "\033[48;2;199;236;201m",
    "pale_orange": "\033[48;2;241;218;176m",
    "soft_pink": "\033[48;2;234;177;178m",
    "light_blue": "\033[48;2;176;218;240m",
    "reset": "\033[0m"
}

# Function to apply sequential colors to tokens and print them
def print_colored_tokens(tokens):
    # Convert tokens to strings
    token_strings = tokens2string(tokens)
    # Count the number of tokens
    token_count = len(token_strings)
    # Count the total number of characters across all tokens
    char_count = sum(len(token) for token in token_strings)

    # Print the Gemma Tokenizer info and counts
    print(f"Tokens: {token_count} | Characters: {char_count}")

    # Create a cycle iterator for colors to apply them sequentially to tokens
    color_cycle = itertools.cycle(colors.values())
    for token in token_strings:
        color = next(color_cycle)  # Get the next color from the cycle
        if color == colors['reset']:  # Skip the reset color to avoid resetting prematurely
            color = next(color_cycle)
        # Print each token with the selected color and reset formatting at the end
        print(f"{color}{token}{colors['reset']}", end="")
```

## Testing Across Multiple Languages

To demonstrate the Gemma Tokenizer's versatility, we provide examples in English, Korean, Japanese, Chinese, French, and Spanish. This showcases its ability to handle a wide range of languages effectively.



```
test_sentences = {
    "English": "Hello, Korea! I'm Gemma.",
    "Korean": "안녕, 한국! 나는 젬마입니다.",
    "Japanese": "こんにちは、韓国！私はジェンマです。",
    "Chinese": "你好，韩国！我是吉玛。",
    "French": "Bonjour, Corée ! Je suis Gemma.",
    "Spanish": "¡Hola, Corea! Soy Gemma."
}

print("Gemma Tokenizer")

for language, sentence in test_sentences.items():
    print(f"Language: {language}")
    tokens = tokenizer(sentence)
    print_colored_tokens(tokens)
    print("\n")
```

    Gemma Tokenizer
    Language: English
    Tokens: 9 | Characters: 24
    [48;2;202;191;234mHello[0m[48;2;199;236;201m,[0m[48;2;241;218;176m Korea[0m[48;2;234;177;178m![0m[48;2;176;218;240m I[0m[48;2;202;191;234m'[0m[48;2;199;236;201mm[0m[48;2;241;218;176m Gemma[0m[48;2;234;177;178m.[0m
    
    Language: Korean
    Tokens: 11 | Characters: 17
    [48;2;202;191;234m안[0m[48;2;199;236;201m녕[0m[48;2;241;218;176m,[0m[48;2;234;177;178m 한국[0m[48;2;176;218;240m![0m[48;2;202;191;234m 나는[0m[48;2;199;236;201m [0m[48;2;241;218;176m젬[0m[48;2;234;177;178m마[0m[48;2;176;218;240m입니다[0m[48;2;202;191;234m.[0m
    
    Language: Japanese
    Tokens: 10 | Characters: 18
    [48;2;202;191;234mこんにちは[0m[48;2;199;236;201m、[0m[48;2;241;218;176m韓国[0m[48;2;234;177;178m！[0m[48;2;176;218;240m私は[0m[48;2;202;191;234mジェ[0m[48;2;199;236;201mン[0m[48;2;241;218;176mマ[0m[48;2;234;177;178mです[0m[48;2;176;218;240m。[0m
    
    Language: Chinese
    Tokens: 8 | Characters: 11
    [48;2;202;191;234m你好[0m[48;2;199;236;201m，[0m[48;2;241;218;176m韩国[0m[48;2;234;177;178m！[0m[48;2;176;218;240m我是[0m[48;2;202;191;234m吉[0m[48;2;199;236;201m玛[0m[48;2;241;218;176m。[0m
    
    Language: French
    Tokens: 8 | Characters: 31
    [48;2;202;191;234mBonjour[0m[48;2;199;236;201m,[0m[48;2;241;218;176m Corée[0m[48;2;234;177;178m ![0m[48;2;176;218;240m Je[0m[48;2;202;191;234m suis[0m[48;2;199;236;201m Gemma[0m[48;2;241;218;176m.[0m
    
    Language: Spanish
    Tokens: 8 | Characters: 24
    [48;2;202;191;234m¡[0m[48;2;199;236;201mHola[0m[48;2;241;218;176m,[0m[48;2;234;177;178m Corea[0m[48;2;176;218;240m![0m[48;2;202;191;234m Soy[0m[48;2;199;236;201m Gemma[0m[48;2;241;218;176m.[0m