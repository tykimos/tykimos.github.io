---
layout: post
title: "챗GPT와 위스퍼를 활용해 해외 유튜브 영상을 한글로 자동 번역하기"
author: Taeyoung Kim
date: 2023-9-9 00:00:00
categories: bic
comments: true
image: https://tykimos.github.io/warehouse/2023/2023-9-9-korean_scripting_from_youtube_with_chatgpt_and_whisper_title.png
---

![img](https://tykimos.github.io/warehouse/2023/2023-9-9-korean_scripting_from_youtube_with_chatgpt_and_whisper_title.png)

## 전체 흐름
```
{ 입력 비디오 파일 }
      |
      v
[ split_video() ]
    / | \
{ 부분비디오파일1 } { 부분비디오파일2 } ... { 부분비디오파일N }
    \ | /
      v
[ video_files2text_files() ]
    / | \
{ 텍스트파일1 } { 텍스트파일2 } ... { 텍스트파일N }
    \ | /
      v
[ combine_text_files() ]
      |
      v
{ 결합된 텍스트파일 }
      |
      v
[ translate_text_file() ]
      |
      v
{ 번역된 텍스트파일 }
```
본 코드는 주어진 비디오 파일을 분석하여 한국어로 번역된 텍스트 파일을 생성하는 과정을 나타냅니다. 

1. `split_video()` 함수는 입력된 비디오 파일을 여러 부분으로 나눕니다. 이때 각 부분의 크기는 10MB를 넘지 않도록 설정되어 있습니다.
2. 나눈 비디오 파일들을 `video_files2text_files()` 함수를 통해 텍스트 파일로 변환합니다. 이때 각 비디오 파일은 독립적으로 텍스트 파일로 변환됩니다.
3. 변환된 모든 텍스트 파일들을 `combine_text_files()` 함수를 통해 하나의 텍스트 파일로 결합합니다. 이때 중복되는 부분은 제거되어 결합됩니다.
4. 결합된 텍스트 파일은 `translate_text_file()` 함수를 통해 한국어로 번역됩니다. 이때 크기가 큰 텍스트 파일은 문장 단위로 분할하여 번역되며, 이때 모델로는 "gpt-4"를 사용하였습니다.
5. 마지막으로 번역된 텍스트 파일이 출력됩니다.

## 함수 목록
- time_it
- time_to_seconds
- get_video_info
- translate_to_korean
- split_video
- video_files2text_files
- combine_text_files
- read_large_file_in_chunks
- translate_text_file
- main
- wrapper

## 함수 설명
### time_it

'time_it' 함수는 파이썬에서 데코레이터를 사용하여 어떤 함수가 실행되는데 걸리는 시간을 측정하는 기능을 합니다. 'time_it' 함수는 하나의 함수(func)를 인자로 받아서 그 함수를 실행시키는 새로운 함수(wrapper)를 반환합니다. 이때 'wrapper' 함수는 가변 인자(*args)와 키워드 인자(**kwargs)를 받아서 'func' 함수를 실행시키며, 이 함수의 실행 시작 시간과 종료 시간을 datetime 모듈을 통해 기록합니다. 함수가 실행된 후에는 시작 시간과 종료 시간의 차이를 계산하여 총 실행 시간(elapsed_time)을 구하고, 이를 출력합니다. 그리고 'func' 함수의 결과를 반환합니다.

```python
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        print(f"{func.__name__} took {elapsed_time} seconds to run.")
        return result
    return wrapper
```

### time_to_seconds

' time_to_seconds ' 함수는 주어진 시간을 초로 변환하는 파이썬 코드입니다. 이 함수는 매개변수로 'timestr'을 받습니다. 'timestr'는 시간, 분, 초를 나타내는 문자열로서, 이들 세 요소는 콜론 (':')으로 구분되어 있습니다. 함수는 먼저 'timestr'을 출력하고, 그 다음으로 이 문자열을 콜론을 기준으로 분리합니다. 분리된 각 요소는 float 타입으로 변환되어 각각 'hours', 'minutes', 'seconds'에 저장됩니다. 이 함수는 마지막으로 계산된 시간, 분, 초를 초 단위로 환산하여 반환합니다. 이는 각각의 시간, 분, 초를 해당하는 초 수로 변환한 후 이들을 모두 더하는 방식으로 이루어집니다. 즉, 'hours * 3600 + minutes * 60 + seconds'의 결과값을 반환합니다.

```python
def time_to_seconds(timestr):
    print(timestr)
    hours, minutes, seconds = map(float, timestr.split(':'))
    return hours * 3600 + minutes * 60 + seconds
```

### get_video_info

'get_video_info' 함수는 파이썬 코드에서 주어진 비디오 파일의 정보를 가져오는 기능을 합니다. 이 함수는 파일 이름을 매개변수로 받아서, ffmpeg를 사용하여 비디오 파일의 정보를 가져옵니다. 먼저, subprocess.run을 사용하여 ffmpeg 명령을 실행하고, 그 결과를 가져옵니다. 그 다음, 'Duration'이라는 단어가 포함된 라인을 찾아서 그 라인에서 비디오의 길이와 비트레이트를 추출합니다. 이 때, 정규식을 사용하여 해당 정보를 추출합니다. 만약 길이와 비트레이트 정보를 찾을 수 없으면 None을 반환합니다. 마지막으로, os.path.getsize를 사용하여 파일의 크기를 바이트 단위로 가져옵니다. 이 함수는 비디오의 길이, 비트레이트, 파일 크기를 반환합니다.

```python
def get_video_info(filename):
    cmd = ["ffmpeg", "-i", filename]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    # "Duration" 라인에서 정보 추출
    duration_line = [line for line in result.stderr.split('\n') if "Duration" in line][0]
    
    # 정규식을 사용하여 duration과 bitrate 추출
    duration_match = re.search(r"Duration: (\d{2}:\d{2}:\d{2}.\d{2})", duration_line)
    bitrate_match = re.search(r"bitrate: (\d+ kb/s)", duration_line)
    
    duration = duration_match.group(1) if duration_match else None
    bitrate = bitrate_match.group(1) if bitrate_match else None

    # 파일 용량 (바이트 단위) 얻기
    file_size = os.path.getsize(filename)
    
    return duration, bitrate, file_size
```

### translate_to_korean

'translate_to_korean' 함수는 주어진 텍스트를 한국어로 번역하는 역할을 합니다. 입력으로는 'temperature', 'system_prompt_translation', 'text' 세 가지를 받아요. 'temperature'는 모델의 출력에 대한 무작위성을 결정하는 요소이고, 'system_prompt_translation'은 시스템 메시지의 번역을 위한 프롬프트입니다. 'text'는 사용자가 입력한 텍스트입니다. 함수는 OpenAI의 ChatCompletion를 이용하여 번역을 수행합니다. 이때, 모델명으로 "gpt-4"를 사용하며, 'temperature'와 'messages'를 파라미터로 전달해요. 'messages'는 'system' 역할의 'system_prompt_translation'과 'user' 역할의 'text'를 포함하는 리스트입니다. 함수는 번역된 메시지를 반환합니다. 이 메시지는 응답에서 'choices'의 첫 번째 항목의 'message'의 'content'입니다.

```python
def translate_to_korean(temperature, system_prompt_translation, text):
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt_translation
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return response['choices'][0]['message']['content']
```

### split_video

'video_file_path'와 'overlap_duration'을 매개변수로 받는 'split_video' 함수는 주어진 비디오 파일을 10MB 크기의 여러 부분으로 분할하는 기능을 수행합니다. 먼저, 'get_video_info' 함수를 통해 비디오 파일의 전체 길이, 비트레이트, 파일 크기 정보를 얻습니다. 이후 파일 크기를 기준으로 파일이 몇 개의 10MB 크기의 조각으로 나눠질지 결정합니다. 그 후 파일의 전체 시간을 초 단위로 변환하고, 파일을 분할할 때 각 분할 부분의 길이를 계산합니다. 이후 for문을 통해 각 분할 부분을 생성합니다. 이 때, 각 분할 부분의 시작 시간은 이전 분할 부분의 끝나는 시간에서 중첩되는 시간을 뺀 값으로, ffmpeg 명령을 통해 분할 부분을 생성하고, 생성된 분할 부분의 파일 이름을 parts 리스트에 추가합니다. 함수는 생성된 모든 분할 부분의 파일 이름을 담은 리스트를 반환합니다.

      중첩을 시키는 이유는 짤리는 구간에서 위스퍼가 음식인식할 때 오류가 발생할 수 있기 때문에 이를 고려하여 중첩시켜서 비디오 파일을 자릅니다. 

```python
def split_video(video_file_path, overlap_duration):
    
    duration, bitrate, file_size = get_video_info(video_file_path)

    num_splits = int(file_size / (10 * 1024 * 1024) + 1)
    print(num_splits)
    duration = time_to_seconds(duration)

    """FFmpeg를 사용하여 MP4 파일을 10MB 조각으로 나누기."""
    parts = []
    
    file_size = os.path.getsize(video_file_path)
    
    segment_duration = duration / num_splits

    for i in range(num_splits):
        start_time = max(i * segment_duration - overlap_duration, 0)  # 10초 중첩 고려
        output_file = f"video_part_{i+1}.mp4"
        cmd = ["ffmpeg", "-ss", str(start_time), "-t", str(segment_duration + overlap_duration), "-i", video_file_path, "-c", "copy", "-y", output_file]
        subprocess.run(cmd)
        parts.append(output_file)

    return parts
```

### video_files2text_files

'video_files2text_files' 함수는 비디오 파일들을 텍스트 파일로 변환하는 기능을 수행합니다. 이 함수는 비디오 파일들을 인자로 받아, 각 비디오 파일을 순회하며 처리합니다. 각 파일은 'rb' 모드로 열어서 openai.Audio.transcribe 메서드를 통해 텍스트로 변환됩니다. 이 변환된 텍스트는 "video_part_script_{i}.txt" 형식의 이름을 가진 새로운 텍스트 파일에 저장됩니다. 그리고 이 파일 이름은 parts라는 리스트에 추가됩니다. 이 과정이 모든 비디오 파일에 대해 반복됩니다. 그리고 각 파일이 처리되는 동안, 현재 처리 중인 파일 번호와 파일명이 출력됩니다. 마지막으로, 모든 비디오 파일이 처리된 후에는 parts 리스트가 반환됩니다. 이 리스트에는 처리된 모든 텍스트 파일의 이름들이 저장되어 있습니다.

```python
def video_files2text_files(video_files):
    
    parts = []
    for i, part_file_name in enumerate(video_files):
        
        with open(part_file_name, 'rb') as part_file:
            transcript = openai.Audio.transcribe("whisper-1", part_file)   
            
            output_file = f"video_part_script_{i}.txt"
            
            with open(output_file, "wt") as outfile:
                outfile.write(transcript['text'])
                parts.append(output_file)

        # 현재 진행 상황 출력
        print(f"Processing file {i} of {len(video_files)}: {part_file_name}")
    return parts
```

### combine_text_files

'combine_text_files' 함수는 여러 텍스트 파일들을 하나로 합치는 함수입니다. 이 함수는 텍스트 파일들의 리스트를 입력으로 받아, 각 파일을 열고 내용을 읽어옵니다. 첫 번째 파일의 경우, 마지막 5단어를 제외하고 모든 내용을 합친 내용에 추가합니다. 이후의 파일들에 대해서는 각각의 첫 5단어를 제거하고, 수정된 내용의 첫 10단어를 추출합니다. 이 10단어의 위치를 합쳐진 내용에서 역순으로 찾아 중복되는 부분을 현재 내용에서 제거한 후, 수정된 내용을 합쳐진 내용에 추가합니다. 모든 파일의 내용을 합친 후, 'combined_text.txt'라는 이름의 파일에 합친 내용을 쓰고, 이 파일의 이름을 반환합니다. 이 함수는 텍스트 파일들을 한 파일로 효과적으로 합치는데 사용될 수 있습니다.

```python
def combine_text_files(text_files):
    
    combined_content = ""
    
    for i, file_path in enumerate(text_files):
        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read()
            
        # 첫 번째 파일의 경우 마지막 5단어를 제외하고 내용을 추가
        if i == 0:
            combined_content += " ".join(content.split()[:-5])
            continue
            
        # 이후 파일들의 경우 첫 5단어 제거
        content = " ".join(content.split()[5:])
        
        # 수정된 내용의 첫 10단어 추출
        first_10_words = " ".join(content.split()[:10])
        
        # 합쳐진 내용에서 이 10단어의 위치를 역순으로 찾기
        overlap_index = combined_content.rfind(first_10_words)
        
        # 중복되는 부분을 현재 내용에서 제거
        if overlap_index != -1:
            content = content[len(first_10_words):].lstrip()
        
        # 수정된 내용을 합쳐진 내용에 추가
        combined_content += content
        
    output_filename = "combined_text.txt"
    
    with open(output_filename, "w", encoding='utf-8') as output_file:
        output_file.write(combined_content)

    return output_filename
```

### read_large_file_in_chunks

'read_large_file_in_chunks' 함수는 파이썬 코드에서 큰 파일을 조각으로 읽는 역할을 합니다. 이 함수는 파일 경로와 청크 크기를 매개변수로 받습니다. 청크 크기의 기본값은 16입니다. 

함수는 먼저 파일을 'rt'(텍스트 읽기 모드)로 열고, utf-8 인코딩을 사용합니다. 그 후, 각 라인을 읽어와서 문장들이 저장되는 리스트에 추가합니다. 라인은 '.', '!', '?'를 기준으로 나누어지며, strip() 함수를 통해 라인 앞뒤의 공백이 제거됩니다.

리스트의 길이가 청크 크기 이상이 될 때까지 문장들을 계속 추가합니다. 리스트의 크기가 청크 크기와 같거나 크게 되면, 청크 크기만큼의 문장들을 합쳐 하나의 문자열로 만들고, 이를 yield로 반환합니다. 이후 청크 크기만큼의 문장들은 리스트에서 제거됩니다.

마지막으로, 청크 크기보다 작은 크기의 문장들이 리스트에 남아있다면, 이들을 모두 합쳐 하나의 문자열로 만들어 yield로 반환합니다. 이 함수는 큰 파일을 청크 크기만큼 나누어 처리하므로, 큰 파일을 효율적으로 처리할 수 있습니다.

```python
def read_large_file_in_chunks(file_path, chunk_size=16):
    
    with open(file_path, "rt", encoding="utf-8") as f:
        sentences = []
        for line in f:
            sentences.extend(re.split(r'[.!?]', line.strip()))
            while len(sentences) >= chunk_size:
                yield " ".join(sentences[:chunk_size])
                sentences = sentences[chunk_size:]
        if sentences:   # 남은 문장들 (chunk_size보다 작은 경우)
            yield " ".join(sentences)
```

### translate_text_file

'`translate_text_file` 함수는 주어진 텍스트 파일을 한국어로 번역하는 기능을 수행합니다. 이 함수는 'combined_text_file', 'temperature', 'system_prompt_translation' 세 개의 인자를 받습니다. 'combined_text_file'는 번역할 텍스트 파일, 'temperature'와 'system_prompt_translation'은 한국어 번역 함수 'translate_to_korean'의 인자입니다. 

처음에는 결과를 저장할 파일 이름인 'result_korean.txt'를 선언하고, 번역된 텍스트 조각들을 저장할 빈 리스트를 생성합니다. 그리고 파일을 작은 조각들로 나누어 읽기 시작합니다. 각 조각은 'translate_to_korean' 함수를 통해 한국어로 번역되며, 번역된 조각은 'result_korean.txt' 파일에 저장됩니다. 이 과정이 반복되면서 모든 텍스트 조각이 한국어로 번역되고 파일에 저장됩니다. 또한, 각 번역된 텍스트 조각은 리스트에 추가되어, 마지막에는 이 리스트의 모든 항목이 'result_korean.txt' 파일에 쓰여집니다. 이렇게 번역된 파일의 이름이 반환됩니다.

```python
def translate_text_file(combined_text_file, temperature, system_prompt_translation):
    output_filename = "result_korean.txt"
    
    translated_text_parts = []
    
    i = 0
    
    for chunk in read_large_file_in_chunks(combined_text_file):
        
        translated_chunk = translate_to_korean(temperature, system_prompt_translation, chunk)
        
        with open(f'{output_filename}_{i}.txt', "wt", encoding="utf-8") as outfile:
            outfile.write(translated_chunk)

        translated_text_parts.append(translated_chunk)
        
        i += 1

    # 현재 진행 상황 출력
    with open(output_filename, "wt", encoding="utf-8") as outfile:
        outfile.write("\n".join(translated_text_parts))

    return output_filename
```

### main

'main' 함수는 프로그램의 주 실행 루틴을 정의합니다. 먼저, 프로그램이 실행될 때 전달된 인자의 수를 확인합니다. 인자의 수가 2개가 아니라면 사용 방법을 출력하고 프로그램을 종료합니다. 인자가 적절하게 주어졌다면, 첫 번째 단계로 비디오를 분할하는 과정을 진행합니다. 이 과정에서는 입력으로 주어진 비디오 파일 경로를 이용해 10개의 비디오 파일로 분할하며, 분할된 비디오 파일들의 목록을 출력합니다. 두 번째 단계에서는 분할된 비디오 파일들을 텍스트 파일로 변환하고, 변환된 텍스트 파일들의 목록을 출력합니다. 세 번째 단계에서는 모든 텍스트 파일들을 하나로 결합하고, 결합된 텍스트를 출력합니다. 네 번째 단계에서는 결합된 텍스트를 번역하는 과정을 진행하며, 최종적으로 번역된 텍스트 파일의 이름을 출력합니다. 이렇게 각 단계별로 진행되며 최종적으로 번역된 텍스트 파일을 생성하는 것이 'main' 함수의 기능입니다.

```python
def main():
    if len(sys.argv) != 2:
        print("usage: python eng_video2kor_txt.py [video file path]")
        sys.exit(1)

    print("1. split_video ...")
    video_files = split_video(sys.argv[1], 10)
    print(video_files)
    
    print("2. video to text ...")
    text_files = video_files2text_files(video_files)
    print(text_files)
    
    print("3. combine texts ...")
    combined_text = combine_text_files(text_files)
    print(combined_text)
    
    print("4. translate texts ...")
    translated_text = translate_text_file(combined_text, 0.7, system_prompt_translation)

    print(f"Done. output file : {translated_text}")
```

### wrapper

'wrapper' 함수는 Python에서 정의된 함수로, 여러 가지 인자(*args, **kwargs)를 받아 처리하는 역할을 합니다. 이 함수는 주로 다른 함수의 실행 시간을 측정하는 데 사용됩니다. 함수는 다음과 같은 순서로 동작합니다.

1. 먼저 함수가 호출되면 현재 시간을 'start_time'에 저장합니다.
2. 다음으로, 주어진 인자들을 사용하여 'func' 함수를 실행하고 그 결과를 'result'에 저장합니다.
3. 'func' 함수의 실행이 끝나면 다시 현재 시간을 'end_time'에 저장합니다.
4. 'end_time'과 'start_time'의 차이를 계산하여 'elapsed_time'에 저장합니다. 이 'elapsed_time'은 'func' 함수의 실행 시간을 나타냅니다.
5. 마지막으로, 'func' 함수의 이름과 그 실행 시간을 출력하고, 'func' 함수의 결과를 반환합니다.

따라서 이 'wrapper' 함수는 주어진 'func' 함수의 실행 시간을 측정하고 출력하는 역할을 합니다.

```python
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        print(f"{func.__name__} took {elapsed_time} seconds to run.")
        return result
```


## 전체 소스코드
```python
import os
import subprocess
import openai
import sys
import re
import datetime

# 주어진 작업의 실행 시간을 측정하고 출력하는 데코레이터 함수
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        print(f"{func.__name__} took {elapsed_time} seconds to run.")
        return result
    return wrapper

# 시간 문자열을 초 단위로 변환하는 함수
def time_to_seconds(timestr):
    print(timestr)
    hours, minutes, seconds = map(float, timestr.split(':'))
    return hours * 3600 + minutes * 60 + seconds

# 주어진 비디오 파일의 정보(길이, 비트레이트, 파일 크기)를 가져오는 함수
def get_video_info(filename):
    cmd = ["ffmpeg", "-i", filename]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    # "Duration" 라인에서 정보 추출
    duration_line = [line for line in result.stderr.split('\n') if "Duration" in line][0]
    
    # 정규식을 사용하여 duration과 bitrate 추출
    duration_match = re.search(r"Duration: (\d{2}:\d{2}:\d{2}.\d{2})", duration_line)
    bitrate_match = re.search(r"bitrate: (\d+ kb/s)", duration_line)
    
    duration = duration_match.group(1) if duration_match else None
    bitrate = bitrate_match.group(1) if bitrate_match else None

    # 파일 용량 (바이트 단위) 얻기
    file_size = os.path.getsize(filename)
    
    return duration, bitrate, file_size

# 시스템 메시지로 번역 작업을 요청할 때 사용하는 문장

system_prompt_translation = "You are a helpful assistant. Your task is to translate the following English text into Korean. Use accurate and fluent Korean. Do not add or remove any information from the text, and use only the context provided."

# 주어진 영문 텍스트를 한글로 번역하는 함수
def translate_to_korean(temperature, system_prompt_translation, text):
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt_translation
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return response['choices'][0]['message']['content']
 
# 주어진 비디오를 10MB 크기의 조각으로 분할하는 함수
@time_it
def split_video(video_file_path, overlap_duration):
    
    duration, bitrate, file_size = get_video_info(video_file_path)

    num_splits = int(file_size / (10 * 1024 * 1024) + 1)
    print(num_splits)
    duration = time_to_seconds(duration)

    """FFmpeg를 사용하여 MP4 파일을 10MB 조각으로 나누기."""
    parts = []
    
    file_size = os.path.getsize(video_file_path)
    
    segment_duration = duration / num_splits

    for i in range(num_splits):
        start_time = max(i * segment_duration - overlap_duration, 0)  # 10초 중첩 고려
        output_file = f"video_part_{i+1}.mp4"
        cmd = ["ffmpeg", "-ss", str(start_time), "-t", str(segment_duration + overlap_duration), "-i", video_file_path, "-c", "copy", "-y", output_file]
        subprocess.run(cmd)
        parts.append(output_file)

    return parts

# 비디오 파일들을 텍스트 파일로 변환하는 함수
@time_it
def video_files2text_files(video_files):
    
    parts = []
    for i, part_file_name in enumerate(video_files):
        
        with open(part_file_name, 'rb') as part_file:
            transcript = openai.Audio.transcribe("whisper-1", part_file)   
            
            output_file = f"video_part_script_{i}.txt"
            
            with open(output_file, "wt") as outfile:
                outfile.write(transcript['text'])
                parts.append(output_file)

        # 현재 진행 상황 출력
        print(f"Processing file {i} of {len(video_files)}: {part_file_name}")
    return parts
        
# 여러 텍스트 파일을 하나의 텍스트 파일로 합치는 함수
@time_it
def combine_text_files(text_files):
    
    combined_content = ""
    
    for i, file_path in enumerate(text_files):
        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read()
            
        # 첫 번째 파일의 경우 마지막 5단어를 제외하고 내용을 추가
        if i == 0:
            combined_content += " ".join(content.split()[:-5])
            continue
            
        # 이후 파일들의 경우 첫 5단어 제거
        content = " ".join(content.split()[5:])
        
        # 수정된 내용의 첫 10단어 추출
        first_10_words = " ".join(content.split()[:10])
        
        # 합쳐진 내용에서 이 10단어의 위치를 역순으로 찾기
        overlap_index = combined_content.rfind(first_10_words)
        
        # 중복되는 부분을 현재 내용에서 제거
        if overlap_index != -1:
            content = content[len(first_10_words):].lstrip()
        
        # 수정된 내용을 합쳐진 내용에 추가
        combined_content += content
        
    output_filename = "combined_text.txt"
    
    with open(output_filename, "w", encoding='utf-8') as output_file:
        output_file.write(combined_content)

    return output_filename

# 큰 텍스트 파일을 문장 단위로 분할하여 읽는 함수
def read_large_file_in_chunks(file_path, chunk_size=16):
    
    with open(file_path, "rt", encoding="utf-8") as f:
        sentences = []
        for line in f:
            sentences.extend(re.split(r'[.!?]', line.strip()))
            while len(sentences) >= chunk_size:
                yield " ".join(sentences[:chunk_size])
                sentences = sentences[chunk_size:]
        if sentences:   # 남은 문장들 (chunk_size보다 작은 경우)
            yield " ".join(sentences)

# 텍스트 파일을 한국어로 번역하는 함수
@time_it
def translate_text_file(combined_text_file, temperature, system_prompt_translation):
    output_filename = "result_korean.txt"
    
    translated_text_parts = []
    
    i = 0
    
    for chunk in read_large_file_in_chunks(combined_text_file):
        
        translated_chunk = translate_to_korean(temperature, system_prompt_translation, chunk)
        
        with open(f'{output_filename}_{i}.txt', "wt", encoding="utf-8") as outfile:
            outfile.write(translated_chunk)

        translated_text_parts.append(translated_chunk)
        
        i += 1

    # 현재 진행 상황 출력
    with open(output_filename, "wt", encoding="utf-8") as outfile:
        outfile.write("\n".join(translated_text_parts))

    return output_filename

def main():
    if len(sys.argv) != 2:
        print("usage: python eng_video2kor_txt.py [video file path]")
        sys.exit(1)

    print("1. split_video ...")
    video_files = split_video(sys.argv[1], 10)
    print(video_files)
    
    print("2. video to text ...")
    text_files = video_files2text_files(video_files)
    print(text_files)
    
    print("3. combine texts ...")
    combined_text = combine_text_files(text_files)
    print(combined_text)
    
    print("4. translate texts ...")
    translated_text = translate_text_file(combined_text, 0.7, system_prompt_translation)

    print(f"Done. output file : {translated_text}")

if __name__ == "__main__":
    main()

```

## 실행 결과 출력

실행 결과를 함께 첨부드립니다. 에러가 나거나 원하시는 결과가 나오지 않을 때 참고하세요.

```
(.venv) (base) tykimos@gimtaeyeong-ui-MacBookAir todayassi % python eng_video2kor_txt.py k
erascore.mp4
1. split_video ...
14
00:36:46.64
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_1.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:00.03 bitrate=  -0.0kbits/s speed=Nframe= 5030 fps=0.0 q=-1.0 Lsize=    8555kB time=00:02:47.60 bitrate= 418.2kbits/s speed=2.14e+03x    
video:5777kB audio:2619kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 1.898184%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_2.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:00.71 bitrate=  -0.0kbits/s speed=Nframe= 5051 fps=0.0 q=-1.0 Lsize=    7091kB time=00:02:47.61 bitrate= 346.6kbits/s speed=2.31e+03x    
video:4300kB audio:2631kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 2.309364%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_3.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:01.28 bitrate=  -0.0kbits/s speed=Nframe= 5067 fps=0.0 q=-1.0 Lsize=    7569kB time=00:02:47.61 bitrate= 369.9kbits/s speed=2.58e+03x    
video:4769kB audio:2639kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 2.170455%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_4.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:01.81 bitrate=  -0.0kbits/s speed=Nframe= 5084 fps=0.0 q=-1.0 Lsize=    6944kB time=00:02:47.61 bitrate= 339.4kbits/s speed=2.73e+03x    
video:4135kB audio:2648kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 2.377090%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_5.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:02.36 bitrate=  -0.0kbits/s speed=Nframe= 5100 fps=0.0 q=-1.0 Lsize=    6722kB time=00:02:47.61 bitrate= 328.5kbits/s speed=2.75e+03x    
video:3904kB audio:2657kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 2.468125%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_6.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:02.91 bitrate=  -0.0kbits/s speed=Nframe= 5117 fps=0.0 q=-1.0 Lsize=    6948kB time=00:02:47.61 bitrate= 339.6kbits/s speed=2.6e+03x    
video:4121kB audio:2665kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 2.390399%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_7.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:03.46 bitrate=  -0.0kbits/s speed=Nframe= 5133 fps=0.0 q=-1.0 Lsize=    7198kB time=00:02:47.61 bitrate= 351.8kbits/s speed=1.89e+03x    
video:4361kB audio:2674kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 2.318366%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_8.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:04.01 bitrate=  -0.0kbits/s speed=Nframe= 5150 fps=0.0 q=-1.0 Lsize=    9026kB time=00:02:47.61 bitrate= 441.1kbits/s speed=1.63e+03x    
video:6183kB audio:2682kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 1.818308%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_9.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:04.57 bitrate=  -0.0kbits/s speed=Nframe= 5166 fps=0.0 q=-1.0 Lsize=    8019kB time=00:02:47.61 bitrate= 391.9kbits/s speed=2.54e+03x    
video:5166kB audio:2691kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 2.066358%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_10.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:00.05 bitrate=  -0.0kbits/s speed=Nframe= 5031 fps=0.0 q=-1.0 Lsize=   10221kB time=00:02:47.61 bitrate= 499.6kbits/s speed=2.61e+03x    
video:7443kB audio:2620kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 1.567671%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_11.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:00.60 bitrate=  -0.0kbits/s speed=Nframe= 5047 fps=0.0 q=-1.0 Lsize=   18247kB time=00:02:47.61 bitrate= 891.8kbits/s speed=2.14e+03x    
video:15459kB audio:2629kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.880158%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_12.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:01.15 bitrate=  -0.0kbits/s speed=Nframe= 5064 fps=0.0 q=-1.0 Lsize=   18055kB time=00:02:47.61 bitrate= 882.4kbits/s speed=2.3e+03x    
video:15258kB audio:2638kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.890635%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_13.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:01.70 bitrate=  -0.0kbits/s speed=Nframe= 5080 fps=0.0 q=-1.0 Lsize=   17349kB time=00:02:47.61 bitrate= 847.9kbits/s speed=2.12e+03x    
video:14543kB audio:2646kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.933106%
ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers
  built with Apple clang version 14.0.3 (clang-1403.0.22.14.1)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/6.0 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      58.  2.100 / 58.  2.100
  libavcodec     60.  3.100 / 60.  3.100
  libavformat    60.  3.100 / 60.  3.100
  libavdevice    60.  1.100 / 60.  1.100
  libavfilter     9.  3.100 /  9.  3.100
  libswscale      7.  1.100 /  7.  1.100
  libswresample   4. 10.100 /  4. 10.100
  libpostproc    57.  1.100 / 57.  1.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'kerascore.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2023-07-20T16:17:05.000000Z
  Duration: 00:36:46.64, start: 0.000000, bitrate: 510 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Output #0, mp4, to 'video_part_14.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    encoder         : Lavf60.3.100
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 379 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 127 kb/s (default)
    Metadata:
      creation_time   : 2023-07-20T16:17:05.000000Z
      handler_name    : ISO Media file produced by Google Inc. Created on: 07/20/2023.
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
frame=    0 fps=0.0 q=-1.0 size=       0kB time=-00:00:02.25 bitrate=  -0.0kbits/s speed=Nframe= 5095 fps=0.0 q=-1.0 Lsize=   17226kB time=00:02:47.59 bitrate= 842.0kbits/s speed=1.89e+03x    
video:14410kB audio:2654kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.945195%
split_video took 2.342886 seconds to run.
['video_part_1.mp4', 'video_part_2.mp4', 'video_part_3.mp4', 'video_part_4.mp4', 'video_part_5.mp4', 'video_part_6.mp4', 'video_part_7.mp4', 'video_part_8.mp4', 'video_part_9.mp4', 'video_part_10.mp4', 'video_part_11.mp4', 'video_part_12.mp4', 'video_part_13.mp4', 'video_part_14.mp4']
2. video to text ...
Processing file 0 of 14: video_part_1.mp4
Processing file 1 of 14: video_part_2.mp4
Processing file 2 of 14: video_part_3.mp4
Processing file 3 of 14: video_part_4.mp4
Processing file 4 of 14: video_part_5.mp4
Processing file 5 of 14: video_part_6.mp4
Processing file 6 of 14: video_part_7.mp4
Processing file 7 of 14: video_part_8.mp4
Processing file 8 of 14: video_part_9.mp4
Processing file 9 of 14: video_part_10.mp4
Processing file 10 of 14: video_part_11.mp4
Processing file 11 of 14: video_part_12.mp4
Processing file 12 of 14: video_part_13.mp4
Processing file 13 of 14: video_part_14.mp4
video_files2text_files took 292.291186 seconds to run.
['video_part_script_0.txt', 'video_part_script_1.txt', 'video_part_script_2.txt', 'video_part_script_3.txt', 'video_part_script_4.txt', 'video_part_script_5.txt', 'video_part_script_6.txt', 'video_part_script_7.txt', 'video_part_script_8.txt', 'video_part_script_9.txt', 'video_part_script_10.txt', 'video_part_script_11.txt', 'video_part_script_12.txt', 'video_part_script_13.txt']
3. combine texts ...
combine_text_files took 0.010328 seconds to run.
combined_text.txt
4. translate texts ...
translate_text_file took 1106.092681 seconds to run.
Done. output file : result_korean.txt
(.venv) (base) tykimos@gimtaeyeong-ui-MacBookAir todayassi % 
```
