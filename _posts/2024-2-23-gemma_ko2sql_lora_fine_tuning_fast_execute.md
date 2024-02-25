---
layout: post
title: "Gemma 한국어 SQL챗봇 LoRA 파인튜닝 빠른실행"
author: 김태영
date: 2024-2-23 00:00:00
categories: llm
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-2-23-gemma_ko2sql_lora_fine_tuning_fast_execute_title_1.png
---
 
![img](http://tykimos.github.io/warehouse/2024/2024-2-23-gemma_ko2sql_lora_fine_tuning_fast_execute_title_1.png)

이번에는 Gemma를 이용해서 한국어를 SQL로 변환하여 DB를 쿼리하는 챗봇을 만들어보겠습니다.

### 함께보기

* 1편 - Gemma 시작하기 빠른실행 (추후 공개)
* [2편 - Gemma LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_lora_fine_tuning_fast_execute/)
* [3편 - Gemma 한국어 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_korean_lora_fine_tuning_fast_execute/)
* [4편 - Gemma 영한번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_en2ko_lora_fine_tuning_fast_execute/)
* [5편 - Gemma 한영번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_ko2en_lora_fine_tuning_fast_execute/)
* [6편 - Gemma 한국어 SQL챗봇 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/23/gemma_ko2sql_lora_fine_tuning_fast_execute/)

### Chinook 데이터베이스

Chinook 데이터베이스는 디지털 미디어 스토어를 모델링한 오픈 소스 샘플 데이터베이스입니다. 이 데이터베이스는 아티스트, 앨범, 미디어 트랙, 인보이스 및 고객 등과 관련된 테이블을 포함하고 있으며, 서로 관련된 테이블들이 사전에 데이터로 채워져 있습니다. Chinook 데이터베이스는 SQLite, PostgreSQL, MySQL, Oracle, MS SQL Server를 포함한 다양한 데이터베이스 형식으로 제공되며, 데이터베이스 및 데이터베이스 디자인에 대해 배우거나 새로운 도구를 시도해보고자 할 때 유용하게 사용될 수 있습니다

* [링크](https://github.com/lerocha/chinook-database)

![img](https://private-user-images.githubusercontent.com/135025/299867754-cea7a05a-5c36-40cd-84c7-488307a123f4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDg4MzE1MjksIm5iZiI6MTcwODgzMTIyOSwicGF0aCI6Ii8xMzUwMjUvMjk5ODY3NzU0LWNlYTdhMDVhLTVjMzYtNDBjZC04NGM3LTQ4ODMwN2ExMjNmNC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMjI1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDIyNVQwMzIwMjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04NjcyMTQwMjMwMDJmMWEzMTBjZDgwODIxYzQ5MmM2OWM4Y2E2ZTYxM2Q4OTMxZjRhMWJmZjNjZjMwMjE3N2ZlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.gV60AZa4xmPBr83BAdN6doj3XJwq2TtaZ681GFFWIfo)

### Chinook 데이터베이스 준비

![img](http://tykimos.github.io/warehouse/2024/2024-2-23-gemma_ko2sql_lora_fine_tuning_fast_execute_1.png)

### gemma LoRA 설정

![img](http://tykimos.github.io/warehouse/2024/2024-2-23-gemma_ko2sql_lora_fine_tuning_fast_execute_2.png)

### chinook-ko2sql-1k 데이터셋

chinook DB를 기반으로 한국어 to SQL 쌍으로 ChatGPT를 통해 만들었습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-2-23-gemma_ko2sql_lora_fine_tuning_fast_execute_3.png)

아래 링크에서 다운로드 받을 수 있습니다.

[다운로드](http://tykimos.github.io/warehouse/2024/chinook-ko2sql-1k.jsonl)

#### 수행결과

![img](http://tykimos.github.io/warehouse/2024/2024-2-23-gemma_ko2sql_lora_fine_tuning_fast_execute_4.png)

```
USER > 모든 트랙의 평균 길이는 몇 분인가요?
1.4143006801605225 seconds.
SELECT AVG(Milliseconds) / 60000.0 AS AverageLengthMinutes FROM Track;
Gemma> [(6.559986868398515,)]
USER > 어떤 미디어 타입이 가장 인기가 많나요?
2.131500720977783 seconds.
SELECT MediaType.Name, COUNT(*) AS TotalTracks FROM MediaType JOIN Track ON MediaType.MediaTypeId = Track.MediaTypeId GROUP BY MediaType.MediaTypeId ORDER BY TotalTracks DESC LIMIT 1;
Gemma> [('MPEG audio file', 3034)]
USER > quit
```

### 추가문의

* 작성자 : 김태영
* 이메일 : tykim@aifactory.page

### 더보기

LoRA 파인튜닝 공식 예제는 다음과 같습니다.

* [https://ai.google.dev/gemma/docs/lora_tuning](https://ai.google.dev/gemma/docs/lora_tuning)

* 1편 - Gemma 시작하기 빠른실행 (추후 공개)
* [2편 - Gemma LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_lora_fine_tuning_fast_execute/)
* [3편 - Gemma 한국어 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_korean_lora_fine_tuning_fast_execute/)
* [4편 - Gemma 영한번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_en2ko_lora_fine_tuning_fast_execute/)
* [5편 - Gemma 한영번역 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/22/gemma_ko2en_lora_fine_tuning_fast_execute/)
* [6편 - Gemma 한국어 SQL챗봇 LoRA 파인튜닝 빠른실행](https://tykimos.github.io/2024/02/23/gemma_ko2sql_lora_fine_tuning_fast_execute/)