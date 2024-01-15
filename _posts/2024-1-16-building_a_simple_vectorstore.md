---
layout: post
title: "Building a Simple Vectorstore"
author: Taeyoung Kim
date: 2024-1-16 00:00:00
categories: bic
comments: true
image: http://tykimos.github.io/warehouse/2024-1-16-building_a_simple_vectorstore_title.jpeg
---

## 전체 흐름
The ASCII visualization for the given source code would look like:

```
{ raw_documents } 
      |
      v
[ SimpleTextLoader().load() ] 
      |
      v
{ raw_documents }
      |
      v
[ SimpleCharacterTextSplitter().split_documents() ] 
      |
      v
{ documents } 
      |
      v
[ SimpleVectorStore() ] 
      |
      v
{ db } 
      |
      v
[ SimpleVectorStore().similarity_search() / SimpleRetriever().get_relevant_documents() / SimpleRetrievalQA().invoke() ] 
      |
      v
{ docs / unique_docs / answer }
```

In the above visualization:

- `{ raw_documents }` is the input document file.
- `[ SimpleTextLoader().load() ]` is a function that loads the raw documents.
- `{ raw_documents }` is the loaded document file.
- `[ SimpleCharacterTextSplitter().split_documents() ]` is a function that splits the documents into chunks.
- `{ documents }` are the chunks of documents.
- `[ SimpleVectorStore() ]` is a function that creates a vector representation of the documents.
- `{ db }` is the vector representation of the documents.
- `[ SimpleVectorStore().similarity_search() / SimpleRetriever().get_relevant_documents() / SimpleRetrievalQA().invoke() ]` are functions that return the most relevant documents based on a query.
- `{ docs / unique_docs / answer }` are the outputs from the similarity_search, get_relevant_documents, and invoke functions.

해당 소스 코드는 문서를 로드하고, 문서를 분할한 후 벡터 표현을 만들고, 쿼리에 기반한 가장 관련 있는 문서를 반환하는 기능들을 포함하고 있습니다. 이러한 모든 기능은 사용자의 입력에 따라 동적으로 실행되게 됩니다. 사용자의 입력이 "quit"일 때까지 이 과정은 계속 반복됩니다.

## 함수 목록
- chat_with_user
- __init__
- load
- __init__
- split_documents
- embed_text
- __init__
- similarity_search
- as_retriever
- __init__
- get_relevant_documents
- __init__
- invoke

## 함수 설명
### chat_with_user

'chat_with_user' 함수는 파이썬 코드에서 사용자와의 대화를 처리하는 기능을 합니다. 사용자로부터 메시지를 입력받아, 'user_message'라는 매개변수를 통해 함수 내부로 전달합니다. 'chain.invoke' 메서드를 사용하여 이 'user_message'를 인자로 받아 AI가 처리하게 하는 코드를 실행합니다. AI는 이 메시지를 처리하고 결과를 'ai_message'라는 변수에 저장합니다. 이 'ai_message'는 함수의 반환값으로 사용자에게 다시 전달됩니다. 이런 과정을 통해, 사용자의 메시지를 AI가 받아 처리하고, 처리된 메시지를 사용자에게 다시 전달하는 역할을 하는 함수입니다.

```python
def chat_with_user(user_message):
    ai_message = chain.invoke(user_message)
    return ai_message
```

### __init__

파이썬 코드에서 '__init__' 함수는 클래스의 인스턴스가 만들어질 때 자동으로 호출되는 특별한 메소드입니다. 이는 클래스의 초기화를 담당하며, '생성자'라고도 불립니다. 위의 '__init__' 함수는 'file_path'라는 파라미터를 받아 클래스의 인스턴스 변수 'self.file_path'에 저장하는 역할을 합니다. 이렇게 인스턴스 생성 시에 필요한 초기 설정을 '__init__' 함수에서 처리하게 됩니다. 이 변수는 같은 클래스 내의 다른 메소드에서도 사용 가능하다는 특징이 있습니다. 따라서, 이 '__init__' 함수는 'file_path' 정보를 가진 객체를 생성할 때 사용하게 됩니다.

```python
    def __init__(self, file_path):
        self.file_path = file_path
```

### load

'load' 함수는 Python 코드에서 특정 파일을 읽고 그 내용을 반환하는 역할을 합니다. 먼저 빈 문자열 text를 선언하고, 이어서 'open' 함수를 이용해 'self.file_path'에 지정된 파일을 'r' 즉, 읽기 모드로 열고, 이를 'file'이라는 변수에 할당합니다. 여기서 'encoding='utf-8''는 파일을 열 때 사용할 인코딩을 지정하며, 주로 한글 등의 다국어 처리를 위해 사용됩니다. 'with' 문을 사용하면 파일을 열고 난 후에 자동으로 파일을 닫아주기 때문에, 별도로 'file.close()'를 호출할 필요가 없습니다. 그 다음, 'file.read()'를 이용해 파일의 모든 내용을 읽어 문자열로 반환하며, 이를 다시 'text'에 할당합니다. 마지막으로 'text'를 반환함으로써, 해당 함수를 호출한 곳에 파일의 내용을 전달하게 됩니다.

```python
    def load(self):
        text = ''
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
```

### __init__

Python 코드에서 `__init__`는 클래스의 생성자 함수입니다. 클래스의 인스턴스를 생성할 때 자동으로 호출되는 특수 메서드입니다. `__init__` 함수는 클래스가 인스턴스화 될 때 실행되며, 클래스의 속성을 초기화하거나 객체의 초기 상태를 설정하는 등 필요한 작업을 수행합니다. 예를 들어, 위에서 `__init__` 함수는 file_path라는 매개변수를 받아서 클래스의 속성으로 설정합니다. 즉, 객체가 생성될 때 file_path를 설정하여 해당 경로를 객체가 사용할 수 있도록 합니다. 이는 객체 지향 프로그래밍에서 중요한 개념으로, 이를 통해 코드의 재사용성이 높아집니다.

```python
    def __init__(self, file_path):
        self.file_path = file_path
```

### split_documents

'`split_documents` 함수는 전달받은 문서를 여러 부분으로 나누는 역할을 합니다. 먼저 이 함수는 전달받은 문서를 'separator_pattern'에 따라 분할합니다. 그리고 'chunks'라는 빈 리스트와 첫 번째 부분을 'current_chunk'로 설정합니다. 그 다음 각 분할 부분에 대해, 만약 'current_chunk'와 현재 분할 부분, 그리고 'separator'의 길이 합이 'chunk_size'보다 크다면 'current_chunk'를 'chunks' 리스트에 추가하고 'current_chunk'를 현재 분할 부분으로 설정합니다. 그렇지 않다면, 'current_chunk'에 'separator'와 현재 분할 부분을 추가합니다. 마지막으로, 'current_chunk'에 내용이 있다면 'chunks' 리스트에 추가합니다. 이 함수는 분할된 문서 부분들을 담은 'chunks' 리스트를 반환합니다.

```python
    def split_documents(self, documents):

        splits = documents.split(self.separator_pattern)

        chunks = []
        current_chunk = splits[0]

        for split in tqdm(splits[1:], desc="splitting..."):

            if len(current_chunk) + len(split) + len(self.separator) > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = split
            else:
                current_chunk += self.separator
                current_chunk += split

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
```

### embed_text

'embed_text' 함수는 OpenAI 클라이언트를 이용해서 텍스트를 임베딩하는 역할을 합니다. 먼저, OpenAI 객체를 생성하고, 이 객체의 'embeddings.create' 메소드를 호출하여 텍스트를 임베딩합니다. 이때, 입력값으로 텍스트와 모델명("text-embedding-ada-002")을 넘겨줍니다. 그리고 이 메소드의 응답으로부터 임베딩 데이터를 추출하여 반환합니다. 이 함수는 주어진 텍스트를 OpenAI의 특정 임베딩 모델을 이용해 벡터 형태로 변환하는 기능을 수행합니다.

```python
    def embed_text(self, text):
        client = OpenAI()
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
```

### __init__

Python 코드에서 '__init__' 함수는 클래스가 객체를 생성할 때 자동으로 호출되는 특별한 메소드입니다. 이 메소드를 생성자 또는 초기화 메소드라고도 부릅니다. 주어진 예제에서 '__init__'는 'file_path'라는 이름의 매개변수를 받고, 이를 self 객체의 속성으로 저장합니다. 여기서 'self'는 클래스의 인스턴스를 참조하는 변수로, 이를 통해 클래스의 속성과 메소드에 접근할 수 있습니다. 따라서 'self.file_path = file_path'라는 코드는 생성된 객체에 파일 경로를 저장하는 역할을 합니다. 이렇게 '__init__' 메소드를 통해 객체가 생성될 때 필요한 초기 설정을 할 수 있습니다.

```python
    def __init__(self, file_path):
        self.file_path = file_path
```

### similarity_search

'similarity_search' 함수는 주어진 쿼리와 가장 유사한 문서들을 찾는 함수입니다. 이 함수는 쿼리와 k(default 값은 4)를 입력받습니다. 먼저, 쿼리를 벡터로 변환하기 위해 임베딩 함수 'embed_text'를 사용합니다. 만약 벡터가 없다면 빈 리스트를 반환합니다. 그 다음에, 코사인 유사도를 이용하여 쿼리 벡터와 각 문서 벡터의 유사도를 계산합니다. 이 유사도를 기반으로 문서들을 정렬하고, 가장 유사도가 높은 상위 k개의 문서를 반환합니다. 여기서 유사도가 높다는 것은 쿼리와 해당 문서가 매우 유사하다는 것을 의미합니다.

```python
    def similarity_search(self, query, k=4):
        query_vector = self.embedding.embed_text(query)

        if not self.vectors:
            return []

        similarities = cosine_similarity([query_vector], self.vectors)[0]
        sorted_doc_similarities = sorted(zip(self.documents, similarities), key=lambda x: x[1], reverse=True)

        return sorted_doc_similarities[:k]
```

### as_retriever

'as_retriever' 함수는 Python 코드에서 사용되는 메소드입니다. 이 함수는 객체 자체를 인자로 받아 'SimpleRetriever' 클래스의 인스턴스를 반환하는 역할을 합니다. 즉, 이 메소드를 호출하는 객체는 자신을 기반으로 한 'SimpleRetriever' 객체를 생성하고 반환하는 역할을 합니다. 이렇게 하면 'SimpleRetriever' 클래스의 기능을 활용하여 원래 객체에 대한 다양한 처리를 수행할 수 있게 됩니다.

```python
    def as_retriever(self):
        return SimpleRetriever(self)
```

### __init__

파이썬 코드에서 '__init__' 함수는 클래스가 객체를 생성할 때 자동으로 호출되는 특별한 메소드입니다. 이것을 생성자(constructor)라고 부르며, 객체가 생성될 때 객체의 초기 상태를 정의하는데 사용됩니다. 위에서 제공된 예제에서, '__init__' 함수는 'self'와 'file_path' 두 개의 매개변수를 받습니다. 'self'는 클래스의 인스턴스를 참조하는데 사용되며, 'file_path'는 사용자가 제공한 경로를 참조합니다. 이 '__init__' 함수는 'self.file_path'라는 인스턴스 변수를 초기화하기 위해 사용됩니다. 이 변수는 나중에 클래스의 다른 메소드에서 사용할 수 있습니다.

```python
    def __init__(self, file_path):
        self.file_path = file_path
```

### get_relevant_documents

'get_relevant_documents' 함수는 주어진 쿼리와 관련된 문서들을 찾아 반환하는 기능을 합니다. 이 함수는 자체 벡터 저장소인 'vector_store'에서 'similarity_search' 메소드를 사용하여 쿼리와 유사한 문서들을 검색합니다. 이렇게 찾아낸 문서들은 'docs'에 저장되고, 이 'docs'가 함수의 반환값으로 돌아가게 됩니다. 즉, 이 함수를 사용하면, 벡터 저장소에서 주어진 쿼리와 유사도가 높은 문서들을 찾아낼 수 있습니다.

```python
    def get_relevant_documents(self, query):
        docs = self.vector_store.similarity_search(query)
        return docs
```

### __init__

파이썬 코드에서 '__init__' 함수는 클래스를 초기화하는 역할을 합니다. 클래스가 인스턴스화될 때마다 자동으로 호출되는 특별한 메소드로, 이 함수를 생성자라고도 부릅니다. 위의 '__init__' 함수는 file_path라는 인자를 받아 클래스 인스턴스의 속성으로 설정합니다. 즉, 이 클래스는 인스턴스화될 때마다 file_path값을 필요로 하며, 그 값을 self.file_path에 저장합니다. 이렇게 저장된 값은 클래스 내의 다른 메소드에서도 사용될 수 있습니다. 이렇게 '__init__' 함수는 인스턴스가 만들어질 때 필요한 초기 설정을 진행하는 역할을 합니다.

```python
    def __init__(self, file_path):
        self.file_path = file_path
```

### invoke

'invoke'라는 함수는 Python에서 특정 기능을 실행하는 역할을 합니다. 먼저, 사용자의 질문(query)를 인자로 받아들입니다. 그 후, 'retriever.get_relevant_documents(query)'를 사용하여 질문과 관련된 문서들을 찾습니다. 이때, 검색된 각 문서에 대해 그 번호와 내용을 출력합니다. 다음으로, 'openai.chat.completions.create'를 사용하여 'gpt-3.5-turbo' 모델에 질문과 관련된 문서들을 바탕으로 대화를 생성하고, 이 결과를 'completion'에 저장합니다. 마지막으로, 이 대화 결과 중 첫 번째 선택사항의 내용을 반환합니다. 이 함수는 챗봇이 사용자의 질문에 대한 응답을 생성하기 위해 사용됩니다.

```python
    def invoke(self, query):

        docs = retriever.get_relevant_documents(query)

        for i, doc in enumerate(docs):
            print("[#" + str(i) + "]", doc[1])
            print(doc[0])

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt_template.format(content=docs)},
                {"role": "user", "content": query}
            ]
        )

        return completion.choices[0].message.content
```


## 전체 소스코드
```python
# -*- coding: utf-8 -*-
"""building_a_simple_vectorstore.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/146n-YEebKvBN9qv2ShFsNeZpGZY2tel4
"""

# !pip install openai # openai 라이브러리를 설치합니다.
# !pip install tqdm

import os
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "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" # 환경변수에 OPENAI_API_KEY를 설정합니다.

class SimpleTextLoader:

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        text = ''
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

class SimpleCharacterTextSplitter:

    def __init__(self, chunk_size, chunk_overlap, separator_pattern='\n\n'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator_pattern = separator_pattern

    def split_documents(self, documents):

        splits = documents.split(self.separator_pattern)

        chunks = []
        current_chunk = splits[0]

        for split in tqdm(splits[1:], desc="splitting..."):

            if len(current_chunk) + len(split) + len(self.separator) > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = split
            else:
                current_chunk += self.separator
                current_chunk += split

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

from openai import OpenAI

class OpenAIEmbeddings:

    def embed_text(self, text):
        client = OpenAI()
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVectorStore:
    def __init__(self, docs, embedding):
        self.embedding = embedding
        self.documents = []
        self.vectors = []

        for doc in tqdm(docs, desc="embedding..."):
            self.documents.append(doc)
            vector = self.embedding.embed_text(doc)
            self.vectors.append(vector)

    def similarity_search(self, query, k=4):
        query_vector = self.embedding.embed_text(query)

        if not self.vectors:
            return []

        similarities = cosine_similarity([query_vector], self.vectors)[0]
        sorted_doc_similarities = sorted(zip(self.documents, similarities), key=lambda x: x[1], reverse=True)

        return sorted_doc_similarities[:k]

    def as_retriever(self):
        return SimpleRetriever(self)

class SimpleRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def get_relevant_documents(self, query):
        docs = self.vector_store.similarity_search(query)
        return docs

raw_documents = SimpleTextLoader('state_of_the_union.txt').load()
text_splitter = SimpleCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = SimpleVectorStore(documents, OpenAIEmbeddings())

documents[0]

len(documents)

documents

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0][0])

import urllib.request

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/puzzlet/constitution-kr/master/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD%20%ED%97%8C%EB%B2%95.txt",
    filename="korea_constitution.txt"
)

raw_documents = SimpleTextLoader('korea_constitution.txt').load()
text_splitter = SimpleCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

documents

db = SimpleVectorStore(documents, OpenAIEmbeddings())

query = "대통령의 임기는 몇 년인가?"
docs = db.similarity_search(query)

docs

retriever = db.as_retriever()

unique_docs = retriever.get_relevant_documents(query="대통령의 임기는 몇 년인가?")

unique_docs

# !pip install openai # openai 라이브러리를 설치합니다.

import openai

system_prompt_template = ("You are a helpful assistant. "
                          "Based on the following content, "
                          "kindly and comprehensively respond to user questions."
                          "[Content]"
                          "{content}"
                          "")

class SimpleRetrievalQA():

    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, query):

        docs = retriever.get_relevant_documents(query)

        for i, doc in enumerate(docs):
            print("[#" + str(i) + "]", doc[1])
            print(doc[0])

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt_template.format(content=docs)},
                {"role": "user", "content": query}
            ]
        )

        return completion.choices[0].message.content

chain = SimpleRetrievalQA(retriever)

answer = chain.invoke("대통령은 중임할 수 있나요?")

print(">> ", answer)

def chat_with_user(user_message):
    ai_message = chain.invoke(user_message)
    return ai_message

while True:
    user_message = input("USER > ")
    if user_message.lower() == "quit":
        break
    ai_message = chat_with_user(user_message)
    print(f" A I > {ai_message}")
```

