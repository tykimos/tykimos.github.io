---
layout: post
title:  "심플 어노테이션 툴 - 첫번째"
author: 김태영
date:   2018-10-16 13:00:00
categories: lecture
comments: true
image: http://tykimos.github.io/warehouse/2018-10-16_Simple_Annotation_Tool_1_title.png
---
김윤기님이 만든 시각장애인 안내하는 'AI' 저에게는 많은 의미에서 감동적이었습니다. 사용목적, 열정, 성능, 노력 등등... 그 중에 어노테이션 툴 (라벨링을 하기 위한 툴)을 30분만에 만들었다는 게시물을 보고, 아차 싶었습니다. 딥러닝 적용을 위해 개발자와 도메인 전문가가 협업을 한다고 했을 때, 어노테이션 툴은 개발자의 몫이라고 생각하고 있었거든요. 그 틀이 깨지는 순간이었습니다. 와~ 어노테이션 툴 개발도 도메인 전문가 영역으로 넘길 수 있겠다는 생각이 들더군요. 

![img](http://tykimos.github.io/warehouse/2018-10-16_Simple_Annotation_Tool_1_1.gif)

소개용 소스코드로 정리하다보니 29줄이 되더군요. 30줄도 안되니 한 번 따라해볼까요? 찬찬히 설명해드리겠습니다. 블로그 글로도 정리했지만 동영상 따라하기를 좋아하시는 분을 위해 녹화도 해봤습니다.

[![video](http://tykimos.github.io/warehouse/2018-10-16_Simple_Annotation_Tool_1_title.png)](https://youtu.be/BYJ4Nr40Lew)

---
### 폴더 및 파일 구성

먼저 폴더 및 파일 구성부터 알아보겠습니다. 

![img](http://tykimos.github.io/warehouse/2018-10-16_Simple_Annotation_Tool_1_2.png)

* data
    * test1.jpg
    * test2.jpg
* tool.py

data 폴더 안에는 라벨링을 해야할 이미지 파일들이 있습니다. tool.py는 이제 우리가 만들 파이썬 코드입니다. 

---
### 이미지  파일 차례차례 보기

그럼 data 폴더 안에 있는 이미지 파일을 하나 하나씩 보도록 할까요? 먼저 소스코드부터 보겠습니다.


```python
import glob # 파일들을 잡기 위한 패키지
import cv2 # 이미지 파일을 처리하기 위한 패키지

img_files = glob.glob('./data/*.jpg') # data폴더 안에 있는 모든 jpg 파일들 목록을 가져옵니다.

cv2.namedWindow('tool') # tool이란 윈도우(창)을 하나 띄웁니다.

for img_file in img_files: # 모든 jpg 파일들 목록에서 jpg 파일명을 하나하나씩 가지고 옵니다.
    img = cv2.imread(img_file) # 가지고 온 jpg 파일명으로부터 이미지를 불러옵니다.
    cv2.imshow('tool', img) # 불러온 이미지를 앞서 만든 tool 창에 띄웁니다.
    cv2.waitKey() # 사용자가 아무 키를 누르기 전 까지 이미지를 띄웁니다. 키를 누르면 다음 이미지를 불러옵니다.
```

실행시켜보면, 아래 그림처럼 창에 이미지가 띄워집니다. 참 쉽게 이미지 뷰어가 만들어지네요. 심지어 키를 누르면 다음 이미지로 바꿔집니다. 

![img](http://tykimos.github.io/warehouse/2018-10-16_Simple_Annotation_Tool_1_3.png)

---
### 이미지 위에 그리드 그리기

김윤기님 프로젝트에서는 인도를 표시하는 것입니다. 표시를 좀 더 쉽게 하기 위해서 이미지 위에 그리드를 표시하였고, 해당 영역을 표시하는 식입니다. BLOCK, ROW, COL을 정의하고, 이미지 위에다 사각형을 그리는 함수를 호출하면 그만입니다.


```python
import glob # 파일들을 잡기 위한 패키지
import cv2 # 이미지 파일을 처리하기 위한 패키지
import numpy as np

BLOCK = 80 # 한 칸의 크기 입니다. 가로 세로 80픽셀을 의미합니다.
ROW = 9 # 행이 9칸이란 뜻입니다. 영상 높이가 720 픽셀인 경우 이를 블록사이즈(80)으로 나눈 값입니다.
COL = 16 # 열이 16칸이란 뜻입니다. 영상 너비가 1280 픽셀인 경우 이를 블록사이즈(80)으로 나눈 값입니다.

grid = np.zeros([ROW, COL], dtype=np.int) # 9 x 16 2차원 배열을 만들어서 값을 0으로 채웁니다.

img_files = glob.glob('./data/*.jpg') # data폴더 안에 있는 모든 jpg 파일들 목록을 가져옵니다.

cv2.namedWindow('tool') # tool이란 윈도우(창)을 하나 띄웁니다.

for img_file in img_files: # 모든 jpg 파일들 목록에서 jpg 파일명을 하나하나씩 가지고 옵니다.
    img = cv2.imread(img_file) # 가지고 온 jpg 파일명으로부터 이미지를 불러옵니다.

    for r in range(ROW): # 행을 하나씩 선택합니다.
        for c in range(COL): # 열을 하나씩 선택합니다. 
            cv2.rectangle(img, (c*BLOCK, r*BLOCK), ((c+1)*BLOCK, (r+1)*BLOCK), (255, 255, 255), cv2.FILLED * grid[r][c]) # 네모를 그립니다.

    cv2.imshow('tool', img) # 불러온 이미지를 앞서 만든 tool 창에 띄웁니다.
    cv2.waitKey() # 사용자가 아무 키를 누르기 전 까지 이미지를 띄웁니다. 키를 누르면 다음 이미지를 불러옵니다.
```

실행하면 아래 그림처럼 그리드가 표시됩니다.

![img](http://tykimos.github.io/warehouse/2018-10-16_Simple_Annotation_Tool_1_4.png)

rectangle() 함수가 조금 복잡한데요. 사각형을 그리기 위해서는 아래 인자들을 지정해줘야 합니다.
    * 사각형을 그릴 이미지 : 여기서는 앞서 불러온 img이겠죠?
    * 왼쪽 위 좌표 : 행, 열 값에다가 블록 픽셀 사이즈를 곱해야 실제 픽셀값 위치를 계산할 수 있습니다.
    * 오른쪽 아래 좌표 : 행, 열 값에 1씩 더한 값에다 블록 픽셀 사이즈를 곱하면, 오른쪽 아래 좌표가 나오겠죠?
    * 색상 : 간단하게 흰색을 설정해봅니다.
    * 모드 : 여기서는 모드가 가장 중요한데요, 0이면 그냥 흰색 라인을 그리고, cv2.FILLED 값이면 흰색으로 채운 사각형이 나옵니다. 그래서 해당 grid 값이 0이면 라인이 나오고, 1이면 흰색으로 칠해지는 효과를 볼 수 있습니다.
    
아직까지는 grid가 모두 0으로 채워져 있으니 흰색 라인만 그려지겠죠?

---
### 마우스 클릭 기능 넣기

자 이번에는 마우스로 어떤 칸을 선택하면 색이 칠해지고, 칠해진 칸을 클릭하면 원상복구 되도록 만들어보겠습니다.


```python
import glob # 파일들을 잡기 위한 패키지
import cv2 # 이미지 파일을 처리하기 위한 패키지
import numpy as np

BLOCK = 80 # 한 칸의 크기 입니다. 가로 세로 80픽셀을 의미합니다.
ROW = 9 # 행이 9칸이란 뜻입니다. 영상 높이가 720 픽셀인 경우 이를 블록사이즈(80)으로 나눈 값입니다.
COL = 16 # 열이 16칸이란 뜻입니다. 영상 너비가 1280 픽셀인 경우 이를 블록사이즈(80)으로 나눈 값입니다.

grid = np.zeros([ROW, COL], dtype=np.int) # 9 x 16 2차원 배열을 만들어서 값을 0으로 채웁니다.

def click(event, x, y, flags, param): # 사용자가 마우스를 클릭했을 때 이 함수가 호출됩니다.
    if event == cv2.EVENT_LBUTTONDOWN: # 왼쪽 마우스를 클릭했을 때
        grid[y // BLOCK][x // BLOCK] = (grid[y // BLOCK][x // BLOCK] + 1) % 2 # 해당 그리드 칸이 0이라면 1로, 1이라면 0으로 만듭니다.

img_files = glob.glob('./data/*.jpg') # data폴더 안에 있는 모든 jpg 파일들 목록을 가져옵니다.

cv2.namedWindow('tool') # tool이란 윈도우(창)을 하나 띄웁니다.
cv2.setMouseCallback('tool', click) # tool 창에 마우스 클릭했을 때, click 함수를 호출하도록 설정합니다.

for img_file in img_files: # 모든 jpg 파일들 목록에서 jpg 파일명을 하나하나씩 가지고 옵니다.
    img = cv2.imread(img_file) # 가지고 온 jpg 파일명으로부터 이미지를 불러옵니다.
    grid.fill(0) # 이미자가 바뀌면 흰색 칠 했던 정보도 초기화 합니다.

    while True: # 스페이스 키를 누르기 전 까지는 계속 이미지를 표출합니다. 무한 반복해서 그리기 때문에 마우스로 클릭하면 그때 그때 바뀐 정보로 그려집니다.

        visual = img.copy() # 매번 img 이미지을 복제해서 visual 이미지로 복제합니다.

        for r in range(ROW): # 행을 하나씩 선택합니다.
            for c in range(COL): # 열을 하나씩 선택합니다. 
                cv2.rectangle(visual, (c*BLOCK, r*BLOCK), ((c+1)*BLOCK, (r+1)*BLOCK), (255, 255, 255), cv2.FILLED * grid[r][c]) # visual 이미지에 네모를 그립니다.

        cv2.imshow('tool', visual) # 사각형을 그린 이미지를 앞서 만든 tool 창에 띄웁니다.

        if cv2.waitKey(2) == 32:  # 사용자가 스페이스 키를 눌렸을 때 다음 이미지를 불러옵니다. 그렇지 않다면 2밀리초 이후에 다시 이미지를 그립니다. 
            break
```

세가지 정도 확인해보겠습니다.
    * 마우스 클릭 기능 추가 : click()이란 함수를 만들고, 이 함수를 'tool' 창에 추가했습니다.
    * 무한 반복으로 이미지 표시 : 마우스 클릭 시 마다 반응하기 위해서 이미지는 계속해서 표시하도록 만들었습니다. 이 상태에서 마우스를 클릭하면 grid 배열 값이 바뀌고, 이미지 표시 시에 바뀐 grid 값으로 선을 그리거나 색을 칠하는 방식입니다. 
    * 사각형 그리는 대상이 visual로 바뀐 점 : 매번 반복할 때마다 원본인 img에서 복제해서 visual를 만든다음 visual에 사각형을 그립니다. 만약 img에다가 사각형을 그리면, 원상복구가 힘들겠죠? 여기서는 마우스를 클릭하면 칠해지고 만약 실수로 클릭했다면 다시 해당 부분을 클릭해서 원상복구할 수 있습니다.
    
![img](http://tykimos.github.io/warehouse/2018-10-16_Simple_Annotation_Tool_1_5.png)

---
### 라벨링 정보를 파일에 쓰기

마지막입니다. 여기서 우리가 필요한 정보는 특정 이미지에 해당하는 색칠된 칸의 정보입니다. 


```python
import glob # 파일들을 잡기 위한 패키지
import cv2 # 이미지 파일을 처리하기 위한 패키지
import numpy as np

BLOCK = 80 # 한 칸의 크기 입니다. 가로 세로 80픽셀을 의미합니다.
ROW = 9 # 행이 9칸이란 뜻입니다. 영상 높이가 720 픽셀인 경우 이를 블록사이즈(80)으로 나눈 값입니다.
COL = 16 # 열이 16칸이란 뜻입니다. 영상 너비가 1280 픽셀인 경우 이를 블록사이즈(80)으로 나눈 값입니다.

grid = np.zeros([ROW, COL], dtype=np.int) # 9 x 16 2차원 배열을 만들어서 값을 0으로 채웁니다.

def click(event, x, y, flags, param): # 사용자가 마우스를 클릭했을 때 이 함수가 호출됩니다.
    if event == cv2.EVENT_LBUTTONDOWN: # 왼쪽 마우스를 클릭했을 때
        grid[y // BLOCK][x // BLOCK] = (grid[y // BLOCK][x // BLOCK] + 1) % 2 # 해당 그리드 칸이 0이라면 1로, 1이라면 0으로 만듭니다.

img_files = glob.glob('./data/*.jpg') # data폴더 안에 있는 모든 jpg 파일들 목록을 가져옵니다.

cv2.namedWindow('tool') # tool이란 윈도우(창)을 하나 띄웁니다.
cv2.setMouseCallback('tool', click) # tool 창에 마우스 클릭했을 때, click 함수를 호출하도록 설정합니다.

for img_file in img_files: # 모든 jpg 파일들 목록에서 jpg 파일명을 하나하나씩 가지고 옵니다.
    img = cv2.imread(img_file) # 가지고 온 jpg 파일명으로부터 이미지를 불러옵니다.
    grid.fill(0) # 이미자가 바뀌면 흰색 칠 했던 정보도 초기화 합니다.

    while True: # 스페이스 키를 누르기 전 까지는 계속 이미지를 표출합니다. 무한 반복해서 그리기 때문에 마우스로 클릭하면 그때 그때 바뀐 정보로 그려집니다.

        visual = img.copy() # 매번 img 이미지을 복제해서 visual 이미지로 복제합니다.

        for r in range(ROW): # 행을 하나씩 선택합니다.
            for c in range(COL): # 열을 하나씩 선택합니다. 
                cv2.rectangle(visual, (c*BLOCK, r*BLOCK), ((c+1)*BLOCK, (r+1)*BLOCK), (255, 255, 255), cv2.FILLED * grid[r][c]) # visual 이미지에 네모를 그립니다.

        cv2.imshow('tool', visual) # 사각형을 그린 이미지를 앞서 만든 tool 창에 띄웁니다.

        if cv2.waitKey(2) == 32:  # 사용자가 스페이스 키를 눌렸을 때 다음 이미지를 불러옵니다. 그렇지 않다면 2밀리초 이후에 다시 이미지를 그립니다. 
            flat = np.array(grid).flatten() # 2차원의 그리드 정보를 1차원으로 바꿉니다.
            fp_label = open('annotation.txt', 'a') # annotation.txt 파일을 추가쓰기 권한으로 엽니다.
            fp_label.writelines(img_file + ',' + ''.join(map(str, flat))+'\n') # 1차원으로 변환된 그리드 정보를 이쁘게 씁니다.
            fp_label.close() # 파일을 닫습니다.
            break
```

바뀐 부분은 마지막에 스페이스 키를 눌었을 때, grid를 파일로 저장하는 부분만 추가되었습니다. flatten() 하거나 .join() 등 조금 복잡한 것들이 보이지만 이것들은 이쁘게 파일에 쓰도록 하기 위함이라 정답이 있는 것은 아니고, 나중에 읽기 편한식으로 바꾸시면 됩니다.

![img](http://tykimos.github.io/warehouse/2018-10-16_Simple_Annotation_Tool_1_6.png)

횡단보고를 클릭한 다음 스페이스 키 누르면 다음 파일로 넘어가고 모든 파일에 대해서 라벨링을 다하면 프로그램이 종료됩니다. 한 이미지 작업을 마칠 때마다, 이미지 파일명과 사용자가 라벨링한 정보인 grid 내용이 01로 표시됩니다. '1'이면 횡단보고이고, '0'이면 아는 것이겠죠?

---
### 마치며

김윤기님의 원래 소스에는 이것보다 좀 더 많은 기능이 들어가 있습니다. 본 강좌에서는 입문하기 부담없을 정도, 그리고 자신감이 생길 정도로 간단한 소스코드와 기능 위주로 작성하였습니다. 따라하기에 성공하셨다면 원 소스를 한 번 살펴보시기 바랍니다~

![img](https://github.com/YoongiKim/Walk-Assistant/raw/master/img/cover.gif)

[https://github.com/YoongiKim/Walk-Assistant/blob/master/annotation.py](https://github.com/YoongiKim/Walk-Assistant/blob/master/annotation.py)

---

### 같이 보기

* [케라스 기초 강좌](https://tykimos.github.io/lecture/)
* [케라스 코리아](https://www.facebook.com/groups/KerasKorea/)
