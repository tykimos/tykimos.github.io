---
layout: post
title: "코딩대신 말로만 만든 게임 동물의 똥숲"
author: 김태영
date: 2024-7-15 01:00:00
categories: 클로드 claude 게임
comments: true
image: http://tykimos.github.io/warehouse/2024/2024-7-15-creating_an_animal_droppings_crossing_game_with_claude_title.jpg
---

클로드(Claude)의 아티팩트(Artifacts)를 이용하여 코딩대신 말로만으로 만든 게임을 소개드립니다. 그 첫번째 게임은 바로 "동물의 똥숲"입니다.

### 함께보기

* [1편 - 클로드를 이용하여 픽셀게임 동물 캐릭터 만들기](https://tykimos.github.io/2024/07/13/creating_pixel_game_animal_characters_with_claude/)
* [2편 - 클로드를 이용한 픽셀 게임판 만들기](https://tykimos.github.io/2024/07/14/creating_a_pixel_game_board_with_claude/)
* [3편 - 말로만 만든 게임 동물의 똥숲](https://tykimos.github.io/2024/07/15/creating_an_animal_droppings_crossing_game_with_claude/)

### 결과 영상

먼저 결과 영상을 보시죠.

<iframe width="100%" height="400" src="https://www.youtube.com/embed/QCEzBZClQp8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### 프롬프트 및 결과

Claude의 artifacts 기능을 활용하여 개발했으며, React 기반으로 되어 있습니다.

#### 게임 컨셉 구상

먼저 게임의 기본 컨셉을 구상했습니다. "동물의 똥숲"은 팩맨 스타일의 미로 게임으로, 플레이어가 외계인 캐릭터를 조종하여 맵에 있는 모든 똥을 수집하는 것이 목표입니다. 동시에 다른 동물 캐릭터들을 피해야 합니다.

#### 캐릭터 디자인

게임에 필요한 캐릭터들을 SVG 형식으로 디자인했습니다. Claude에게 각 캐릭터의 특징을 설명하고, SVG 코드로 표현해달라고 요청했습니다. 캐릭터 제작이 궁금하시면 [1편 - 클로드를 이용하여 픽셀게임 동물 캐릭터 만들기](https://tykimos.github.io/2024/07/13/creating_pixel_game_animal_characters_with_claude/)을 참고하십시요.

1편에 이어 추가된 캐릭터는 바로 이번 게임의 주인공인 "라키루키"입니다. 라키루키는 자리안(Zarian) 행성에서 온 외계인 친구입니다~

![img](http://tykimos.github.io/warehouse/2024/2024-7-15-creating_an_animal_droppings_crossing_game_with_claude_1.jpg)

#### 맵 디자인

게임의 맵을 설계했습니다. 우물 모양의 구조를 기본으로 하되, 모든 영역에 접근할 수 있도록 통로를 만들었습니다. 맵은 19x19 그리드로 구성되어 있으며, 각 셀은 다음 중 하나의 상태를 가집니다:

* 0: 빈 공간 (이동 가능)
* 1: 나무 (장애물)
* 2: 숲 (외벽)
* 3: 우물 (장애물)

저는 장애물과 길이 있다고만 말했고, 위 상태에 대한 설계는 모두 Claude가 알아서 해줬답니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-7-15-creating_an_animal_droppings_crossing_game_with_claude_2.jpg)

#### 게임 로직 구현

React hooks를 사용하여 게임의 주요 로직을 구현했습니다:

* useState: 게임 상태 관리 (플레이어 위치, 동물 위치, 똥 위치, 점수 등)
* useEffect: 키보드 입력 처리, 동물 이동, 게임 루프 구현
* useRef: 캔버스 및 이미지 캐시 참조

주요 기능:

* 플레이어 이동
* 동물 AI (랜덤 이동)
* 충돌 감지
* 똥 수집 및 점수 계산

위 내용도 사실 어떻게 구현이 되었는 지 코드 리뷰도 하지 않았습니다. 

#### 그래픽 구현
HTML5 Canvas를 사용하여 게임 그래픽을 구현했습니다. SVG 이미지를 캔버스에 그리는 방식으로 캐릭터를 표현했고, 맵과 똥은 캔버스 API를 직접 사용하여 그렸습니다.

#### UI 디자인

게임 화면 옆에 귀여운 스타일의 정보 패널을 추가했습니다. 이 패널에는 다음 정보가 표시됩니다:

* 게임 제목
* 현재 점수
* 남은 생명 (외계인 아이콘으로 표시)
* 게임 오버 시 재시작 버튼

UI를 만들기 위해서 아래처럼 지시문을 작성했습니다.

![img](http://tykimos.github.io/warehouse/2024/2024-7-15-creating_an_animal_droppings_crossing_game_with_claude_3.jpg)

#### 최적화

아티팩트는 구동 가능한 텍스트 길이가 제한되어 있기 때문에 코드의 양을 최대한 줄여야 합니다. 코드를 최대한 간결하게 만들기 위해 여러 차례 리팩토링을 진행했습니다. 불필요한 렌더링을 줄이고, 반복되는 로직을 함수로 추출하는 등의 작업을 수행했습니다. 이 과정을 위해서 아래처럼 지시문을 작성했습니다. 

![img](http://tykimos.github.io/warehouse/2024/2024-7-15-creating_an_animal_droppings_crossing_game_with_claude_4.jpg)

#### 테스트 및 디버깅

게임을 여러 번 플레이해보며 버그를 수정하고 게임플레이를 개선했습니다. 특히 맵 디자인을 여러 번 수정하여 모든 똥을 수집할 수 있도록 만드는 데 주력했습니다.

#### 배포

Claude의 artifacts 기능을 사용하여 게임을 배포했습니다. 이 기능은 React 애플리케이션을 지원하며, 완성된 게임을 즉시 플레이할 수 있는 링크를 제공합니다.

![img](http://tykimos.github.io/warehouse/2024/2024-7-15-creating_an_animal_droppings_crossing_game_with_claude_5.jpg)

### 게임해보기

실제 게임은 아래 링크를 통해 해보실 수 있습니다. 

[게임 해보기](https://claude.site/artifacts/77177a5e-efaa-4329-a239-e8f6163f663c)

### 시놉시스

자리안 행성은 자연적인 자정작용이 부족하여 심각한 환경 위기를 겪고 있다. 바이오가스는 이 위기를 해결할 유일한 에너지원으로 자리안의 과학자들은 지구에서 풍부한 동물 배설물을 통해 이를 얻을 수 있음을 발견했다. 이에 따라 자리안 행성에서 귀여운 외계인 라키루키가 지구로 파견된다.

라키루키는 지구의 숲속에서 동물들을 피해 다니며 배설물을 수집해야 한다. 임무를 수행하는 동안 라키루키는 다채로운 지구의 동물들과 만나며, 그들과의 우정을 쌓아간다. 그러나 라키루키는 인간들에게 들키지 않고 임무를 완수해야 하는 어려움에 직면한다. 라키루키는 지구의 다양한 환경을 탐험하며, 자리안 행성의 에너지 위기를 해결할 중요한 자원을 확보해 나간다.

라키루키는 예상치 못한 사건들을 겪으면서 성장하고, 지구의 동물들과 상호작용하며 소중한 교훈을 얻는다. 과연 라키루키는 지구에서 무사히 임무를 완수하고 자리안 행성으로 돌아갈 수 있을까? 라키루키의 모험은 시작된다.

(아래는 미드저니로 그린 라키루키의 컨셉 아트입니다.)

![img](http://tykimos.github.io/warehouse/2024/2024-7-15-creating_an_animal_droppings_crossing_game_with_claude_6.png)

### 결론

지금까지 Claude의 Artifacts를 이용해서 대화만으로 게임을 만들어보는 방법에 대해 알아봤습니다. 앞으로 AI를 활용한 게임 개발이 더욱 보편화되면, 개발자들은 더 창의적이고 복잡한 게임 로직과 디자인에 집중할 수 있게 될 것입니다. 