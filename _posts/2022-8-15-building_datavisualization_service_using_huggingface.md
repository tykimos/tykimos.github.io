---
layout: post
title:  "허깅페이스로 데이터 시각화 웹서비스 간단히 만들기"
author: Taeyoung Kim
date:   2022-8-15 00:00:00
categories: tech
comments: true
image: http://tykimos.github.io/warehouse/2022-8-15-building_datavisualization_service_using_huggingface_title1.png
---

안녕하세요. 타이키모스입니다. 이번 시간에는 허깅페이스로 데이터 시각화 웹서비스를 간단하게 만들어보겠습니다.회원수나 판매량 등의 누적량 그래프를 파이썬 시각화 패키지로 그리고 있었는데요.
웹 서비스로 만들고싶어 아래 조건을 만족하는 방법을 찾아보고 있었습니다.

* 첫째, 웹서비스이지만 파이썬 코드로만 가능할 것.
* 둘째, 별도의 서버나 클라우드 비용이 필요하지 않을 것.
* 셋째, 개발 및 유지보수가 쉬울 것.
* 넷째, 깔끔할 것.

여러가지 방법을 고민했으나 결론은 허깅페이스과 스트림릿 조합으로 만들게 되었습니다. 그럼 함께 만들어볼까요?

먼저 허깅페이스에 접속하여 로그인을 합니다. 상단 메뉴에 보면 Spaces가 있는데요. Spaces는 머신러닝 데모용 서비스를 아주 쉽게 생성하고 실제 웹에서 서비스가 될 수 있도록 호스트하기 위한 공간입니다. 그럼 이제 나만의 Space를 하나 만들어보겠습니다. 상단에 Create new Space 버튼을 클릭합니다.

Space 이름을 설정한 후, 적절한 라이센스 정책을 선택합니다. 다음은 Space SDK를 설정합니다.

다양한 웹인터페이스를 지원하는 서비스일 경우 스트림릿을, 직관적이고 심플한 입출력 인터페이스 중심의 서비스라면 그라디오를 추천드립니다. 저는 데이터 시각화 기능이 제공되는 스트림릿으로 선정했습니다.
마지막으로는 공개와 비공개를 선택할 수 있습니다. 모두 설정하였다면, Create space 버튼을 선택합니다.

짜잔! 아직은 비어있지만, Space가 만들어졌습니다. 레포 복사 및 커밋하는 방법이 소개되어 있지만, 웹 상에서 바로 앱 파일을 만들어보겠습니다. Create 링크를 클릭합니다. 그럼 웹 상에서 앱 파일을 바로 수정할 수 있는 창이 띄워지는데요. 미리 짜둔 소스코드를 복사붙이기를 하겠습니다. 이 코드는 회원 가입, 판매이력 등 날짜가 포함된 데이터를 누적량으로 표시하는 소스코드입니다. 소스코드 입력을 다 했다면, 커밋 뉴 파일 버튼을 클릭합니다.웹에서 서비스가 구동될 수 있도록 앱을 구성하는 동안에는 Building이라고 표시됩니다. 완료되면 Running으로 바뀝니다. 시간이 살짝 걸리니, 소스코드를 간단하게만 살펴볼까요?

처음에는 필요한 파이썬 패키지 불러오는 코드가 있습니다.

  import streamlit as st
  import pandas as pd
  import numpy as np
  from datetime import datetime, timedelta
  from io import StringIO
    
에스티가 스트림릿 패키지의 약자인데요. 스트림릿에 타이틀을 설정하고, 파일 업로드 컨트롤을 추가합니다.

  st.title('Cumulative Trend')

  uploaded_file = st.file_uploader("Choose a csv file including 'date' column.")

  if uploaded_file is not None:

다음은 업로드된 CSV파일에서 데이트 컬럼을 불러온 뒤 결측치를 제거합니다.

      df = pd.read_csv(uploaded_file, sep=",", usecols=["date"])
      df = df.dropna(axis=0)

      date_list = pd.to_datetime(df.squeeze()).dt.date.tolist()
      date_list.sort()
      
데이터의 시작 날짜와 마지막 날짜 정보를 얻어옵니다.

      start_date = date_list[0]
      end_date = date_list[-1]
      day_count = (end_date - start_date).days + 1
      
X는 일일 단위로 시작 날짜부터 마지막 날짜까지 설정합니다. Y에는 해당일에 데이터가 있다면 1을 추가하여 해당일에 대한 데이터 개수를 계산합니다.

      x = np.arange(start_date, end_date + timedelta(days=1), timedelta(days=1)).astype(datetime)
      y = np.zeros(day_count)

Y의 누적량을 계산하여 Y 언더바 카운트에 저장합니다.

      for d in date_list:
          y[(d - start_date).days] += 1

      y_count = np.cumsum(y)

X와 Y 언더바 카운트를 각각 X축와 Y축으로 설정한 뒤, 스트림릿에서 제공하는 영역 차트를 이용하여 시각화 시킵니다.

      df = pd.DataFrame({'date': x, 'count': y_count})

마지막으로 유효한 행 수와 날짜수 그리고 시작일과 종료일을 표시합니다.

      st.area_chart(df.set_index('date'))

      col1, col2 = st.columns(2)
      col1.metric("rows", len(date_list))
      col2.metric("days", day_count)

      col3, col4 = st.columns(2)
      col3.metric("start", str(start_date))
      col4.metric("end", str(end_date))

그럼 상단 메뉴에서 앱을 클릭해볼까요? 바로 서비스가 만들어졌습니다. 반가운 파일선택버튼이 보이네요. Browse files 버튼을 클릭하여 준비된 CSV파일을 선택합니다. 단, CSV파일에는 date라는 열이 포함되어 있어야 합니다. 그럼 업로드 진행바가 잠시 보인 후 업로드가 완료되면 그래프가 보입니다. X축은 날짜, Y축은 누적 회원 가입수가 보이네요. 하단에는 매트릭으로 표현한 여러 값들이 보입니다. 그래프에 마우스를 올려보면, 마우스 지점의 해당하는 정보를 볼 수 있습니다. 그리고 점세개 메뉴를 선택하면 그래프를 이미지로 다운로드 받을 수 있습니다. 이렇게 데이터 시각화 웹서비스를 간단하게 만들어봤습니다. 웹상에서 호스팅을 하기 위해 허깅페이스의 Space를 사용하였고, 스트림릿을 이용하여 인터페이스를 구현했습니다. 여러분도 데이터시각화용 웹서비스를 쉽고 빠르게 만들어보세요.
