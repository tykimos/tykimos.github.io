---
layout: post
title:  "Apple Silicon 칩(M1)에 미디어파이프(Mediapipe) 설치하기"
author: Taeyoung Kim
date:   2022-12-01 00:00:00
categories: tech
comments: true
image: http://tykimos.github.io/warehouse/2022-12-01-Mediapipe_installing_on_macOS_with_Apple_Silicon_title1.png
---

안녕하세요. 타이키모스입니다. 미디어파이프를 파이썬에서 간단하게 테스트해보려면 코랩이나 파이썬 가상환경에서 설치해도 되지만, 모바일 기기나 데스크탑 용으로 어플리케이션을 개발하려면 설치 및 빌드까지 해야합니다. 미디어파이프 공식페이지에 아래처럼 macOS 설치법이 있으나, Apple Silicon칩 기반 M1 유저인 경우에는 제대로 설치가 안됩니다. 여러가지 방법이 있을 수 있겠으나 제가 설치에 성공한 버전으로 정리해봤습니다.

![image]([http://tykimos.github.io/warehouse/2022-12-01-Mediapipe_installing_on_macOS_with_Apple_Silicon_title1.png](http://tykimos.github.io/warehouse/2022-12-01-Mediapipe_installing_on_macOS_with_Apple_Silicon_title1.png))

1. Homebrew 설치

터미널에서 아래 명령으로 설치합니다.

```bash
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Xcode 설치

터미널에서 아래 명령으로 설치합니다.

```bash
$ xcode-select --install
```

만약 제대로 설치가 안되고 아래와 같은 오류를 만나셨나요?

```bash
xcode-select: error: command line tools are already installed, use "Software Update" to install updates
```

그럼 이미 설치되어서 나오는 에러이므로 삭제하고 다시 설치합니다.

```bash
$ sudo rm -rf /Library/Developer/CommandLineTools
$ sudo xcode-select --install
```

승인 팝업 창이 뜨면 확인 버튼을 클릭하여 설치를 완료합니다.

3. 미디어파이프 다운로드

깃 클론을 통해서 미디어파이프를 다운로드 합니다.

```bash
$ git clone https://github.com/google/mediapipe.git
$ cd mediapipe
```

4. 미디어파이프에 설정된 바젤(bazel) 버전 확인

mediapipe 폴더 내에 ".bazelversion" 파일을 열어 바젤 버전을 확인합니다. 저의 경우에는 5.2.0으로 표기되어 있습니다.

```bash
5.2.0
```

5. 4번에서 확인한 바젤 버전으로 설치합니다. "5.2.0"으로 확인되었으니 이 버전으로 설치합니다.

먼저 바젤 설치 파일을 다운로드 합니다.

```bash
$ export BAZEL_VERSION=5.2.0
$ curl -fLO "https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-darwin-x86_64.sh"
```

바젤 설치를 시작합니다.

```bash
$ chmod +x "bazel-$BAZEL_VERSION-installer-darwin-x86_64.sh"
$ ./bazel-$BAZEL_VERSION-installer-darwin-x86_64.sh --user
```

바젤 실행 파일 경로를 추가합니다.

```bash
$ export PATH="$PATH:$HOME/bin"
```

바젤 버전을 확인합니다. "5.2.0"으로 설정했으니, "5.2.0"가 나오면 정상입니다.

```bash
bazel --version
```

6. OpenCV와 FFmpeg를 설치합니다.

```bash
$ brew install opencv@3
$ brew uninstall --ignore-dependencies glog
```

7. 파이썬을 설치합니다. 

```bash
$ brew install python
$ sudo ln -s -f /usr/local/bin/python3.7 /usr/local/bin/python
$ python --version
$ pip3 install --user six
```

8. mediapipe/framework/port/BUILD 파일에서 링크옵션을 추가합니다.

변경전
```
cc_library(
    name = "status",
    hdrs = [
        "canonical_errors.h",
        "status.h",
        "status_builder.h",
        "status_macros.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":source_location",
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
```

변경후
```
cc_library(
    name = "status",
    hdrs = [
        "canonical_errors.h",
        "status.h",
        "status_builder.h",
        "status_macros.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":source_location",
        "//mediapipe/framework:port",
        "//mediapipe/framework/deps:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    linkopts = select({
        "//mediapipe:macos": [
            "-framework CoreFoundation",
        ],
        "//conditions:default": [],
    }),
)
```

위 과정을 수행하지 않으면 아래 에러가 발생합니다.

```
dyld[99566]: symbol not found in flat namespace '_CFRelease'
```

9. "Hello World!"의 C++ 예제를 실행합니다.
 
```bash
$ export GLOG_logtostderr=1
$ bazel run --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world:hello_world
```

10. 아래와 같이 터미널에 출력되면 정상적으로 설치된 것입니다.

```bash
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
```

설치과정은 늘 녹록지 않고 삽질하는 기분이네요. 설치가 잘 안되면 스트레칭 한 번 하고 다시 시작해봅시다!
