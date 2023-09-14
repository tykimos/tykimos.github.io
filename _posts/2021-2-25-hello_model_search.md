---
layout: post
title:  "코랩에서 구글 모델 서치 구동하기"
author: 김태영
date:   2021-2-25 13:00:00
categories: python
comments: true
image: http://tykimos.github.io/warehouse/2021-2-25-hello_model_search_title_1.png
---
구글 모델 서치는 기존 NAS(네트워크 아키텍처 탐색)의 단점을 보안하고, 효율적이고 자동으로 최적의 모델을 개발할 수 있도록 오픈소스로 제공되는 플랫폼입니다. 이러한 구글 모델 서치를 코랩에서 바로 구동해볼 수 있도록 테스트한 결과를 공유합니다. 깃헙 설치부터 패키지 설치, 환경 설정, 예제 소스 코드 설명 그리고 출력값을 확인해보도록 하겠습니다. 

### 관련 정보
---

*   블로그 주소: https://ai.googleblog.com/2021/02/introducing-model-search-open-source.html
*   깃헙 주소: https://github.com/google/model_search



<a href="https://colab.research.google.com/github/tykimos/hello-ai/blob/main/hello_model_search.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### 환경설정
---

먼저 깃헙에서 소스코드 다운로드 받습니다.


```python
!git clone https://github.com/google/model_search.git
```

    Cloning into 'model_search'...
    remote: Enumerating objects: 141, done.[K
    remote: Counting objects: 100% (141/141), done.[K
    remote: Compressing objects: 100% (109/109), done.[K
    remote: Total 141 (delta 30), reused 136 (delta 28), pack-reused 0[K
    Receiving objects: 100% (141/141), 214.12 KiB | 3.02 MiB/s, done.
    Resolving deltas: 100% (30/30), done.


"model_search" 폴더 안으로 현재 디렉토리를 변경합니다. 


```python
%cd model_search
```

    /content/model_search


파이썬 패키지를 설치를 위해서 아래 명령을 통해 미리 정의한 패키지를 설치합니다.


```python
!pip install -r requirements.txt
```

    Requirement already satisfied: six==1.15.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 1)) (1.15.0)
    Requirement already satisfied: sklearn==0.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 2)) (0.0)
    Collecting tensorflow==2.2.0
    [?25l  Downloading https://files.pythonhosted.org/packages/4c/1a/0d79814736cfecc825ab8094b39648cc9c46af7af1bae839928acb73b4dd/tensorflow-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl (516.2MB)
    [K     |████████████████████████████████| 516.2MB 16kB/s 
    [?25hRequirement already satisfied: absl-py==0.10.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 4)) (0.10.0)
...
    Installing collected packages: tensorflow-estimator, tensorboard, tensorflow, tf-slim, ml-metadata, terminaltables, colorama, keras-tuner, mock
      Found existing installation: tensorflow-estimator 2.4.0
        Uninstalling tensorflow-estimator-2.4.0:
          Successfully uninstalled tensorflow-estimator-2.4.0
      Found existing installation: tensorboard 2.4.1
        Uninstalling tensorboard-2.4.1:
          Successfully uninstalled tensorboard-2.4.1
      Found existing installation: tensorflow 2.4.1
        Uninstalling tensorflow-2.4.1:
          Successfully uninstalled tensorflow-2.4.1
    Successfully installed colorama-0.4.4 keras-tuner-1.0.2 ml-metadata-0.26.0 mock-4.0.3 tensorboard-2.2.2 tensorflow-2.2.0 tensorflow-estimator-2.2.0 terminaltables-3.1.0 tf-slim-1.1.0


미리 정의한 구글 프로토콜 버퍼 파일(데이터를 저장하는 하나의 방식)을 파이썬 코드로 변환합니다.


```python
!protoc --python_out=./ model_search/proto/hparam.proto
!protoc --python_out=./ model_search/proto/phoenix_spec.proto
!protoc --python_out=./ model_search/proto/distillation_spec.proto
!protoc --python_out=./ model_search/proto/ensembling_spec.proto
!protoc --python_out=./ model_search/proto/transfer_learning_spec.proto
```

코랩 실행 시에 명령 인자 해석 오류가 일어나지 않도록 명령 인자 처리를 수행합니다.


```python
import sys
from absl import app

sys.argv = sys.argv[:1]

try:
  app.run(lambda argv: None)
except:
  pass
```

### 시작하기
---

블로그에 나와있는 간단한 예제를 시작하겠습니다. 주요 인자 위주로 살펴보면,

* number_models=200
    * 200개 모델 검색을 시도합니다.
* root_dir="/tmp/run_example"
    * 검색한 모델 평가 결과를 담고 있는 디렉토리를 지정합니다. 텐서보드에서 열 수 있으며, 평가메트릭과 함께 보실 수 있습니다.
* logits_dimension=2
    * 이진 분류 모델을 탐색합니다.
* filename="model_search/data/testdata/csv_random_data.csv"
    * 데이터셋 파일 경로를 지정합니다.
* spec="model_search/configs/dnn_config.pbtxt"
    * 모델 탐색을 위한 스펙이 정의된 파일을 지정합니다.


```python
import model_search
from model_search import constants
from model_search import single_trainer
from model_search.data import csv_data

trainer = single_trainer.SingleTrainer(
    data=csv_data.Provider(
        label_index=0,
        logits_dimension=2,
        record_defaults=[0, 0, 0, 0],
        filename="model_search/data/testdata/csv_random_data.csv"),
        spec="model_search/configs/dnn_config.pbtxt")

trainer.try_models(
    number_models=200,
    train_steps=1000,
    eval_steps=100,
    root_dir="/tmp/run_example",
    batch_size=32,
    experiment_name="example",
    experiment_owner="model_search_user")
```

    [0m
    I0225 06:13:21.385653 139685296322432 estimator.py:2066] Saving dict for global step 1111: accuracy = 1.0, auc_pr = 1.0, auc_roc = 0.9999998, global_step = 1111, loss = 0.0, num_parameters = 218394
    I0225 06:13:21.672063 139685296322432 estimator.py:2127] Saving 'checkpoint_path' summary for global step 1111: /tmp/run_example/tuner-1/130/model.ckpt-1111
    I0225 06:13:21.676702 139685296322432 phoenix.py:123] Saving the following evaluation dictionary.
    I0225 06:13:21.683635 139685296322432 phoenix.py:124] {'accuracy': 1.0, 'auc_pr': 1.0, 'auc_roc': 0.9999998211860657, 'loss': 0.0, 'num_parameters': 218394, 'global_step': 1111}
    I0225 06:13:21.688025 139685296322432 ml_metadata_db.py:156] Storing the following evaluation dictionary,
    I0225 06:13:21.689974 139685296322432 ml_metadata_db.py:157] {'accuracy': 1.0, 'auc_pr': 1.0, 'auc_roc': 0.9999998211860657, 'loss': 0.0, 'num_parameters': 218394, 'global_step': 1111}
    I0225 06:13:21.691386 139685296322432 ml_metadata_db.py:158] For the model in the following model dictionary,
    I0225 06:13:21.694412 139685296322432 ml_metadata_db.py:159] /tmp/run_example/tuner-1/130
    I0225 06:13:21.723419 139685296322432 oss_trainer_lib.py:256] Evaluation results: {'accuracy': 1.0, 'auc_pr': 1.0, 'auc_roc': 0.9999998, 'loss': 0.0, 'num_parameters': 218394, 'global_step': 1111}
    I0225 06:13:24.253669 139685296322432 oss_trainer_lib.py:281] creating directory: /tmp/run_example/tuner-1/131
    I0225 06:13:24.255254 139685296322432 oss_trainer_lib.py:328] Tuner id: tuner-1
    I0225 06:13:24.268173 139685296322432 oss_trainer_lib.py:329] Training with the following hyperparameters: 
    I0225 06:13:24.269536 139685296322432 oss_trainer_lib.py:330] {'learning_rate': 0.01, 'new_block_type': 'FIXED_OUTPUT_FULLY_CONNECTED_256', 'optimizer': 'momentum', 'initial_architecture_0': 'FIXED_OUTPUT_FULLY_CONNECTED_128', 'exponential_decay_rate': 0.75, 'exponential_decay_steps': 250, 'gradient_max_norm': 5, 'dropout_rate': 0.5500000216066837, 'initial_architecture': ['FIXED_OUTPUT_FULLY_CONNECTED_128']}
    I0225 06:13:24.280349 139685296322432 run_config.py:550] TF_CONFIG environment variable: {'model_dir': '/tmp/run_example/tuner-1/131', 'session_master': ''}
    I0225 06:13:24.281677 139685296322432 run_config.py:973] Using model_dir in TF_CONFIG: /tmp/run_example/tuner-1/131
    I0225 06:13:24.284329 139685296322432 estimator.py:191] Using config: {'_model_dir': '/tmp/run_example/tuner-1/131', '_tf_random_seed': None, '_save_summary_steps': 2000, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 120, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    ...
    I0225 06:33:46.598200 139685296322432 base_tower_generator.py:112] Building from existing checkpoint.
    I0225 06:33:47.663316 139685296322432 estimator.py:1171] Done calling model_fn.
    I0225 06:33:47.701185 139685296322432 evaluation.py:255] Starting evaluation at 2021-02-25T06:33:47Z
    I0225 06:33:47.878722 139685296322432 monitored_session.py:246] Graph was finalized.
    I0225 06:33:47.883833 139685296322432 saver.py:1293] Restoring parameters from /tmp/run_example/tuner-1/200/model.ckpt-1111
    I0225 06:33:48.062550 139685296322432 session_manager.py:505] Running local_init_op.
    I0225 06:33:48.119457 139685296322432 session_manager.py:508] Done running local_init_op.
    I0225 06:33:48.513041 139685296322432 evaluation.py:273] Inference Time : 0.81043s
    I0225 06:33:48.514187 139685296322432 evaluation.py:276] Finished evaluation at 2021-02-25-06:33:48
    I0225 06:33:48.516498 139685296322432 estimator.py:2066] Saving dict for global step 1111: accuracy = 1.0, auc_pr = 1.0, auc_roc = 0.9999998, global_step = 1111, loss = 0.0, num_parameters = 73783
    I0225 06:33:48.789884 139685296322432 estimator.py:2127] Saving 'checkpoint_path' summary for global step 1111: /tmp/run_example/tuner-1/200/model.ckpt-1111
    I0225 06:33:48.792447 139685296322432 phoenix.py:123] Saving the following evaluation dictionary.
    I0225 06:33:48.794008 139685296322432 phoenix.py:124] {'accuracy': 1.0, 'auc_pr': 1.0, 'auc_roc': 0.9999998211860657, 'loss': 0.0, 'num_parameters': 73783, 'global_step': 1111}
    I0225 06:33:48.795351 139685296322432 ml_metadata_db.py:156] Storing the following evaluation dictionary,
    I0225 06:33:48.796216 139685296322432 ml_metadata_db.py:157] {'accuracy': 1.0, 'auc_pr': 1.0, 'auc_roc': 0.9999998211860657, 'loss': 0.0, 'num_parameters': 73783, 'global_step': 1111}
    I0225 06:33:48.797077 139685296322432 ml_metadata_db.py:158] For the model in the following model dictionary,
    I0225 06:33:48.797994 139685296322432 ml_metadata_db.py:159] /tmp/run_example/tuner-1/200
    I0225 06:33:48.815671 139685296322432 oss_trainer_lib.py:256] Evaluation results: {'accuracy': 1.0, 'auc_pr': 1.0, 'auc_roc': 0.9999998, 'loss': 0.0, 'num_parameters': 73783, 'global_step': 1111}
    I0225 06:33:48.829422 139685296322432 oss_trainer_lib.py:281] creating directory: /tmp/run_example/tuner-1/201


### 마무리
---

구글 모델 서치를 코랩에서 구동하기 위한 환경설정 및 소스코드 예제를 살펴보았습니다. 다음에는 모델 서치 결과 확인 및 활용 방안에 대해서 알아보겠습니다.
