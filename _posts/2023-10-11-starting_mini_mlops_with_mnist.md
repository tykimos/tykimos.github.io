---
layout: post
title: "MNIST으로 미니 MLOps 시작하기"
author: Taeyoung Kim
date: 2023-10-11 00:00:00
categories: mlops, keras, mnist, mlops_mnist
comments: true
---

```python
!pip install gdown
```

    Requirement already satisfied: gdown in /usr/local/lib/python3.10/dist-packages (4.6.6)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from gdown) (3.12.4)
    Requirement already satisfied: requests[socks] in /usr/local/lib/python3.10/dist-packages (from gdown) (2.31.0)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from gdown) (1.16.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from gdown) (4.66.1)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from gdown) (4.11.2)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->gdown) (2.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (3.3.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (2.0.6)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (2023.7.22)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown) (1.7.1)



```python
# Google Drive 공유 링크에서 'file/d/' 이후와 '/view?usp=sharing' 사이의 ID를 복사합니다.

file_id = "1PJgXZHstmcByB73bzi4tz03DZhoX7U06"
url = f"https://drive.google.com/uc?id={file_id}"
```


```python
import os

# 다운로드 전의 파일 리스트를 가져옵니다.
before_files = set(os.listdir())

# gdown을 사용하여 파일을 다운로드합니다.
!gdown {url}

# 다운로드 후의 파일 리스트를 가져옵니다.
after_files = set(os.listdir())

# 다운로드된 파일명을 찾습니다.
downloaded_files = after_files - before_files
if downloaded_files:
    filename = downloaded_files.pop()
    print(f"Downloaded file: {filename}")
    downloaded_filepath = filename
else:
    print("No file downloaded.")
```

    Downloading...
    From: https://drive.google.com/uc?id=1PJgXZHstmcByB73bzi4tz03DZhoX7U06
    To: /content/mnist9.zip
    100% 11.5M/11.5M [00:00<00:00, 31.4MB/s]
    Downloaded file: mnist9.zip



```python
!unzip {downloaded_filepath}
```

    Archive:  mnist9.zip
      inflating: config.yaml             
      inflating: dataset.py              
      inflating: flow_validation.py      
       creating: mnist_data/
      inflating: mnist_data/y_train.npy  
      inflating: mnist_data/y_test.npy   
      inflating: mnist_data/x_test.npy   
      inflating: mnist_data/x_train.npy  
      inflating: model_inference.py      
      inflating: model_score.txt         
      inflating: random_inference.py     
      inflating: random_score.txt        
      inflating: score.py                
      inflating: train.py                



```python
# 이후에 원하는 Python 스크립트를 실행
!python flow_validation.py
```

    Deleted: random_score.txt
    Deleted: model_score.txt
    Files cleanup completed.
    Start 'python random_inference.py mnist_data/x_test.npy y_rand_pred.npy'
    Done 0.21 seconds
    Start 'python score.py mnist_data/y_test.npy y_rand_pred.npy random_score.txt'
    Done 0.19 seconds
    Start 'python train.py mnist_data/x_train.npy mnist_data/y_train.npy mnist_model.h5'
    2023-10-10 23:22:20.933384: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-10 23:22:22.046486: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    Epoch 1/5
    1875/1875 [==============================] - 5s 2ms/step - loss: 2.3962 - accuracy: 0.8572
    Epoch 2/5
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.3759 - accuracy: 0.9061
    Epoch 3/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.2956 - accuracy: 0.9261
    Epoch 4/5
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.2589 - accuracy: 0.9337
    Epoch 5/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.2292 - accuracy: 0.9412
    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(
    Done 31.71 seconds
    Start 'python model_inference.py mnist_model.h5 mnist_data/x_test.npy y_pred.npy'
    2023-10-10 23:22:52.618229: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-10 23:22:53.663297: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    313/313 [==============================] - 1s 1ms/step
    Done 4.82 seconds
    Start 'python score.py mnist_data/y_test.npy y_pred.npy model_score.txt'
    Done 0.18 seconds
    y_rand_pred_filepath check passed! (+10 points)
    random_score_filepath check passed! (+10 points)
    model_filepath check passed! (+10 points)
    y_pred_filepath check passed! (+10 points)
    model_score_filepath check passed! (+10 points)
    score_validation check passed! (+10 points)
    Total score: 100 / 100

## 코랩 파일 링크

해당 코랩 파일은 [링크](https://colab.research.google.com/drive/1WypSucaXUkcfqy2yp9L_8xuokM5ueT6r)에서 보실 수 있습니다.

