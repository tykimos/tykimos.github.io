---
layout: post
title:  "The latest Keras.io Code Examples Analysis 1 - Timeline"
author: Taeyoung Kim
date:   2022-2-25 00:00:00
categories: tech
comments: true
image: http://tykimos.github.io/warehouse/2022-2-25-The_latest_Keras_io_Code_Examples_Analysis_1_Timeline_title2.png
---

The Keras.io Code Example is frequently updated. To track the latest information, this code checks when the example was added or changed. 

![img](http://tykimos.github.io/warehouse/2022-2-25-The_latest_Keras_io_Code_Examples_Analysis_1_Timeline_title2.png)

Since this post includes Python source code and HTML rendering, I recommend you to see the Google Colab environment of this ipython notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tykimos/tykimos.github.io/blob/master/warehouse/2022-2-25-The_latest_Keras_io_Code_Examples_Analysis_1_Timeline.ipynb)

## Download Keras.io Code Example

This shell downloads all keras repository from github.


```python
# download keras-io source code

!wget --no-check-certificate https://github.com/keras-team/keras-io/archive/refs/heads/master.zip
```

    --2022-02-25 05:08:46--  https://github.com/keras-team/keras-io/archive/refs/heads/master.zip
    Resolving github.com (github.com)... 140.82.112.4
    Connecting to github.com (github.com)|140.82.112.4|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://codeload.github.com/keras-team/keras-io/zip/refs/heads/master [following]
    --2022-02-25 05:08:46--  https://codeload.github.com/keras-team/keras-io/zip/refs/heads/master
    Resolving codeload.github.com (codeload.github.com)... 140.82.121.10
    Connecting to codeload.github.com (codeload.github.com)|140.82.121.10|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [application/zip]
    Saving to: ‘master.zip.1’
    
    master.zip.1            [        <=>         ] 121.75M  22.7MB/s    in 5.7s    
    
    2022-02-25 05:08:53 (21.2 MB/s) - ‘master.zip.1’ saved [127669641]
    



```python
# unzip keras-io source code

import os
import zipfile

zf = zipfile.ZipFile('master.zip', 'r')
zf.extractall('keras-io')
examples_path = 'keras-io/keras-io-master/examples/'
```


```python
# ref: https://wikidocs.net/39

def search(dir_path, list_file_path, list_dir_path, dir_path_depth):

    try:
        filenames = os.listdir(dir_path)
        for filename in filenames:
            full_filename = os.path.join(dir_path, filename)
            if os.path.isdir(full_filename):
                search(full_filename, list_file_path, list_dir_path, dir_path_depth)
                if full_filename.count('/') == dir_path_depth:
                    list_dir_path.append(full_filename.split('/')[3])
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.py': 
                    list_file_path.append(full_filename)
    except PermissionError:
        pass

py_files = []
catagories_path = []
search(examples_path, py_files, catagories_path, 3)

print(catagories_path)
```
    ['rl', 'keras_recipes', 'graph', 'structured_data', 'vision', 'timeseries', 'nlp', 'audio', 'generative']


## Get meta information of each code example

This shell gets meta information from the first cell of each source code using parsing.

For example:

```
"""
Title: Image classification from scratch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/27
Last modified: 2020/04/28
Description: Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.
"""
```


```python
from collections import defaultdict
import pandas as pd

def get_meta(source_code_file):
    meta_dict = {}
    fp = open(source_code_file, 'rt')
    check = 0
    while True:
        line = fp.readline()
        if line == '"""\n':
            if check == 0:
                check = 1
                continue
            elif check == 1:
                break;
        if not line: 
            break
        tokens = line.split(':',1)
        meta_dict[tokens[0]] = tokens[1].lstrip().rstrip('\n')
    fp.close()
    return meta_dict

meta_dict_list = {'Filename' : [], 
                  'Category' : [], 
                  'Title' : [],
                  'Author' : [],
                  'Date created' : [],
                  'Last modified' : [],
                  'Description' : [],
                  'Authors' : []}

for py_file in py_files:
    meta_dict = get_meta(py_file)

    meta_dict_list['Filename'].append(py_file)
    meta_dict_list['Category'].append(py_file.split('/')[3])

    for key in meta_dict_list.keys():
        if key == 'Filename' or key == 'Category':
            continue
        try:  
            value = meta_dict[key]
            meta_dict_list[key].append(value)
        except KeyError:
            meta_dict_list[key].append('')
```

## Generate a dataframe from meta

This shell generates a dataframe from the meta dictionary.


```python
# genrate dataframe from meta_dict

meta_df = pd.DataFrame(meta_dict_list)

meta_df['Date created'] = pd.to_datetime(meta_df['Date created'])
meta_df['Last modified'] = pd.to_datetime(meta_df['Last modified'])
meta_df['Link'] = meta_df['Filename'].str.replace('keras-io/keras-io-master/examples/', 'https://keras.io/examples/').str.replace('.py', '')
meta_df['Title'] = "<a href=" + meta_df['Link'] + ">" + meta_df['Title'] + "</a>"

report_df = meta_df[['Category', 'Title', 'Date created', 'Last modified']]
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:7: FutureWarning: The default value of regex will change from True to False in a future version.
      import sys


## Generate a plot accumulated count of code example by field

This shell generates a plot accumulated count of code example by field.


```python
import IPython
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.colors as mcolors
import numpy as np

def plot_accumulated_count(title, dates, categories):

    # Get some auxilliary values
    min_date = date2num(dates[0])
    max_date = date2num(dates[-1])
    days = int(max_date - min_date + 1)

    categories_key = list(set(categories))
    categories_key.append('total')

    # Initialize X and Y axes
    x = np.arange(min_date, max_date + 1)
    y = np.zeros((len(categories_key), days))
    y_sum = np.zeros((len(categories_key), days))

    # Iterate over dates, increase registration array
    for i, date in enumerate(dates):
        date_index = int(date2num(date) - min_date)
        category_index = categories_key.index(categories[i])
        y[category_index, date_index] += 1
        y[-1, date_index] += 1 # total

    for i, value in enumerate(categories_key):
        y_sum[i] = np.cumsum(y[i])

    color_list = list(mcolors.TABLEAU_COLORS.items())

    # Plot
    plt.figure(dpi=150)

    for i, value in enumerate(categories_key):
        plt.plot_date(x, y_sum[i], xdate=True, ydate=False, ls='-', ms=-1, linewidth=0.5, color=color_list[i][1], label = value)    

    plt.fill_between(x, 0, y_sum[-1], facecolor='#D0F3FF')
    
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(title)
    plt.show()

# date plot
sorted_report_df = report_df.sort_values(by=['Date created'], ascending=True)
dates = pd.to_datetime(sorted_report_df['Date created']).tolist()
categories = sorted_report_df['Category'].tolist()
plot_accumulated_count('Keras Code Example', dates, categories)
```

![png](http://tykimos.github.io/warehouse/2022-2-25-The_latest_Keras_io_Code_Examples_Analysis_1_Timeline_10_0.png)
    


## Render a HTML from data frame sorted by date

This shell renders HTML from the data frame sorted by date.


```python
# sorted by 'Date created'

sorted_report_df = report_df.sort_values(by=['Date created'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```

| Category        | Title                                                                                                                                               | Date created        | Last modified       |
|:----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:--------------------|
| structured_data | [Classification with TensorFlow Decision Forests](https://keras.io/examples/structured_data/classification_with_tfdf)                               | 2022-01-25 | 2022-01-25 |
| vision          | [Augmenting convnets with aggregated attention](https://keras.io/examples/vision/patch_convnet)                                                     | 2022-01-22 | 2022-01-22 |
| structured_data | [Structured data learning with TabTransformer](https://keras.io/examples/structured_data/tabtransformer)                                            | 2022-01-18 | 2022-01-18 |
| nlp             | [Question Answering with Hugging Face Transformers](https://keras.io/examples/nlp/question_answering)                                               | 2022-01-13 | 2022-01-13 |
| vision          | [Video Vision Transformer](https://keras.io/examples/vision/vivit)                                                                                  | 2022-01-12 | 2022-01-12 |
| vision          | [Train a Vision Transformer on small datasets](https://keras.io/examples/vision/vit_small_ds)                                                       | 2022-01-07 | 2022-01-10 |
| timeseries      | [Traffic forecasting using graph neural networks and LSTM](https://keras.io/examples/timeseries/timeseries_traffic_forecasting)                     | 2021-12-28 | 2021-12-28 |
| generative      | [GauGAN for conditional image generation](https://keras.io/examples/generative/gaugan)                                                              | 2021-12-26 | 2022-01-03 |
| vision          | [Masked image modeling with Autoencoders](https://keras.io/examples/vision/masked_image_modeling)                                                   | 2021-12-20 | 2021-12-21 |
| vision          | [Learning to tokenize in Vision Transformers](https://keras.io/examples/vision/token_learner)                                                       | 2021-12-10 | 2021-12-15 |
| generative      | [Neural Style Transfer with AdaIN](https://keras.io/examples/generative/adain)                                                                      | 2021-11-08 | 2021-11-08 |
| vision          | [Barlow Twins for Contrastive SSL](https://keras.io/examples/vision/barlow_twins)                                                                   | 2021-11-04 | 2021-12-20 |
| keras_recipes   | [Customizing the convolution operation of a Conv2D layer](https://keras.io/examples/keras_recipes/subclassing_conv_layers)                          | 2021-11-03 | 2021-11-03 |
| nlp             | [Review Classification using Active Learning](https://keras.io/examples/nlp/active_learning_review_classification)                                  | 2021-10-29 | 2021-10-29 |
| generative      | [Data-efficient GANs with Adaptive Discriminator Augmentation](https://keras.io/examples/generative/gan_ada)                                        | 2021-10-28 | 2021-10-28 |
| vision          | [MobileViT: A mobile-friendly Transformer-based model for image classification](https://keras.io/examples/vision/mobilevit)                         | 2021-10-20 | 2021-10-20 |
| vision          | [Image classification with EANet (External Attention Transformer)](https://keras.io/examples/vision/eanet)                                          | 2021-10-19 | 2021-10-19 |
| vision          | [Image classification with ConvMixer](https://keras.io/examples/vision/convmixer)                                                                   | 2021-10-12 | 2021-10-12 |
| vision          | [FixRes: Fixing train-test resolution discrepancy](https://keras.io/examples/vision/fixres)                                                         | 2021-10-08 | 2021-10-10 |
| keras_recipes   | [Evaluating and exporting scikit-learn metrics in a Keras callback](https://keras.io/examples/keras_recipes/sklearn_metric_callbacks)               | 2021-10-07 | 2021-10-07 |
| nlp             | [Text Generation using FNet](https://keras.io/examples/nlp/text_generation_fnet)                                                                    | 2021-10-05 | 2021-10-05 |
| vision          | [Metric learning for image similarity search using TensorFlow Similarity](https://keras.io/examples/vision/metric_learning_tf_similarity)           | 2021-09-30 | 2021-09-30 |
| audio           | [Automatic Speech Recognition using CTC](https://keras.io/examples/audio/ctc_asr)                                                                   | 2021-09-26 | 2021-09-26 |
| vision          | [Image Classification using BigTransfer (BiT)](https://keras.io/examples/vision/bit)                                                                | 2021-09-24 | 2021-09-24 |
| vision          | [Zero-DCE for low-light image enhancement](https://keras.io/examples/vision/zero_dce)                                                               | 2021-09-18 | 2021-09-19 |
| graph           | [Graph attention network (GAT) for node classification](https://keras.io/examples/graph/gat_node_classification)                                    | 2021-09-13 | 2021-12-26 |
| vision          | [Self-supervised contrastive learning with NNCLR](https://keras.io/examples/vision/nnclr)                                                           | 2021-09-13 | 2021-09-13 |
| vision          | [Low-light image enhancement using MIRNet](https://keras.io/examples/vision/mirnet)                                                                 | 2021-09-11 | 2021-09-15 |
| vision          | [Near-duplicate image search](https://keras.io/examples/vision/near_dup_search)                                                                     | 2021-09-10 | 2021-09-10 |
| vision          | [Image classification with Swin Transformers](https://keras.io/examples/vision/swin_transformers)                                                   | 2021-09-08 | 2021-09-08 |
| vision          | [Multiclass semantic segmentation using DeepLabV3+](https://keras.io/examples/vision/deeplabv3_plus)                                                | 2021-08-31 | 2021-09-01 |
| vision          | [Monocular depth estimation](https://keras.io/examples/vision/depth_estimation)                                                                     | 2021-08-30 | 2021-08-30 |
| keras_recipes   | [Writing Keras Models With TensorFlow NumPy](https://keras.io/examples/keras_recipes/tensorflow_nu_models)                                          | 2021-08-28 | 2021-08-28 |
| vision          | [Handwriting recognition](https://keras.io/examples/vision/handwriting_recognition)                                                                 | 2021-08-16 | 2021-08-16 |
| vision          | [Classification using Attention-based Deep Multiple Instance Learning (MIL).](https://keras.io/examples/vision/attention_mil_classification)        | 2021-08-16 | 2021-11-25 |
| graph           | [Message-passing neural network (MPNN) for molecular property prediction](https://keras.io/examples/graph/mpnn-molecular-graphs)                    | 2021-08-16 | 2021-12-27 |
| vision          | [3D volumetric rendering with NeRF](https://keras.io/examples/vision/nerf)                                                                          | 2021-08-09 | 2021-08-09 |
| nlp             | [Multimodal entailment](https://keras.io/examples/nlp/multimodal_entailment)                                                                        | 2021-08-08 | 2021-08-15 |
| keras_recipes   | [Knowledge distillation recipes](https://keras.io/examples/keras_recipes/better_knowledge_distillation)                                             | 2021-08-01 | 2021-08-01 |
| vision          | [Involutional neural networks](https://keras.io/examples/vision/involution)                                                                         | 2021-07-25 | 2021-07-25 |
| generative      | [Vector-Quantized Variational Autoencoders](https://keras.io/examples/generative/vq_vae)                                                            | 2021-07-21 | 2021-07-21 |
| generative      | [Conditional GAN](https://keras.io/examples/generative/conditional_gan)                                                                             | 2021-07-13 | 2021-07-15 |
| generative      | [Face image generation with StyleGAN](https://keras.io/examples/generative/stylegan)                                                                | 2021-07-01 | 2021-07-01 |
| generative      | [WGAN-GP with R-GCN for the generation of small molecular graphs](https://keras.io/examples/generative/wgan-graphs)                                 | 2021-06-30 | 2021-06-30 |
| vision          | [Compact Convolutional Transformers](https://keras.io/examples/vision/cct)                                                                          | 2021-06-30 | 2021-06-30 |
| timeseries      | [Timeseries classification with a Transformer model](https://keras.io/examples/timeseries/timeseries_classification_transformer)                    | 2021-06-25 | 2021-08-05 |
| rl              | [Proximal Policy Optimization](https://keras.io/examples/rl/ppo_cartpole)                                                                           | 2021-06-24 | 2021-06-24 |
| nlp             | [Named Entity Recognition using Transformers](https://keras.io/examples/nlp/ner_transformers)                                                       | 2021-06-23 | 2021-06-24 |
| vision          | [Semi-supervision and domain adaptation with AdaMatch](https://keras.io/examples/vision/adamatch)                                                   | 2021-06-19 | 2021-06-19 |
| vision          | [Gradient Centralization for Better Training Performance](https://keras.io/examples/vision/gradient_centralization)                                 | 2021-06-18 | 2021-06-18 |
| vision          | [Video Classification with Transformers](https://keras.io/examples/vision/video_transformers)                                                       | 2021-06-08 | 2021-06-08 |
| vision          | [CutMix data augmentation for image classification](https://keras.io/examples/vision/cutmix)                                                        | 2021-06-08 | 2021-06-08 |
| vision          | [Next-Frame Video Prediction with Convolutional LSTMs](https://keras.io/examples/vision/conv_lstm)                                                  | 2021-06-02 | 2021-06-05 |
| vision          | [Image classification with modern MLP models](https://keras.io/examples/vision/mlp_image_classification)                                            | 2021-05-30 | 2021-05-30 |
| graph           | [Node Classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations)                                                     | 2021-05-30 | 2021-05-30 |
| vision          | [Image Captioning](https://keras.io/examples/vision/image_captioning)                                                                               | 2021-05-29 | 2021-10-31 |
| vision          | [Video Classification with a CNN-RNN Architecture](https://keras.io/examples/vision/video_classification)                                           | 2021-05-28 | 2021-06-05 |
| nlp             | [English-to-Spanish translation with a sequence-to-sequence Transformer](https://keras.io/examples/nlp/neural_machine_translation_with_transformer) | 2021-05-26 | 2021-05-26 |
| keras_recipes   | [Estimating required sample size for model training](https://keras.io/examples/keras_recipes/sample_size_estimate)                                  | 2021-05-20 | 2021-06-06 |
| graph           | [Graph representation learning with node2vec](https://keras.io/examples/graph/node2vec_movielens)                                                   | 2021-05-15 | 2021-05-15 |
| vision          | [Image similarity estimation using a Siamese Network with a contrastive loss](https://keras.io/examples/vision/siamese_contrastive)                 | 2021-05-06 | 2021-05-06 |
| vision          | [Keypoint Detection with Transfer Learning](https://keras.io/examples/vision/keypoint_detection)                                                    | 2021-05-02 | 2021-05-02 |
| vision          | [Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer)                                                         | 2021-04-30 | 2021-05-13 |
| vision          | [Image classification with Perceiver](https://keras.io/examples/vision/perceiver_image_classification)                                              | 2021-04-30 | 2021-01-30 |
| vision          | [Semi-supervised image classification using contrastive pretraining with SimCLR](https://keras.io/examples/vision/semisupervised_simclr)            | 2021-04-24 | 2021-04-24 |
| vision          | [Consistency training with supervision](https://keras.io/examples/vision/consistency_training)                                                      | 2021-04-13 | 2021-04-19 |
| vision          | [Image similarity estimation using a Siamese Network with a triplet loss](https://keras.io/examples/vision/siamese_network)                         | 2021-03-25 | 2021-03-25 |
| vision          | [Self-supervised contrastive learning with SimSiam](https://keras.io/examples/vision/simsiam)                                                       | 2021-03-19 | 2021-03-20 |
| vision          | [RandAugment for Image Classification for Improved Robustness](https://keras.io/examples/vision/randaugment)                                        | 2021-03-13 | 2021-03-17 |
| vision          | [MixUp augmentation for image classification](https://keras.io/examples/vision/mixup)                                                               | 2021-03-06 | 2021-03-06 |
| vision          | [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder)                                                       | 2021-03-01 | 2021-03-01 |
| vision          | [Semantic Image Clustering](https://keras.io/examples/vision/semantic_image_clustering)                                                             | 2021-02-28 | 2021-02-28 |
| keras_recipes   | [Creating TFRecords](https://keras.io/examples/keras_recipes/creating_tfrecords)                                                                    | 2021-02-27 | 2021-02-27 |
| keras_recipes   | [Memory-efficient embeddings for recommendation systems](https://keras.io/examples/keras_recipes/memory_efficient_embeddings)                       | 2021-02-15 | 2021-02-15 |
| structured_data | [Classification with Gated Residual and Variable Selection Networks](https://keras.io/examples/structured_data/classification_with_grn_and_vsn)     | 2021-02-10 | 2021-02-10 |
| audio           | [MelGAN-based spectrogram inversion using feature matching](https://keras.io/examples/audio/melgan_spectrogram_inversion)                           | 2021-02-09 | 2021-09-15 |
| nlp             | [Natural language image search with a Dual Encoder](https://keras.io/examples/nlp/nl_image_search)                                                  | 2021-01-30 | 2021-01-30 |
| vision          | [Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer)                       | 2021-01-18 | 2021-01-18 |
| structured_data | [Classification with Neural Decision Forests](https://keras.io/examples/structured_data/deep_neural_decision_forests)                               | 2021-01-15 | 2021-01-15 |
| keras_recipes   | [Probabilistic Bayesian Neural Networks](https://keras.io/examples/keras_recipes/bayesian_neural_networks)                                          | 2021-01-15 | 2021-01-15 |
| audio           | [Automatic Speech Recognition with Transformer](https://keras.io/examples/audio/transformer_asr)                                                    | 2021-01-13 | 2021-01-13 |
| structured_data | [Structured data learning with Wide, Deep, and Cross networks](https://keras.io/examples/structured_data/wide_deep_cross_networks)                  | 2020-12-31 | 2021-05-05 |
| structured_data | [A Transformer-based recommendation system](https://keras.io/examples/structured_data/movielens_recommendations_transformers)                       | 2020-12-30 | 2020-12-30 |
| vision          | [Supervised Contrastive Learning](https://keras.io/examples/vision/supervised-contrastive-learning)                                                 | 2020-11-30 | 2020-11-30 |
| vision          | [Point cloud segmentation with PointNet](https://keras.io/examples/vision/pointnet_segmentation)                                                    | 2020-10-23 | 2020-10-24 |
| nlp             | [Large-scale multi-label text classification](https://keras.io/examples/nlp/multi_label_classification)                                             | 2020-09-25 | 2020-12-23 |
| vision          | [3D image classification from CT scans](https://keras.io/examples/vision/3D_image_classification)                                                   | 2020-09-23 | 2020-09-23 |
| nlp             | [End-to-end Masked Language Modeling with BERT](https://keras.io/examples/nlp/masked_language_modeling)                                             | 2020-09-18 | 2020-09-18 |
| vision          | [Knowledge Distillation](https://keras.io/examples/vision/knowledge_distillation)                                                                   | 2020-09-01 | 2020-09-01 |
| nlp             | [Semantic Similarity with BERT](https://keras.io/examples/nlp/semantic_similarity_with_bert)                                                        | 2020-08-15 | 2020-08-29 |
| generative      | [CycleGAN](https://keras.io/examples/generative/cyclegan)                                                                                           | 2020-08-12 | 2020-08-12 |
| generative      | [Density estimation using Real NVP](https://keras.io/examples/generative/real_nvp)                                                                  | 2020-08-10 | 2020-08-10 |
| keras_recipes   | [How to train a Keras model on TFRecord files](https://keras.io/examples/keras_recipes/tfrecord)                                                    | 2020-07-29 | 2020-08-07 |
| vision          | [Pneumonia Classification on TPU](https://keras.io/examples/vision/xray_classification_with_tpus)                                                   | 2020-07-28 | 2020-08-24 |
| vision          | [Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel)                              | 2020-07-28 | 2020-08-27 |
| timeseries      | [Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)                               | 2020-07-21 | 2021-07-16 |
| vision          | [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning)            | 2020-06-30 | 2020-07-16 |
| timeseries      | [Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting)                                | 2020-06-23 | 2020-07-20 |
| vision          | [OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr)                                                                      | 2020-06-14 | 2020-06-26 |
| audio           | [Speaker Recognition](https://keras.io/examples/audio/speaker_recognition_using_cnn)                                                                | 2020-06-14 | 2020-03-07 |
| structured_data | [Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch)                | 2020-06-09 | 2020-06-09 |
| vision          | [Metric learning for image similarity search](https://keras.io/examples/vision/metric_learning)                                                     | 2020-06-05 | 2020-06-09 |
| rl              | [Deep Deterministic Policy Gradient (DDPG)](https://keras.io/examples/rl/ddpg_pendulum)                                                             | 2020-06-04 | 2020-09-21 |
| vision          | [Model interpretability with Integrated Gradients](https://keras.io/examples/vision/integrated_gradients)                                           | 2020-06-02 | 2020-06-02 |
| timeseries      | [Timeseries anomaly detection using an Autoencoder](https://keras.io/examples/timeseries/timeseries_anomaly_detection)                              | 2020-05-31 | 2020-05-31 |
| generative      | [Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt)                                     | 2020-05-29 | 2020-05-29 |
| vision          | [Visualizing what convnets learn](https://keras.io/examples/vision/visualizing_what_convnets_learn)                                                 | 2020-05-29 | 2020-05-29 |
| vision          | [Point cloud classification with PointNet](https://keras.io/examples/vision/pointnet)                                                               | 2020-05-25 | 2020-05-26 |
| structured_data | [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens)                    | 2020-05-24 | 2020-05-24 |
| rl              | [Deep Q-Learning for Atari Breakout](https://keras.io/examples/rl/deep_q_network_breakout)                                                          | 2020-05-23 | 2020-06-17 |
| nlp             | [Text Extraction with BERT](https://keras.io/examples/nlp/text_extraction_with_bert)                                                                | 2020-05-23 | 2020-05-23 |
| vision          | [Few-Shot learning with Reptile](https://keras.io/examples/vision/reptile)                                                                          | 2020-05-21 | 2020-05-30 |
| vision          | [Object Detection with RetinaNet](https://keras.io/examples/vision/retinanet)                                                                       | 2020-05-17 | 2020-07-14 |
| generative      | [PixelCNN](https://keras.io/examples/generative/pixelcnn)                                                                                           | 2020-05-17 | 2020-05-23 |
| keras_recipes   | [Keras debugging tips](https://keras.io/examples/keras_recipes/debugging_tips)                                                                      | 2020-05-16 | 2020-05-16 |
| rl              | [Actor Critic Method](https://keras.io/examples/rl/actor_critic_cartpole)                                                                           | 2020-05-13 | 2020-05-13 |
| nlp             | [Text classification with Switch Transformer](https://keras.io/examples/nlp/text_classification_with_switch_transformer)                            | 2020-05-10 | 2021-02-15 |
| nlp             | [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer)                                          | 2020-05-10 | 2020-05-10 |
| generative      | [WGAN-GP overriding `Model.train_step`](https://keras.io/examples/generative/wgan_gp)                                                               | 2020-05-09 | 2020-05-09 |
| nlp             | [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings)                                                       | 2020-05-05 | 2020-05-05 |
| generative      | [Variational AutoEncoder](https://keras.io/examples/generative/vae)                                                                                 | 2020-05-03 | 2020-05-03 |
| nlp             | [Bidirectional LSTM on IMDB](https://keras.io/examples/nlp/bidirectional_lstm_imdb)                                                                 | 2020-05-03 | 2020-05-03 |
| vision          | [Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch)                                             | 2020-04-27 | 2020-04-28 |
| vision          | [Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam)                                                                | 2020-04-26 | 2021-03-07 |
| keras_recipes   | [A Quasi-SVM in Keras](https://keras.io/examples/keras_recipes/quasi_svm)                                                                           | 2020-04-17 | 2020-04-17 |
| nlp             | [Text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch)                                                  | 2019-11-06 | 2020-05-17 |
| structured_data | [Imbalanced classification: credit card fraud detection](https://keras.io/examples/structured_data/imbalanced_classification)                       | 2019-05-28 | 2020-04-17 |
| keras_recipes   | [Endpoint layer pattern](https://keras.io/examples/keras_recipes/endpoint_layer_pattern)                                                            | 2019-05-10 | 2019-05-10 |
| generative      | [DCGAN to generate face images](https://keras.io/examples/generative/dcgan_overriding_train_step)                                                   | 2019-04-29 | 2021-01-01 |
| vision          | [Image segmentation with a U-Net-like architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation)                                | 2019-03-20 | 2020-04-20 |
| nlp             | [Character-level recurrent sequence-to-sequence model](https://keras.io/examples/nlp/lstm_seq2seq)                                                  | 2017-09-29 | 2020-04-26 |
| generative      | [Deep Dream](https://keras.io/examples/generative/deep_dream)                                                                                       | 2016-01-13 | 2020-05-02 |
| generative      | [Neural style transfer](https://keras.io/examples/generative/neural_style_transfer)                                                                 | 2016-01-11 | 2020-05-02 |
| keras_recipes   | [Simple custom layer example: Antirectifier](https://keras.io/examples/keras_recipes/antirectifier)                                                 | 2016-01-06 | 2020-04-20 |
| nlp             | [Sequence to sequence learning for performing number addition](https://keras.io/examples/nlp/addition_rnn)                                          | 2015-08-17 | 2020-04-17 |
| vision          | [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet)                                                                              | 2015-06-19 | 2020-04-21 |
| generative      | [Character-level text generation with LSTM](https://keras.io/examples/generative/lstm_character_level_text_generation)                              | 2015-06-15 | 2020-04-30 |



```python
# sorted by 'Last modified'

sorted_report_df = report_df.sort_values(by=['Last modified'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```


| Category        | Title                                                                                                                                               | Date created        | Last modified       |
|:----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:--------------------|
| structured_data | [Classification with TensorFlow Decision Forests](https://keras.io/examples/structured_data/classification_with_tfdf)                               | 2022-01-25 | 2022-01-25 |
| vision          | [Augmenting convnets with aggregated attention](https://keras.io/examples/vision/patch_convnet)                                                     | 2022-01-22 | 2022-01-22 |
| structured_data | [Structured data learning with TabTransformer](https://keras.io/examples/structured_data/tabtransformer)                                            | 2022-01-18 | 2022-01-18 |
| nlp             | [Question Answering with Hugging Face Transformers](https://keras.io/examples/nlp/question_answering)                                               | 2022-01-13 | 2022-01-13 |
| vision          | [Video Vision Transformer](https://keras.io/examples/vision/vivit)                                                                                  | 2022-01-12 | 2022-01-12 |
| vision          | [Train a Vision Transformer on small datasets](https://keras.io/examples/vision/vit_small_ds)                                                       | 2022-01-07 | 2022-01-10 |
| generative      | [GauGAN for conditional image generation](https://keras.io/examples/generative/gaugan)                                                              | 2021-12-26 | 2022-01-03 |
| timeseries      | [Traffic forecasting using graph neural networks and LSTM](https://keras.io/examples/timeseries/timeseries_traffic_forecasting)                     | 2021-12-28 | 2021-12-28 |
| graph           | [Message-passing neural network (MPNN) for molecular property prediction](https://keras.io/examples/graph/mpnn-molecular-graphs)                    | 2021-08-16 | 2021-12-27 |
| graph           | [Graph attention network (GAT) for node classification](https://keras.io/examples/graph/gat_node_classification)                                    | 2021-09-13 | 2021-12-26 |
| vision          | [Masked image modeling with Autoencoders](https://keras.io/examples/vision/masked_image_modeling)                                                   | 2021-12-20 | 2021-12-21 |
| vision          | [Barlow Twins for Contrastive SSL](https://keras.io/examples/vision/barlow_twins)                                                                   | 2021-11-04 | 2021-12-20 |
| vision          | [Learning to tokenize in Vision Transformers](https://keras.io/examples/vision/token_learner)                                                       | 2021-12-10 | 2021-12-15 |
| vision          | [Classification using Attention-based Deep Multiple Instance Learning (MIL).](https://keras.io/examples/vision/attention_mil_classification)        | 2021-08-16 | 2021-11-25 |
| generative      | [Neural Style Transfer with AdaIN](https://keras.io/examples/generative/adain)                                                                      | 2021-11-08 | 2021-11-08 |
| keras_recipes   | [Customizing the convolution operation of a Conv2D layer](https://keras.io/examples/keras_recipes/subclassing_conv_layers)                          | 2021-11-03 | 2021-11-03 |
| vision          | [Image Captioning](https://keras.io/examples/vision/image_captioning)                                                                               | 2021-05-29 | 2021-10-31 |
| nlp             | [Review Classification using Active Learning](https://keras.io/examples/nlp/active_learning_review_classification)                                  | 2021-10-29 | 2021-10-29 |
| generative      | [Data-efficient GANs with Adaptive Discriminator Augmentation](https://keras.io/examples/generative/gan_ada)                                        | 2021-10-28 | 2021-10-28 |
| vision          | [MobileViT: A mobile-friendly Transformer-based model for image classification](https://keras.io/examples/vision/mobilevit)                         | 2021-10-20 | 2021-10-20 |
| vision          | [Image classification with EANet (External Attention Transformer)](https://keras.io/examples/vision/eanet)                                          | 2021-10-19 | 2021-10-19 |
| vision          | [Image classification with ConvMixer](https://keras.io/examples/vision/convmixer)                                                                   | 2021-10-12 | 2021-10-12 |
| vision          | [FixRes: Fixing train-test resolution discrepancy](https://keras.io/examples/vision/fixres)                                                         | 2021-10-08 | 2021-10-10 |
| keras_recipes   | [Evaluating and exporting scikit-learn metrics in a Keras callback](https://keras.io/examples/keras_recipes/sklearn_metric_callbacks)               | 2021-10-07 | 2021-10-07 |
| nlp             | [Text Generation using FNet](https://keras.io/examples/nlp/text_generation_fnet)                                                                    | 2021-10-05 | 2021-10-05 |
| vision          | [Metric learning for image similarity search using TensorFlow Similarity](https://keras.io/examples/vision/metric_learning_tf_similarity)           | 2021-09-30 | 2021-09-30 |
| audio           | [Automatic Speech Recognition using CTC](https://keras.io/examples/audio/ctc_asr)                                                                   | 2021-09-26 | 2021-09-26 |
| vision          | [Image Classification using BigTransfer (BiT)](https://keras.io/examples/vision/bit)                                                                | 2021-09-24 | 2021-09-24 |
| vision          | [Zero-DCE for low-light image enhancement](https://keras.io/examples/vision/zero_dce)                                                               | 2021-09-18 | 2021-09-19 |
| audio           | [MelGAN-based spectrogram inversion using feature matching](https://keras.io/examples/audio/melgan_spectrogram_inversion)                           | 2021-02-09 | 2021-09-15 |
| vision          | [Low-light image enhancement using MIRNet](https://keras.io/examples/vision/mirnet)                                                                 | 2021-09-11 | 2021-09-15 |
| vision          | [Self-supervised contrastive learning with NNCLR](https://keras.io/examples/vision/nnclr)                                                           | 2021-09-13 | 2021-09-13 |
| vision          | [Near-duplicate image search](https://keras.io/examples/vision/near_dup_search)                                                                     | 2021-09-10 | 2021-09-10 |
| vision          | [Image classification with Swin Transformers](https://keras.io/examples/vision/swin_transformers)                                                   | 2021-09-08 | 2021-09-08 |
| vision          | [Multiclass semantic segmentation using DeepLabV3+](https://keras.io/examples/vision/deeplabv3_plus)                                                | 2021-08-31 | 2021-09-01 |
| vision          | [Monocular depth estimation](https://keras.io/examples/vision/depth_estimation)                                                                     | 2021-08-30 | 2021-08-30 |
| keras_recipes   | [Writing Keras Models With TensorFlow NumPy](https://keras.io/examples/keras_recipes/tensorflow_nu_models)                                          | 2021-08-28 | 2021-08-28 |
| vision          | [Handwriting recognition](https://keras.io/examples/vision/handwriting_recognition)                                                                 | 2021-08-16 | 2021-08-16 |
| nlp             | [Multimodal entailment](https://keras.io/examples/nlp/multimodal_entailment)                                                                        | 2021-08-08 | 2021-08-15 |
| vision          | [3D volumetric rendering with NeRF](https://keras.io/examples/vision/nerf)                                                                          | 2021-08-09 | 2021-08-09 |
| timeseries      | [Timeseries classification with a Transformer model](https://keras.io/examples/timeseries/timeseries_classification_transformer)                    | 2021-06-25 | 2021-08-05 |
| keras_recipes   | [Knowledge distillation recipes](https://keras.io/examples/keras_recipes/better_knowledge_distillation)                                             | 2021-08-01 | 2021-08-01 |
| vision          | [Involutional neural networks](https://keras.io/examples/vision/involution)                                                                         | 2021-07-25 | 2021-07-25 |
| generative      | [Vector-Quantized Variational Autoencoders](https://keras.io/examples/generative/vq_vae)                                                            | 2021-07-21 | 2021-07-21 |
| timeseries      | [Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)                               | 2020-07-21 | 2021-07-16 |
| generative      | [Conditional GAN](https://keras.io/examples/generative/conditional_gan)                                                                             | 2021-07-13 | 2021-07-15 |
| generative      | [Face image generation with StyleGAN](https://keras.io/examples/generative/stylegan)                                                                | 2021-07-01 | 2021-07-01 |
| generative      | [WGAN-GP with R-GCN for the generation of small molecular graphs](https://keras.io/examples/generative/wgan-graphs)                                 | 2021-06-30 | 2021-06-30 |
| vision          | [Compact Convolutional Transformers](https://keras.io/examples/vision/cct)                                                                          | 2021-06-30 | 2021-06-30 |
| rl              | [Proximal Policy Optimization](https://keras.io/examples/rl/ppo_cartpole)                                                                           | 2021-06-24 | 2021-06-24 |
| nlp             | [Named Entity Recognition using Transformers](https://keras.io/examples/nlp/ner_transformers)                                                       | 2021-06-23 | 2021-06-24 |
| vision          | [Semi-supervision and domain adaptation with AdaMatch](https://keras.io/examples/vision/adamatch)                                                   | 2021-06-19 | 2021-06-19 |
| vision          | [Gradient Centralization for Better Training Performance](https://keras.io/examples/vision/gradient_centralization)                                 | 2021-06-18 | 2021-06-18 |
| vision          | [CutMix data augmentation for image classification](https://keras.io/examples/vision/cutmix)                                                        | 2021-06-08 | 2021-06-08 |
| vision          | [Video Classification with Transformers](https://keras.io/examples/vision/video_transformers)                                                       | 2021-06-08 | 2021-06-08 |
| keras_recipes   | [Estimating required sample size for model training](https://keras.io/examples/keras_recipes/sample_size_estimate)                                  | 2021-05-20 | 2021-06-06 |
| vision          | [Video Classification with a CNN-RNN Architecture](https://keras.io/examples/vision/video_classification)                                           | 2021-05-28 | 2021-06-05 |
| vision          | [Next-Frame Video Prediction with Convolutional LSTMs](https://keras.io/examples/vision/conv_lstm)                                                  | 2021-06-02 | 2021-06-05 |
| graph           | [Node Classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations)                                                     | 2021-05-30 | 2021-05-30 |
| vision          | [Image classification with modern MLP models](https://keras.io/examples/vision/mlp_image_classification)                                            | 2021-05-30 | 2021-05-30 |
| nlp             | [English-to-Spanish translation with a sequence-to-sequence Transformer](https://keras.io/examples/nlp/neural_machine_translation_with_transformer) | 2021-05-26 | 2021-05-26 |
| graph           | [Graph representation learning with node2vec](https://keras.io/examples/graph/node2vec_movielens)                                                   | 2021-05-15 | 2021-05-15 |
| vision          | [Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer)                                                         | 2021-04-30 | 2021-05-13 |
| vision          | [Image similarity estimation using a Siamese Network with a contrastive loss](https://keras.io/examples/vision/siamese_contrastive)                 | 2021-05-06 | 2021-05-06 |
| structured_data | [Structured data learning with Wide, Deep, and Cross networks](https://keras.io/examples/structured_data/wide_deep_cross_networks)                  | 2020-12-31 | 2021-05-05 |
| vision          | [Keypoint Detection with Transfer Learning](https://keras.io/examples/vision/keypoint_detection)                                                    | 2021-05-02 | 2021-05-02 |
| vision          | [Semi-supervised image classification using contrastive pretraining with SimCLR](https://keras.io/examples/vision/semisupervised_simclr)            | 2021-04-24 | 2021-04-24 |
| vision          | [Consistency training with supervision](https://keras.io/examples/vision/consistency_training)                                                      | 2021-04-13 | 2021-04-19 |
| vision          | [Image similarity estimation using a Siamese Network with a triplet loss](https://keras.io/examples/vision/siamese_network)                         | 2021-03-25 | 2021-03-25 |
| vision          | [Self-supervised contrastive learning with SimSiam](https://keras.io/examples/vision/simsiam)                                                       | 2021-03-19 | 2021-03-20 |
| vision          | [RandAugment for Image Classification for Improved Robustness](https://keras.io/examples/vision/randaugment)                                        | 2021-03-13 | 2021-03-17 |
| vision          | [Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam)                                                                | 2020-04-26 | 2021-03-07 |
| vision          | [MixUp augmentation for image classification](https://keras.io/examples/vision/mixup)                                                               | 2021-03-06 | 2021-03-06 |
| vision          | [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder)                                                       | 2021-03-01 | 2021-03-01 |
| vision          | [Semantic Image Clustering](https://keras.io/examples/vision/semantic_image_clustering)                                                             | 2021-02-28 | 2021-02-28 |
| keras_recipes   | [Creating TFRecords](https://keras.io/examples/keras_recipes/creating_tfrecords)                                                                    | 2021-02-27 | 2021-02-27 |
| keras_recipes   | [Memory-efficient embeddings for recommendation systems](https://keras.io/examples/keras_recipes/memory_efficient_embeddings)                       | 2021-02-15 | 2021-02-15 |
| nlp             | [Text classification with Switch Transformer](https://keras.io/examples/nlp/text_classification_with_switch_transformer)                            | 2020-05-10 | 2021-02-15 |
| structured_data | [Classification with Gated Residual and Variable Selection Networks](https://keras.io/examples/structured_data/classification_with_grn_and_vsn)     | 2021-02-10 | 2021-02-10 |
| nlp             | [Natural language image search with a Dual Encoder](https://keras.io/examples/nlp/nl_image_search)                                                  | 2021-01-30 | 2021-01-30 |
| vision          | [Image classification with Perceiver](https://keras.io/examples/vision/perceiver_image_classification)                                              | 2021-04-30 | 2021-01-30 |
| vision          | [Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer)                       | 2021-01-18 | 2021-01-18 |
| keras_recipes   | [Probabilistic Bayesian Neural Networks](https://keras.io/examples/keras_recipes/bayesian_neural_networks)                                          | 2021-01-15 | 2021-01-15 |
| structured_data | [Classification with Neural Decision Forests](https://keras.io/examples/structured_data/deep_neural_decision_forests)                               | 2021-01-15 | 2021-01-15 |
| audio           | [Automatic Speech Recognition with Transformer](https://keras.io/examples/audio/transformer_asr)                                                    | 2021-01-13 | 2021-01-13 |
| generative      | [DCGAN to generate face images](https://keras.io/examples/generative/dcgan_overriding_train_step)                                                   | 2019-04-29 | 2021-01-01 |
| structured_data | [A Transformer-based recommendation system](https://keras.io/examples/structured_data/movielens_recommendations_transformers)                       | 2020-12-30 | 2020-12-30 |
| nlp             | [Large-scale multi-label text classification](https://keras.io/examples/nlp/multi_label_classification)                                             | 2020-09-25 | 2020-12-23 |
| vision          | [Supervised Contrastive Learning](https://keras.io/examples/vision/supervised-contrastive-learning)                                                 | 2020-11-30 | 2020-11-30 |
| vision          | [Point cloud segmentation with PointNet](https://keras.io/examples/vision/pointnet_segmentation)                                                    | 2020-10-23 | 2020-10-24 |
| vision          | [3D image classification from CT scans](https://keras.io/examples/vision/3D_image_classification)                                                   | 2020-09-23 | 2020-09-23 |
| rl              | [Deep Deterministic Policy Gradient (DDPG)](https://keras.io/examples/rl/ddpg_pendulum)                                                             | 2020-06-04 | 2020-09-21 |
| nlp             | [End-to-end Masked Language Modeling with BERT](https://keras.io/examples/nlp/masked_language_modeling)                                             | 2020-09-18 | 2020-09-18 |
| vision          | [Knowledge Distillation](https://keras.io/examples/vision/knowledge_distillation)                                                                   | 2020-09-01 | 2020-09-01 |
| nlp             | [Semantic Similarity with BERT](https://keras.io/examples/nlp/semantic_similarity_with_bert)                                                        | 2020-08-15 | 2020-08-29 |
| vision          | [Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel)                              | 2020-07-28 | 2020-08-27 |
| vision          | [Pneumonia Classification on TPU](https://keras.io/examples/vision/xray_classification_with_tpus)                                                   | 2020-07-28 | 2020-08-24 |
| generative      | [CycleGAN](https://keras.io/examples/generative/cyclegan)                                                                                           | 2020-08-12 | 2020-08-12 |
| generative      | [Density estimation using Real NVP](https://keras.io/examples/generative/real_nvp)                                                                  | 2020-08-10 | 2020-08-10 |
| keras_recipes   | [How to train a Keras model on TFRecord files](https://keras.io/examples/keras_recipes/tfrecord)                                                    | 2020-07-29 | 2020-08-07 |
| timeseries      | [Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting)                                | 2020-06-23 | 2020-07-20 |
| vision          | [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning)            | 2020-06-30 | 2020-07-16 |
| vision          | [Object Detection with RetinaNet](https://keras.io/examples/vision/retinanet)                                                                       | 2020-05-17 | 2020-07-14 |
| vision          | [OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr)                                                                      | 2020-06-14 | 2020-06-26 |
| rl              | [Deep Q-Learning for Atari Breakout](https://keras.io/examples/rl/deep_q_network_breakout)                                                          | 2020-05-23 | 2020-06-17 |
| vision          | [Metric learning for image similarity search](https://keras.io/examples/vision/metric_learning)                                                     | 2020-06-05 | 2020-06-09 |
| structured_data | [Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch)                | 2020-06-09 | 2020-06-09 |
| vision          | [Model interpretability with Integrated Gradients](https://keras.io/examples/vision/integrated_gradients)                                           | 2020-06-02 | 2020-06-02 |
| timeseries      | [Timeseries anomaly detection using an Autoencoder](https://keras.io/examples/timeseries/timeseries_anomaly_detection)                              | 2020-05-31 | 2020-05-31 |
| vision          | [Few-Shot learning with Reptile](https://keras.io/examples/vision/reptile)                                                                          | 2020-05-21 | 2020-05-30 |
| vision          | [Visualizing what convnets learn](https://keras.io/examples/vision/visualizing_what_convnets_learn)                                                 | 2020-05-29 | 2020-05-29 |
| generative      | [Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt)                                     | 2020-05-29 | 2020-05-29 |
| vision          | [Point cloud classification with PointNet](https://keras.io/examples/vision/pointnet)                                                               | 2020-05-25 | 2020-05-26 |
| structured_data | [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens)                    | 2020-05-24 | 2020-05-24 |
| generative      | [PixelCNN](https://keras.io/examples/generative/pixelcnn)                                                                                           | 2020-05-17 | 2020-05-23 |
| nlp             | [Text Extraction with BERT](https://keras.io/examples/nlp/text_extraction_with_bert)                                                                | 2020-05-23 | 2020-05-23 |
| nlp             | [Text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch)                                                  | 2019-11-06 | 2020-05-17 |
| keras_recipes   | [Keras debugging tips](https://keras.io/examples/keras_recipes/debugging_tips)                                                                      | 2020-05-16 | 2020-05-16 |
| rl              | [Actor Critic Method](https://keras.io/examples/rl/actor_critic_cartpole)                                                                           | 2020-05-13 | 2020-05-13 |
| nlp             | [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer)                                          | 2020-05-10 | 2020-05-10 |
| generative      | [WGAN-GP overriding `Model.train_step`](https://keras.io/examples/generative/wgan_gp)                                                               | 2020-05-09 | 2020-05-09 |
| nlp             | [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings)                                                       | 2020-05-05 | 2020-05-05 |
| nlp             | [Bidirectional LSTM on IMDB](https://keras.io/examples/nlp/bidirectional_lstm_imdb)                                                                 | 2020-05-03 | 2020-05-03 |
| generative      | [Variational AutoEncoder](https://keras.io/examples/generative/vae)                                                                                 | 2020-05-03 | 2020-05-03 |
| generative      | [Neural style transfer](https://keras.io/examples/generative/neural_style_transfer)                                                                 | 2016-01-11 | 2020-05-02 |
| generative      | [Deep Dream](https://keras.io/examples/generative/deep_dream)                                                                                       | 2016-01-13 | 2020-05-02 |
| generative      | [Character-level text generation with LSTM](https://keras.io/examples/generative/lstm_character_level_text_generation)                              | 2015-06-15 | 2020-04-30 |
| vision          | [Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch)                                             | 2020-04-27 | 2020-04-28 |
| nlp             | [Character-level recurrent sequence-to-sequence model](https://keras.io/examples/nlp/lstm_seq2seq)                                                  | 2017-09-29 | 2020-04-26 |
| vision          | [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet)                                                                              | 2015-06-19 | 2020-04-21 |
| keras_recipes   | [Simple custom layer example: Antirectifier](https://keras.io/examples/keras_recipes/antirectifier)                                                 | 2016-01-06 | 2020-04-20 |
| vision          | [Image segmentation with a U-Net-like architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation)                                | 2019-03-20 | 2020-04-20 |
| keras_recipes   | [A Quasi-SVM in Keras](https://keras.io/examples/keras_recipes/quasi_svm)                                                                           | 2020-04-17 | 2020-04-17 |
| structured_data | [Imbalanced classification: credit card fraud detection](https://keras.io/examples/structured_data/imbalanced_classification)                       | 2019-05-28 | 2020-04-17 |
| nlp             | [Sequence to sequence learning for performing number addition](https://keras.io/examples/nlp/addition_rnn)                                          | 2015-08-17 | 2020-04-17 |
| audio           | [Speaker Recognition](https://keras.io/examples/audio/speaker_recognition_using_cnn)                                                                | 2020-06-14 | 2020-03-07 |
| keras_recipes   | [Endpoint layer pattern](https://keras.io/examples/keras_recipes/endpoint_layer_pattern)                                                            | 2019-05-10 | 2019-05-10 |



```python
# sorted by 'Category' and 'Date created'

sorted_report_df = report_df.sort_values(by=['Category', 'Date created'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```


| Category        | Title                                                                                                                                               | Date created        | Last modified       |
|:----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:--------------------|
| vision          | [Augmenting convnets with aggregated attention](https://keras.io/examples/vision/patch_convnet)                                                     | 2022-01-22 | 2022-01-22 |
| vision          | [Video Vision Transformer](https://keras.io/examples/vision/vivit)                                                                                  | 2022-01-12 | 2022-01-12 |
| vision          | [Train a Vision Transformer on small datasets](https://keras.io/examples/vision/vit_small_ds)                                                       | 2022-01-07 | 2022-01-10 |
| vision          | [Masked image modeling with Autoencoders](https://keras.io/examples/vision/masked_image_modeling)                                                   | 2021-12-20 | 2021-12-21 |
| vision          | [Learning to tokenize in Vision Transformers](https://keras.io/examples/vision/token_learner)                                                       | 2021-12-10 | 2021-12-15 |
| vision          | [Barlow Twins for Contrastive SSL](https://keras.io/examples/vision/barlow_twins)                                                                   | 2021-11-04 | 2021-12-20 |
| vision          | [MobileViT: A mobile-friendly Transformer-based model for image classification](https://keras.io/examples/vision/mobilevit)                         | 2021-10-20 | 2021-10-20 |
| vision          | [Image classification with EANet (External Attention Transformer)](https://keras.io/examples/vision/eanet)                                          | 2021-10-19 | 2021-10-19 |
| vision          | [Image classification with ConvMixer](https://keras.io/examples/vision/convmixer)                                                                   | 2021-10-12 | 2021-10-12 |
| vision          | [FixRes: Fixing train-test resolution discrepancy](https://keras.io/examples/vision/fixres)                                                         | 2021-10-08 | 2021-10-10 |
| vision          | [Metric learning for image similarity search using TensorFlow Similarity](https://keras.io/examples/vision/metric_learning_tf_similarity)           | 2021-09-30 | 2021-09-30 |
| vision          | [Image Classification using BigTransfer (BiT)](https://keras.io/examples/vision/bit)                                                                | 2021-09-24 | 2021-09-24 |
| vision          | [Zero-DCE for low-light image enhancement](https://keras.io/examples/vision/zero_dce)                                                               | 2021-09-18 | 2021-09-19 |
| vision          | [Self-supervised contrastive learning with NNCLR](https://keras.io/examples/vision/nnclr)                                                           | 2021-09-13 | 2021-09-13 |
| vision          | [Low-light image enhancement using MIRNet](https://keras.io/examples/vision/mirnet)                                                                 | 2021-09-11 | 2021-09-15 |
| vision          | [Near-duplicate image search](https://keras.io/examples/vision/near_dup_search)                                                                     | 2021-09-10 | 2021-09-10 |
| vision          | [Image classification with Swin Transformers](https://keras.io/examples/vision/swin_transformers)                                                   | 2021-09-08 | 2021-09-08 |
| vision          | [Multiclass semantic segmentation using DeepLabV3+](https://keras.io/examples/vision/deeplabv3_plus)                                                | 2021-08-31 | 2021-09-01 |
| vision          | [Monocular depth estimation](https://keras.io/examples/vision/depth_estimation)                                                                     | 2021-08-30 | 2021-08-30 |
| vision          | [Handwriting recognition](https://keras.io/examples/vision/handwriting_recognition)                                                                 | 2021-08-16 | 2021-08-16 |
| vision          | [Classification using Attention-based Deep Multiple Instance Learning (MIL).](https://keras.io/examples/vision/attention_mil_classification)        | 2021-08-16 | 2021-11-25 |
| vision          | [3D volumetric rendering with NeRF](https://keras.io/examples/vision/nerf)                                                                          | 2021-08-09 | 2021-08-09 |
| vision          | [Involutional neural networks](https://keras.io/examples/vision/involution)                                                                         | 2021-07-25 | 2021-07-25 |
| vision          | [Compact Convolutional Transformers](https://keras.io/examples/vision/cct)                                                                          | 2021-06-30 | 2021-06-30 |
| vision          | [Semi-supervision and domain adaptation with AdaMatch](https://keras.io/examples/vision/adamatch)                                                   | 2021-06-19 | 2021-06-19 |
| vision          | [Gradient Centralization for Better Training Performance](https://keras.io/examples/vision/gradient_centralization)                                 | 2021-06-18 | 2021-06-18 |
| vision          | [CutMix data augmentation for image classification](https://keras.io/examples/vision/cutmix)                                                        | 2021-06-08 | 2021-06-08 |
| vision          | [Video Classification with Transformers](https://keras.io/examples/vision/video_transformers)                                                       | 2021-06-08 | 2021-06-08 |
| vision          | [Next-Frame Video Prediction with Convolutional LSTMs](https://keras.io/examples/vision/conv_lstm)                                                  | 2021-06-02 | 2021-06-05 |
| vision          | [Image classification with modern MLP models](https://keras.io/examples/vision/mlp_image_classification)                                            | 2021-05-30 | 2021-05-30 |
| vision          | [Image Captioning](https://keras.io/examples/vision/image_captioning)                                                                               | 2021-05-29 | 2021-10-31 |
| vision          | [Video Classification with a CNN-RNN Architecture](https://keras.io/examples/vision/video_classification)                                           | 2021-05-28 | 2021-06-05 |
| vision          | [Image similarity estimation using a Siamese Network with a contrastive loss](https://keras.io/examples/vision/siamese_contrastive)                 | 2021-05-06 | 2021-05-06 |
| vision          | [Keypoint Detection with Transfer Learning](https://keras.io/examples/vision/keypoint_detection)                                                    | 2021-05-02 | 2021-05-02 |
| vision          | [Image classification with Perceiver](https://keras.io/examples/vision/perceiver_image_classification)                                              | 2021-04-30 | 2021-01-30 |
| vision          | [Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer)                                                         | 2021-04-30 | 2021-05-13 |
| vision          | [Semi-supervised image classification using contrastive pretraining with SimCLR](https://keras.io/examples/vision/semisupervised_simclr)            | 2021-04-24 | 2021-04-24 |
| vision          | [Consistency training with supervision](https://keras.io/examples/vision/consistency_training)                                                      | 2021-04-13 | 2021-04-19 |
| vision          | [Image similarity estimation using a Siamese Network with a triplet loss](https://keras.io/examples/vision/siamese_network)                         | 2021-03-25 | 2021-03-25 |
| vision          | [Self-supervised contrastive learning with SimSiam](https://keras.io/examples/vision/simsiam)                                                       | 2021-03-19 | 2021-03-20 |
| vision          | [RandAugment for Image Classification for Improved Robustness](https://keras.io/examples/vision/randaugment)                                        | 2021-03-13 | 2021-03-17 |
| vision          | [MixUp augmentation for image classification](https://keras.io/examples/vision/mixup)                                                               | 2021-03-06 | 2021-03-06 |
| vision          | [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder)                                                       | 2021-03-01 | 2021-03-01 |
| vision          | [Semantic Image Clustering](https://keras.io/examples/vision/semantic_image_clustering)                                                             | 2021-02-28 | 2021-02-28 |
| vision          | [Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer)                       | 2021-01-18 | 2021-01-18 |
| vision          | [Supervised Contrastive Learning](https://keras.io/examples/vision/supervised-contrastive-learning)                                                 | 2020-11-30 | 2020-11-30 |
| vision          | [Point cloud segmentation with PointNet](https://keras.io/examples/vision/pointnet_segmentation)                                                    | 2020-10-23 | 2020-10-24 |
| vision          | [3D image classification from CT scans](https://keras.io/examples/vision/3D_image_classification)                                                   | 2020-09-23 | 2020-09-23 |
| vision          | [Knowledge Distillation](https://keras.io/examples/vision/knowledge_distillation)                                                                   | 2020-09-01 | 2020-09-01 |
| vision          | [Pneumonia Classification on TPU](https://keras.io/examples/vision/xray_classification_with_tpus)                                                   | 2020-07-28 | 2020-08-24 |
| vision          | [Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel)                              | 2020-07-28 | 2020-08-27 |
| vision          | [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning)            | 2020-06-30 | 2020-07-16 |
| vision          | [OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr)                                                                      | 2020-06-14 | 2020-06-26 |
| vision          | [Metric learning for image similarity search](https://keras.io/examples/vision/metric_learning)                                                     | 2020-06-05 | 2020-06-09 |
| vision          | [Model interpretability with Integrated Gradients](https://keras.io/examples/vision/integrated_gradients)                                           | 2020-06-02 | 2020-06-02 |
| vision          | [Visualizing what convnets learn](https://keras.io/examples/vision/visualizing_what_convnets_learn)                                                 | 2020-05-29 | 2020-05-29 |
| vision          | [Point cloud classification with PointNet](https://keras.io/examples/vision/pointnet)                                                               | 2020-05-25 | 2020-05-26 |
| vision          | [Few-Shot learning with Reptile](https://keras.io/examples/vision/reptile)                                                                          | 2020-05-21 | 2020-05-30 |
| vision          | [Object Detection with RetinaNet](https://keras.io/examples/vision/retinanet)                                                                       | 2020-05-17 | 2020-07-14 |
| vision          | [Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch)                                             | 2020-04-27 | 2020-04-28 |
| vision          | [Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam)                                                                | 2020-04-26 | 2021-03-07 |
| vision          | [Image segmentation with a U-Net-like architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation)                                | 2019-03-20 | 2020-04-20 |
| vision          | [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet)                                                                              | 2015-06-19 | 2020-04-21 |
| timeseries      | [Traffic forecasting using graph neural networks and LSTM](https://keras.io/examples/timeseries/timeseries_traffic_forecasting)                     | 2021-12-28 | 2021-12-28 |
| timeseries      | [Timeseries classification with a Transformer model](https://keras.io/examples/timeseries/timeseries_classification_transformer)                    | 2021-06-25 | 2021-08-05 |
| timeseries      | [Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)                               | 2020-07-21 | 2021-07-16 |
| timeseries      | [Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting)                                | 2020-06-23 | 2020-07-20 |
| timeseries      | [Timeseries anomaly detection using an Autoencoder](https://keras.io/examples/timeseries/timeseries_anomaly_detection)                              | 2020-05-31 | 2020-05-31 |
| structured_data | [Classification with TensorFlow Decision Forests](https://keras.io/examples/structured_data/classification_with_tfdf)                               | 2022-01-25 | 2022-01-25 |
| structured_data | [Structured data learning with TabTransformer](https://keras.io/examples/structured_data/tabtransformer)                                            | 2022-01-18 | 2022-01-18 |
| structured_data | [Classification with Gated Residual and Variable Selection Networks](https://keras.io/examples/structured_data/classification_with_grn_and_vsn)     | 2021-02-10 | 2021-02-10 |
| structured_data | [Classification with Neural Decision Forests](https://keras.io/examples/structured_data/deep_neural_decision_forests)                               | 2021-01-15 | 2021-01-15 |
| structured_data | [Structured data learning with Wide, Deep, and Cross networks](https://keras.io/examples/structured_data/wide_deep_cross_networks)                  | 2020-12-31 | 2021-05-05 |
| structured_data | [A Transformer-based recommendation system](https://keras.io/examples/structured_data/movielens_recommendations_transformers)                       | 2020-12-30 | 2020-12-30 |
| structured_data | [Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch)                | 2020-06-09 | 2020-06-09 |
| structured_data | [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens)                    | 2020-05-24 | 2020-05-24 |
| structured_data | [Imbalanced classification: credit card fraud detection](https://keras.io/examples/structured_data/imbalanced_classification)                       | 2019-05-28 | 2020-04-17 |
| rl              | [Proximal Policy Optimization](https://keras.io/examples/rl/ppo_cartpole)                                                                           | 2021-06-24 | 2021-06-24 |
| rl              | [Deep Deterministic Policy Gradient (DDPG)](https://keras.io/examples/rl/ddpg_pendulum)                                                             | 2020-06-04 | 2020-09-21 |
| rl              | [Deep Q-Learning for Atari Breakout](https://keras.io/examples/rl/deep_q_network_breakout)                                                          | 2020-05-23 | 2020-06-17 |
| rl              | [Actor Critic Method](https://keras.io/examples/rl/actor_critic_cartpole)                                                                           | 2020-05-13 | 2020-05-13 |
| nlp             | [Question Answering with Hugging Face Transformers](https://keras.io/examples/nlp/question_answering)                                               | 2022-01-13 | 2022-01-13 |
| nlp             | [Review Classification using Active Learning](https://keras.io/examples/nlp/active_learning_review_classification)                                  | 2021-10-29 | 2021-10-29 |
| nlp             | [Text Generation using FNet](https://keras.io/examples/nlp/text_generation_fnet)                                                                    | 2021-10-05 | 2021-10-05 |
| nlp             | [Multimodal entailment](https://keras.io/examples/nlp/multimodal_entailment)                                                                        | 2021-08-08 | 2021-08-15 |
| nlp             | [Named Entity Recognition using Transformers](https://keras.io/examples/nlp/ner_transformers)                                                       | 2021-06-23 | 2021-06-24 |
| nlp             | [English-to-Spanish translation with a sequence-to-sequence Transformer](https://keras.io/examples/nlp/neural_machine_translation_with_transformer) | 2021-05-26 | 2021-05-26 |
| nlp             | [Natural language image search with a Dual Encoder](https://keras.io/examples/nlp/nl_image_search)                                                  | 2021-01-30 | 2021-01-30 |
| nlp             | [Large-scale multi-label text classification](https://keras.io/examples/nlp/multi_label_classification)                                             | 2020-09-25 | 2020-12-23 |
| nlp             | [End-to-end Masked Language Modeling with BERT](https://keras.io/examples/nlp/masked_language_modeling)                                             | 2020-09-18 | 2020-09-18 |
| nlp             | [Semantic Similarity with BERT](https://keras.io/examples/nlp/semantic_similarity_with_bert)                                                        | 2020-08-15 | 2020-08-29 |
| nlp             | [Text Extraction with BERT](https://keras.io/examples/nlp/text_extraction_with_bert)                                                                | 2020-05-23 | 2020-05-23 |
| nlp             | [Text classification with Switch Transformer](https://keras.io/examples/nlp/text_classification_with_switch_transformer)                            | 2020-05-10 | 2021-02-15 |
| nlp             | [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer)                                          | 2020-05-10 | 2020-05-10 |
| nlp             | [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings)                                                       | 2020-05-05 | 2020-05-05 |
| nlp             | [Bidirectional LSTM on IMDB](https://keras.io/examples/nlp/bidirectional_lstm_imdb)                                                                 | 2020-05-03 | 2020-05-03 |
| nlp             | [Text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch)                                                  | 2019-11-06 | 2020-05-17 |
| nlp             | [Character-level recurrent sequence-to-sequence model](https://keras.io/examples/nlp/lstm_seq2seq)                                                  | 2017-09-29 | 2020-04-26 |
| nlp             | [Sequence to sequence learning for performing number addition](https://keras.io/examples/nlp/addition_rnn)                                          | 2015-08-17 | 2020-04-17 |
| keras_recipes   | [Customizing the convolution operation of a Conv2D layer](https://keras.io/examples/keras_recipes/subclassing_conv_layers)                          | 2021-11-03 | 2021-11-03 |
| keras_recipes   | [Evaluating and exporting scikit-learn metrics in a Keras callback](https://keras.io/examples/keras_recipes/sklearn_metric_callbacks)               | 2021-10-07 | 2021-10-07 |
| keras_recipes   | [Writing Keras Models With TensorFlow NumPy](https://keras.io/examples/keras_recipes/tensorflow_nu_models)                                          | 2021-08-28 | 2021-08-28 |
| keras_recipes   | [Knowledge distillation recipes](https://keras.io/examples/keras_recipes/better_knowledge_distillation)                                             | 2021-08-01 | 2021-08-01 |
| keras_recipes   | [Estimating required sample size for model training](https://keras.io/examples/keras_recipes/sample_size_estimate)                                  | 2021-05-20 | 2021-06-06 |
| keras_recipes   | [Creating TFRecords](https://keras.io/examples/keras_recipes/creating_tfrecords)                                                                    | 2021-02-27 | 2021-02-27 |
| keras_recipes   | [Memory-efficient embeddings for recommendation systems](https://keras.io/examples/keras_recipes/memory_efficient_embeddings)                       | 2021-02-15 | 2021-02-15 |
| keras_recipes   | [Probabilistic Bayesian Neural Networks](https://keras.io/examples/keras_recipes/bayesian_neural_networks)                                          | 2021-01-15 | 2021-01-15 |
| keras_recipes   | [How to train a Keras model on TFRecord files](https://keras.io/examples/keras_recipes/tfrecord)                                                    | 2020-07-29 | 2020-08-07 |
| keras_recipes   | [Keras debugging tips](https://keras.io/examples/keras_recipes/debugging_tips)                                                                      | 2020-05-16 | 2020-05-16 |
| keras_recipes   | [A Quasi-SVM in Keras](https://keras.io/examples/keras_recipes/quasi_svm)                                                                           | 2020-04-17 | 2020-04-17 |
| keras_recipes   | [Endpoint layer pattern](https://keras.io/examples/keras_recipes/endpoint_layer_pattern)                                                            | 2019-05-10 | 2019-05-10 |
| keras_recipes   | [Simple custom layer example: Antirectifier](https://keras.io/examples/keras_recipes/antirectifier)                                                 | 2016-01-06 | 2020-04-20 |
| graph           | [Graph attention network (GAT) for node classification](https://keras.io/examples/graph/gat_node_classification)                                    | 2021-09-13 | 2021-12-26 |
| graph           | [Message-passing neural network (MPNN) for molecular property prediction](https://keras.io/examples/graph/mpnn-molecular-graphs)                    | 2021-08-16 | 2021-12-27 |
| graph           | [Node Classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations)                                                     | 2021-05-30 | 2021-05-30 |
| graph           | [Graph representation learning with node2vec](https://keras.io/examples/graph/node2vec_movielens)                                                   | 2021-05-15 | 2021-05-15 |
| generative      | [GauGAN for conditional image generation](https://keras.io/examples/generative/gaugan)                                                              | 2021-12-26 | 2022-01-03 |
| generative      | [Neural Style Transfer with AdaIN](https://keras.io/examples/generative/adain)                                                                      | 2021-11-08 | 2021-11-08 |
| generative      | [Data-efficient GANs with Adaptive Discriminator Augmentation](https://keras.io/examples/generative/gan_ada)                                        | 2021-10-28 | 2021-10-28 |
| generative      | [Vector-Quantized Variational Autoencoders](https://keras.io/examples/generative/vq_vae)                                                            | 2021-07-21 | 2021-07-21 |
| generative      | [Conditional GAN](https://keras.io/examples/generative/conditional_gan)                                                                             | 2021-07-13 | 2021-07-15 |
| generative      | [Face image generation with StyleGAN](https://keras.io/examples/generative/stylegan)                                                                | 2021-07-01 | 2021-07-01 |
| generative      | [WGAN-GP with R-GCN for the generation of small molecular graphs](https://keras.io/examples/generative/wgan-graphs)                                 | 2021-06-30 | 2021-06-30 |
| generative      | [CycleGAN](https://keras.io/examples/generative/cyclegan)                                                                                           | 2020-08-12 | 2020-08-12 |
| generative      | [Density estimation using Real NVP](https://keras.io/examples/generative/real_nvp)                                                                  | 2020-08-10 | 2020-08-10 |
| generative      | [Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt)                                     | 2020-05-29 | 2020-05-29 |
| generative      | [PixelCNN](https://keras.io/examples/generative/pixelcnn)                                                                                           | 2020-05-17 | 2020-05-23 |
| generative      | [WGAN-GP overriding `Model.train_step`](https://keras.io/examples/generative/wgan_gp)                                                               | 2020-05-09 | 2020-05-09 |
| generative      | [Variational AutoEncoder](https://keras.io/examples/generative/vae)                                                                                 | 2020-05-03 | 2020-05-03 |
| generative      | [DCGAN to generate face images](https://keras.io/examples/generative/dcgan_overriding_train_step)                                                   | 2019-04-29 | 2021-01-01 |
| generative      | [Deep Dream](https://keras.io/examples/generative/deep_dream)                                                                                       | 2016-01-13 | 2020-05-02 |
| generative      | [Neural style transfer](https://keras.io/examples/generative/neural_style_transfer)                                                                 | 2016-01-11 | 2020-05-02 |
| generative      | [Character-level text generation with LSTM](https://keras.io/examples/generative/lstm_character_level_text_generation)                              | 2015-06-15 | 2020-04-30 |
| audio           | [Automatic Speech Recognition using CTC](https://keras.io/examples/audio/ctc_asr)                                                                   | 2021-09-26 | 2021-09-26 |
| audio           | [MelGAN-based spectrogram inversion using feature matching](https://keras.io/examples/audio/melgan_spectrogram_inversion)                           | 2021-02-09 | 2021-09-15 |
| audio           | [Automatic Speech Recognition with Transformer](https://keras.io/examples/audio/transformer_asr)                                                    | 2021-01-13 | 2021-01-13 |
| audio           | [Speaker Recognition](https://keras.io/examples/audio/speaker_recognition_using_cnn)                                                                | 2020-06-14 | 2020-03-07 |



```python
# sorted by 'Category' and 'Last modified'

sorted_report_df = report_df.sort_values(by=['Category', 'Last modified'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```

| Category        | Title                                                                                                                                               | Date created        | Last modified       |
|:----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:--------------------|
| vision          | [Augmenting convnets with aggregated attention](https://keras.io/examples/vision/patch_convnet)                                                     | 2022-01-22 | 2022-01-22 |
| vision          | [Video Vision Transformer](https://keras.io/examples/vision/vivit)                                                                                  | 2022-01-12 | 2022-01-12 |
| vision          | [Train a Vision Transformer on small datasets](https://keras.io/examples/vision/vit_small_ds)                                                       | 2022-01-07 | 2022-01-10 |
| vision          | [Masked image modeling with Autoencoders](https://keras.io/examples/vision/masked_image_modeling)                                                   | 2021-12-20 | 2021-12-21 |
| vision          | [Barlow Twins for Contrastive SSL](https://keras.io/examples/vision/barlow_twins)                                                                   | 2021-11-04 | 2021-12-20 |
| vision          | [Learning to tokenize in Vision Transformers](https://keras.io/examples/vision/token_learner)                                                       | 2021-12-10 | 2021-12-15 |
| vision          | [Classification using Attention-based Deep Multiple Instance Learning (MIL).](https://keras.io/examples/vision/attention_mil_classification)        | 2021-08-16 | 2021-11-25 |
| vision          | [Image Captioning](https://keras.io/examples/vision/image_captioning)                                                                               | 2021-05-29 | 2021-10-31 |
| vision          | [MobileViT: A mobile-friendly Transformer-based model for image classification](https://keras.io/examples/vision/mobilevit)                         | 2021-10-20 | 2021-10-20 |
| vision          | [Image classification with EANet (External Attention Transformer)](https://keras.io/examples/vision/eanet)                                          | 2021-10-19 | 2021-10-19 |
| vision          | [Image classification with ConvMixer](https://keras.io/examples/vision/convmixer)                                                                   | 2021-10-12 | 2021-10-12 |
| vision          | [FixRes: Fixing train-test resolution discrepancy](https://keras.io/examples/vision/fixres)                                                         | 2021-10-08 | 2021-10-10 |
| vision          | [Metric learning for image similarity search using TensorFlow Similarity](https://keras.io/examples/vision/metric_learning_tf_similarity)           | 2021-09-30 | 2021-09-30 |
| vision          | [Image Classification using BigTransfer (BiT)](https://keras.io/examples/vision/bit)                                                                | 2021-09-24 | 2021-09-24 |
| vision          | [Zero-DCE for low-light image enhancement](https://keras.io/examples/vision/zero_dce)                                                               | 2021-09-18 | 2021-09-19 |
| vision          | [Low-light image enhancement using MIRNet](https://keras.io/examples/vision/mirnet)                                                                 | 2021-09-11 | 2021-09-15 |
| vision          | [Self-supervised contrastive learning with NNCLR](https://keras.io/examples/vision/nnclr)                                                           | 2021-09-13 | 2021-09-13 |
| vision          | [Near-duplicate image search](https://keras.io/examples/vision/near_dup_search)                                                                     | 2021-09-10 | 2021-09-10 |
| vision          | [Image classification with Swin Transformers](https://keras.io/examples/vision/swin_transformers)                                                   | 2021-09-08 | 2021-09-08 |
| vision          | [Multiclass semantic segmentation using DeepLabV3+](https://keras.io/examples/vision/deeplabv3_plus)                                                | 2021-08-31 | 2021-09-01 |
| vision          | [Monocular depth estimation](https://keras.io/examples/vision/depth_estimation)                                                                     | 2021-08-30 | 2021-08-30 |
| vision          | [Handwriting recognition](https://keras.io/examples/vision/handwriting_recognition)                                                                 | 2021-08-16 | 2021-08-16 |
| vision          | [3D volumetric rendering with NeRF](https://keras.io/examples/vision/nerf)                                                                          | 2021-08-09 | 2021-08-09 |
| vision          | [Involutional neural networks](https://keras.io/examples/vision/involution)                                                                         | 2021-07-25 | 2021-07-25 |
| vision          | [Compact Convolutional Transformers](https://keras.io/examples/vision/cct)                                                                          | 2021-06-30 | 2021-06-30 |
| vision          | [Semi-supervision and domain adaptation with AdaMatch](https://keras.io/examples/vision/adamatch)                                                   | 2021-06-19 | 2021-06-19 |
| vision          | [Gradient Centralization for Better Training Performance](https://keras.io/examples/vision/gradient_centralization)                                 | 2021-06-18 | 2021-06-18 |
| vision          | [CutMix data augmentation for image classification](https://keras.io/examples/vision/cutmix)                                                        | 2021-06-08 | 2021-06-08 |
| vision          | [Video Classification with Transformers](https://keras.io/examples/vision/video_transformers)                                                       | 2021-06-08 | 2021-06-08 |
| vision          | [Video Classification with a CNN-RNN Architecture](https://keras.io/examples/vision/video_classification)                                           | 2021-05-28 | 2021-06-05 |
| vision          | [Next-Frame Video Prediction with Convolutional LSTMs](https://keras.io/examples/vision/conv_lstm)                                                  | 2021-06-02 | 2021-06-05 |
| vision          | [Image classification with modern MLP models](https://keras.io/examples/vision/mlp_image_classification)                                            | 2021-05-30 | 2021-05-30 |
| vision          | [Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer)                                                         | 2021-04-30 | 2021-05-13 |
| vision          | [Image similarity estimation using a Siamese Network with a contrastive loss](https://keras.io/examples/vision/siamese_contrastive)                 | 2021-05-06 | 2021-05-06 |
| vision          | [Keypoint Detection with Transfer Learning](https://keras.io/examples/vision/keypoint_detection)                                                    | 2021-05-02 | 2021-05-02 |
| vision          | [Semi-supervised image classification using contrastive pretraining with SimCLR](https://keras.io/examples/vision/semisupervised_simclr)            | 2021-04-24 | 2021-04-24 |
| vision          | [Consistency training with supervision](https://keras.io/examples/vision/consistency_training)                                                      | 2021-04-13 | 2021-04-19 |
| vision          | [Image similarity estimation using a Siamese Network with a triplet loss](https://keras.io/examples/vision/siamese_network)                         | 2021-03-25 | 2021-03-25 |
| vision          | [Self-supervised contrastive learning with SimSiam](https://keras.io/examples/vision/simsiam)                                                       | 2021-03-19 | 2021-03-20 |
| vision          | [RandAugment for Image Classification for Improved Robustness](https://keras.io/examples/vision/randaugment)                                        | 2021-03-13 | 2021-03-17 |
| vision          | [Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam)                                                                | 2020-04-26 | 2021-03-07 |
| vision          | [MixUp augmentation for image classification](https://keras.io/examples/vision/mixup)                                                               | 2021-03-06 | 2021-03-06 |
| vision          | [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder)                                                       | 2021-03-01 | 2021-03-01 |
| vision          | [Semantic Image Clustering](https://keras.io/examples/vision/semantic_image_clustering)                                                             | 2021-02-28 | 2021-02-28 |
| vision          | [Image classification with Perceiver](https://keras.io/examples/vision/perceiver_image_classification)                                              | 2021-04-30 | 2021-01-30 |
| vision          | [Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer)                       | 2021-01-18 | 2021-01-18 |
| vision          | [Supervised Contrastive Learning](https://keras.io/examples/vision/supervised-contrastive-learning)                                                 | 2020-11-30 | 2020-11-30 |
| vision          | [Point cloud segmentation with PointNet](https://keras.io/examples/vision/pointnet_segmentation)                                                    | 2020-10-23 | 2020-10-24 |
| vision          | [3D image classification from CT scans](https://keras.io/examples/vision/3D_image_classification)                                                   | 2020-09-23 | 2020-09-23 |
| vision          | [Knowledge Distillation](https://keras.io/examples/vision/knowledge_distillation)                                                                   | 2020-09-01 | 2020-09-01 |
| vision          | [Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel)                              | 2020-07-28 | 2020-08-27 |
| vision          | [Pneumonia Classification on TPU](https://keras.io/examples/vision/xray_classification_with_tpus)                                                   | 2020-07-28 | 2020-08-24 |
| vision          | [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning)            | 2020-06-30 | 2020-07-16 |
| vision          | [Object Detection with RetinaNet](https://keras.io/examples/vision/retinanet)                                                                       | 2020-05-17 | 2020-07-14 |
| vision          | [OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr)                                                                      | 2020-06-14 | 2020-06-26 |
| vision          | [Metric learning for image similarity search](https://keras.io/examples/vision/metric_learning)                                                     | 2020-06-05 | 2020-06-09 |
| vision          | [Model interpretability with Integrated Gradients](https://keras.io/examples/vision/integrated_gradients)                                           | 2020-06-02 | 2020-06-02 |
| vision          | [Few-Shot learning with Reptile](https://keras.io/examples/vision/reptile)                                                                          | 2020-05-21 | 2020-05-30 |
| vision          | [Visualizing what convnets learn](https://keras.io/examples/vision/visualizing_what_convnets_learn)                                                 | 2020-05-29 | 2020-05-29 |
| vision          | [Point cloud classification with PointNet](https://keras.io/examples/vision/pointnet)                                                               | 2020-05-25 | 2020-05-26 |
| vision          | [Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch)                                             | 2020-04-27 | 2020-04-28 |
| vision          | [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet)                                                                              | 2015-06-19 | 2020-04-21 |
| vision          | [Image segmentation with a U-Net-like architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation)                                | 2019-03-20 | 2020-04-20 |
| timeseries      | [Traffic forecasting using graph neural networks and LSTM](https://keras.io/examples/timeseries/timeseries_traffic_forecasting)                     | 2021-12-28 | 2021-12-28 |
| timeseries      | [Timeseries classification with a Transformer model](https://keras.io/examples/timeseries/timeseries_classification_transformer)                    | 2021-06-25 | 2021-08-05 |
| timeseries      | [Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)                               | 2020-07-21 | 2021-07-16 |
| timeseries      | [Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting)                                | 2020-06-23 | 2020-07-20 |
| timeseries      | [Timeseries anomaly detection using an Autoencoder](https://keras.io/examples/timeseries/timeseries_anomaly_detection)                              | 2020-05-31 | 2020-05-31 |
| structured_data | [Classification with TensorFlow Decision Forests](https://keras.io/examples/structured_data/classification_with_tfdf)                               | 2022-01-25 | 2022-01-25 |
| structured_data | [Structured data learning with TabTransformer](https://keras.io/examples/structured_data/tabtransformer)                                            | 2022-01-18 | 2022-01-18 |
| structured_data | [Structured data learning with Wide, Deep, and Cross networks](https://keras.io/examples/structured_data/wide_deep_cross_networks)                  | 2020-12-31 | 2021-05-05 |
| structured_data | [Classification with Gated Residual and Variable Selection Networks](https://keras.io/examples/structured_data/classification_with_grn_and_vsn)     | 2021-02-10 | 2021-02-10 |
| structured_data | [Classification with Neural Decision Forests](https://keras.io/examples/structured_data/deep_neural_decision_forests)                               | 2021-01-15 | 2021-01-15 |
| structured_data | [A Transformer-based recommendation system](https://keras.io/examples/structured_data/movielens_recommendations_transformers)                       | 2020-12-30 | 2020-12-30 |
| structured_data | [Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch)                | 2020-06-09 | 2020-06-09 |
| structured_data | [Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens)                    | 2020-05-24 | 2020-05-24 |
| structured_data | [Imbalanced classification: credit card fraud detection](https://keras.io/examples/structured_data/imbalanced_classification)                       | 2019-05-28 | 2020-04-17 |
| rl              | [Proximal Policy Optimization](https://keras.io/examples/rl/ppo_cartpole)                                                                           | 2021-06-24 | 2021-06-24 |
| rl              | [Deep Deterministic Policy Gradient (DDPG)](https://keras.io/examples/rl/ddpg_pendulum)                                                             | 2020-06-04 | 2020-09-21 |
| rl              | [Deep Q-Learning for Atari Breakout](https://keras.io/examples/rl/deep_q_network_breakout)                                                          | 2020-05-23 | 2020-06-17 |
| rl              | [Actor Critic Method](https://keras.io/examples/rl/actor_critic_cartpole)                                                                           | 2020-05-13 | 2020-05-13 |
| nlp             | [Question Answering with Hugging Face Transformers](https://keras.io/examples/nlp/question_answering)                                               | 2022-01-13 | 2022-01-13 |
| nlp             | [Review Classification using Active Learning](https://keras.io/examples/nlp/active_learning_review_classification)                                  | 2021-10-29 | 2021-10-29 |
| nlp             | [Text Generation using FNet](https://keras.io/examples/nlp/text_generation_fnet)                                                                    | 2021-10-05 | 2021-10-05 |
| nlp             | [Multimodal entailment](https://keras.io/examples/nlp/multimodal_entailment)                                                                        | 2021-08-08 | 2021-08-15 |
| nlp             | [Named Entity Recognition using Transformers](https://keras.io/examples/nlp/ner_transformers)                                                       | 2021-06-23 | 2021-06-24 |
| nlp             | [English-to-Spanish translation with a sequence-to-sequence Transformer](https://keras.io/examples/nlp/neural_machine_translation_with_transformer) | 2021-05-26 | 2021-05-26 |
| nlp             | [Text classification with Switch Transformer](https://keras.io/examples/nlp/text_classification_with_switch_transformer)                            | 2020-05-10 | 2021-02-15 |
| nlp             | [Natural language image search with a Dual Encoder](https://keras.io/examples/nlp/nl_image_search)                                                  | 2021-01-30 | 2021-01-30 |
| nlp             | [Large-scale multi-label text classification](https://keras.io/examples/nlp/multi_label_classification)                                             | 2020-09-25 | 2020-12-23 |
| nlp             | [End-to-end Masked Language Modeling with BERT](https://keras.io/examples/nlp/masked_language_modeling)                                             | 2020-09-18 | 2020-09-18 |
| nlp             | [Semantic Similarity with BERT](https://keras.io/examples/nlp/semantic_similarity_with_bert)                                                        | 2020-08-15 | 2020-08-29 |
| nlp             | [Text Extraction with BERT](https://keras.io/examples/nlp/text_extraction_with_bert)                                                                | 2020-05-23 | 2020-05-23 |
| nlp             | [Text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch)                                                  | 2019-11-06 | 2020-05-17 |
| nlp             | [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer)                                          | 2020-05-10 | 2020-05-10 |
| nlp             | [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings)                                                       | 2020-05-05 | 2020-05-05 |
| nlp             | [Bidirectional LSTM on IMDB](https://keras.io/examples/nlp/bidirectional_lstm_imdb)                                                                 | 2020-05-03 | 2020-05-03 |
| nlp             | [Character-level recurrent sequence-to-sequence model](https://keras.io/examples/nlp/lstm_seq2seq)                                                  | 2017-09-29 | 2020-04-26 |
| nlp             | [Sequence to sequence learning for performing number addition](https://keras.io/examples/nlp/addition_rnn)                                          | 2015-08-17 | 2020-04-17 |
| keras_recipes   | [Customizing the convolution operation of a Conv2D layer](https://keras.io/examples/keras_recipes/subclassing_conv_layers)                          | 2021-11-03 | 2021-11-03 |
| keras_recipes   | [Evaluating and exporting scikit-learn metrics in a Keras callback](https://keras.io/examples/keras_recipes/sklearn_metric_callbacks)               | 2021-10-07 | 2021-10-07 |
| keras_recipes   | [Writing Keras Models With TensorFlow NumPy](https://keras.io/examples/keras_recipes/tensorflow_nu_models)                                          | 2021-08-28 | 2021-08-28 |
| keras_recipes   | [Knowledge distillation recipes](https://keras.io/examples/keras_recipes/better_knowledge_distillation)                                             | 2021-08-01 | 2021-08-01 |
| keras_recipes   | [Estimating required sample size for model training](https://keras.io/examples/keras_recipes/sample_size_estimate)                                  | 2021-05-20 | 2021-06-06 |
| keras_recipes   | [Creating TFRecords](https://keras.io/examples/keras_recipes/creating_tfrecords)                                                                    | 2021-02-27 | 2021-02-27 |
| keras_recipes   | [Memory-efficient embeddings for recommendation systems](https://keras.io/examples/keras_recipes/memory_efficient_embeddings)                       | 2021-02-15 | 2021-02-15 |
| keras_recipes   | [Probabilistic Bayesian Neural Networks](https://keras.io/examples/keras_recipes/bayesian_neural_networks)                                          | 2021-01-15 | 2021-01-15 |
| keras_recipes   | [How to train a Keras model on TFRecord files](https://keras.io/examples/keras_recipes/tfrecord)                                                    | 2020-07-29 | 2020-08-07 |
| keras_recipes   | [Keras debugging tips](https://keras.io/examples/keras_recipes/debugging_tips)                                                                      | 2020-05-16 | 2020-05-16 |
| keras_recipes   | [Simple custom layer example: Antirectifier](https://keras.io/examples/keras_recipes/antirectifier)                                                 | 2016-01-06 | 2020-04-20 |
| keras_recipes   | [A Quasi-SVM in Keras](https://keras.io/examples/keras_recipes/quasi_svm)                                                                           | 2020-04-17 | 2020-04-17 |
| keras_recipes   | [Endpoint layer pattern](https://keras.io/examples/keras_recipes/endpoint_layer_pattern)                                                            | 2019-05-10 | 2019-05-10 |
| graph           | [Message-passing neural network (MPNN) for molecular property prediction](https://keras.io/examples/graph/mpnn-molecular-graphs)                    | 2021-08-16 | 2021-12-27 |
| graph           | [Graph attention network (GAT) for node classification](https://keras.io/examples/graph/gat_node_classification)                                    | 2021-09-13 | 2021-12-26 |
| graph           | [Node Classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations)                                                     | 2021-05-30 | 2021-05-30 |
| graph           | [Graph representation learning with node2vec](https://keras.io/examples/graph/node2vec_movielens)                                                   | 2021-05-15 | 2021-05-15 |
| generative      | [GauGAN for conditional image generation](https://keras.io/examples/generative/gaugan)                                                              | 2021-12-26 | 2022-01-03 |
| generative      | [Neural Style Transfer with AdaIN](https://keras.io/examples/generative/adain)                                                                      | 2021-11-08 | 2021-11-08 |
| generative      | [Data-efficient GANs with Adaptive Discriminator Augmentation](https://keras.io/examples/generative/gan_ada)                                        | 2021-10-28 | 2021-10-28 |
| generative      | [Vector-Quantized Variational Autoencoders](https://keras.io/examples/generative/vq_vae)                                                            | 2021-07-21 | 2021-07-21 |
| generative      | [Conditional GAN](https://keras.io/examples/generative/conditional_gan)                                                                             | 2021-07-13 | 2021-07-15 |
| generative      | [Face image generation with StyleGAN](https://keras.io/examples/generative/stylegan)                                                                | 2021-07-01 | 2021-07-01 |
| generative      | [WGAN-GP with R-GCN for the generation of small molecular graphs](https://keras.io/examples/generative/wgan-graphs)                                 | 2021-06-30 | 2021-06-30 |
| generative      | [DCGAN to generate face images](https://keras.io/examples/generative/dcgan_overriding_train_step)                                                   | 2019-04-29 | 2021-01-01 |
| generative      | [CycleGAN](https://keras.io/examples/generative/cyclegan)                                                                                           | 2020-08-12 | 2020-08-12 |
| generative      | [Density estimation using Real NVP](https://keras.io/examples/generative/real_nvp)                                                                  | 2020-08-10 | 2020-08-10 |
| generative      | [Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt)                                     | 2020-05-29 | 2020-05-29 |
| generative      | [PixelCNN](https://keras.io/examples/generative/pixelcnn)                                                                                           | 2020-05-17 | 2020-05-23 |
| generative      | [WGAN-GP overriding `Model.train_step`](https://keras.io/examples/generative/wgan_gp)                                                               | 2020-05-09 | 2020-05-09 |
| generative      | [Variational AutoEncoder](https://keras.io/examples/generative/vae)                                                                                 | 2020-05-03 | 2020-05-03 |
| generative      | [Deep Dream](https://keras.io/examples/generative/deep_dream)                                                                                       | 2016-01-13 | 2020-05-02 |
| generative      | [Neural style transfer](https://keras.io/examples/generative/neural_style_transfer)                                                                 | 2016-01-11 | 2020-05-02 |
| generative      | [Character-level text generation with LSTM](https://keras.io/examples/generative/lstm_character_level_text_generation)                              | 2015-06-15 | 2020-04-30 |
| audio           | [Automatic Speech Recognition using CTC](https://keras.io/examples/audio/ctc_asr)                                                                   | 2021-09-26 | 2021-09-26 |
| audio           | [MelGAN-based spectrogram inversion using feature matching](https://keras.io/examples/audio/melgan_spectrogram_inversion)                           | 2021-02-09 | 2021-09-15 |
| audio           | [Automatic Speech Recognition with Transformer](https://keras.io/examples/audio/transformer_asr)                                                    | 2021-01-13 | 2021-01-13 |
| audio           | [Speaker Recognition](https://keras.io/examples/audio/speaker_recognition_using_cnn)                                                                | 2020-06-14 | 2020-03-07 |

