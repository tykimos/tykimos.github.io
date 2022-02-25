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

| Category        | Title                                                                                                                                                        | Date created        | Last modified       |
|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:--------------------|
| structured_data | <a href=https://keras.io/examples/structured_data/classification_with_tfdf>Classification with TensorFlow Decision Forests</a>                               | 2022-01-25 | 2022-01-25 |
| vision          | <a href=https://keras.io/examples/vision/patch_convnet>Augmenting convnets with aggregated attention</a>                                                     | 2022-01-22 | 2022-01-22 |
| structured_data | <a href=https://keras.io/examples/structured_data/tabtransformer>Structured data learning with TabTransformer</a>                                            | 2022-01-18 | 2022-01-18 |
| nlp             | <a href=https://keras.io/examples/nlp/question_answering>Question Answering with Hugging Face Transformers</a>                                               | 2022-01-13 | 2022-01-13 |
| vision          | <a href=https://keras.io/examples/vision/vivit>Video Vision Transformer</a>                                                                                  | 2022-01-12 | 2022-01-12 |
| vision          | <a href=https://keras.io/examples/vision/vit_small_ds>Train a Vision Transformer on small datasets</a>                                                       | 2022-01-07 | 2022-01-10 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_traffic_forecasting>Traffic forecasting using graph neural networks and LSTM</a>                     | 2021-12-28 | 2021-12-28 |
| generative      | <a href=https://keras.io/examples/generative/gaugan>GauGAN for conditional image generation</a>                                                              | 2021-12-26 | 2022-01-03 |
| vision          | <a href=https://keras.io/examples/vision/masked_image_modeling>Masked image modeling with Autoencoders</a>                                                   | 2021-12-20 | 2021-12-21 |
| vision          | <a href=https://keras.io/examples/vision/token_learner>Learning to tokenize in Vision Transformers</a>                                                       | 2021-12-10 | 2021-12-15 |
| generative      | <a href=https://keras.io/examples/generative/adain>Neural Style Transfer with AdaIN</a>                                                                      | 2021-11-08 | 2021-11-08 |
| vision          | <a href=https://keras.io/examples/vision/barlow_twins>Barlow Twins for Contrastive SSL</a>                                                                   | 2021-11-04 | 2021-12-20 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/subclassing_conv_layers>Customizing the convolution operation of a Conv2D layer</a>                          | 2021-11-03 | 2021-11-03 |
| nlp             | <a href=https://keras.io/examples/nlp/active_learning_review_classification>Review Classification using Active Learning</a>                                  | 2021-10-29 | 2021-10-29 |
| generative      | <a href=https://keras.io/examples/generative/gan_ada>Data-efficient GANs with Adaptive Discriminator Augmentation</a>                                        | 2021-10-28 | 2021-10-28 |
| vision          | <a href=https://keras.io/examples/vision/mobilevit>MobileViT: A mobile-friendly Transformer-based model for image classification</a>                         | 2021-10-20 | 2021-10-20 |
| vision          | <a href=https://keras.io/examples/vision/eanet>Image classification with EANet (External Attention Transformer)</a>                                          | 2021-10-19 | 2021-10-19 |
| vision          | <a href=https://keras.io/examples/vision/convmixer>Image classification with ConvMixer</a>                                                                   | 2021-10-12 | 2021-10-12 |
| vision          | <a href=https://keras.io/examples/vision/fixres>FixRes: Fixing train-test resolution discrepancy</a>                                                         | 2021-10-08 | 2021-10-10 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/sklearn_metric_callbacks>Evaluating and exporting scikit-learn metrics in a Keras callback</a>               | 2021-10-07 | 2021-10-07 |
| nlp             | <a href=https://keras.io/examples/nlp/text_generation_fnet>Text Generation using FNet</a>                                                                    | 2021-10-05 | 2021-10-05 |
| vision          | <a href=https://keras.io/examples/vision/metric_learning_tf_similarity>Metric learning for image similarity search using TensorFlow Similarity</a>           | 2021-09-30 | 2021-09-30 |
| audio           | <a href=https://keras.io/examples/audio/ctc_asr>Automatic Speech Recognition using CTC</a>                                                                   | 2021-09-26 | 2021-09-26 |
| vision          | <a href=https://keras.io/examples/vision/bit>Image Classification using BigTransfer (BiT)</a>                                                                | 2021-09-24 | 2021-09-24 |
| vision          | <a href=https://keras.io/examples/vision/zero_dce>Zero-DCE for low-light image enhancement</a>                                                               | 2021-09-18 | 2021-09-19 |
| graph           | <a href=https://keras.io/examples/graph/gat_node_classification>Graph attention network (GAT) for node classification</a>                                    | 2021-09-13 | 2021-12-26 |
| vision          | <a href=https://keras.io/examples/vision/nnclr>Self-supervised contrastive learning with NNCLR</a>                                                           | 2021-09-13 | 2021-09-13 |
| vision          | <a href=https://keras.io/examples/vision/mirnet>Low-light image enhancement using MIRNet</a>                                                                 | 2021-09-11 | 2021-09-15 |
| vision          | <a href=https://keras.io/examples/vision/near_dup_search>Near-duplicate image search</a>                                                                     | 2021-09-10 | 2021-09-10 |
| vision          | <a href=https://keras.io/examples/vision/swin_transformers>Image classification with Swin Transformers</a>                                                   | 2021-09-08 | 2021-09-08 |
| vision          | <a href=https://keras.io/examples/vision/deeplabv3_plus>Multiclass semantic segmentation using DeepLabV3+</a>                                                | 2021-08-31 | 2021-09-01 |
| vision          | <a href=https://keras.io/examples/vision/depth_estimation>Monocular depth estimation</a>                                                                     | 2021-08-30 | 2021-08-30 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/tensorflow_nu_models>Writing Keras Models With TensorFlow NumPy</a>                                          | 2021-08-28 | 2021-08-28 |
| vision          | <a href=https://keras.io/examples/vision/handwriting_recognition>Handwriting recognition</a>                                                                 | 2021-08-16 | 2021-08-16 |
| vision          | <a href=https://keras.io/examples/vision/attention_mil_classification>Classification using Attention-based Deep Multiple Instance Learning (MIL).</a>        | 2021-08-16 | 2021-11-25 |
| graph           | <a href=https://keras.io/examples/graph/mpnn-molecular-graphs>Message-passing neural network (MPNN) for molecular property prediction</a>                    | 2021-08-16 | 2021-12-27 |
| vision          | <a href=https://keras.io/examples/vision/nerf>3D volumetric rendering with NeRF</a>                                                                          | 2021-08-09 | 2021-08-09 |
| nlp             | <a href=https://keras.io/examples/nlp/multimodal_entailment>Multimodal entailment</a>                                                                        | 2021-08-08 | 2021-08-15 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/better_knowledge_distillation>Knowledge distillation recipes</a>                                             | 2021-08-01 | 2021-08-01 |
| vision          | <a href=https://keras.io/examples/vision/involution>Involutional neural networks</a>                                                                         | 2021-07-25 | 2021-07-25 |
| generative      | <a href=https://keras.io/examples/generative/vq_vae>Vector-Quantized Variational Autoencoders</a>                                                            | 2021-07-21 | 2021-07-21 |
| generative      | <a href=https://keras.io/examples/generative/conditional_gan>Conditional GAN</a>                                                                             | 2021-07-13 | 2021-07-15 |
| generative      | <a href=https://keras.io/examples/generative/stylegan>Face image generation with StyleGAN</a>                                                                | 2021-07-01 | 2021-07-01 |
| generative      | <a href=https://keras.io/examples/generative/wgan-graphs>WGAN-GP with R-GCN for the generation of small molecular graphs</a>                                 | 2021-06-30 | 2021-06-30 |
| vision          | <a href=https://keras.io/examples/vision/cct>Compact Convolutional Transformers</a>                                                                          | 2021-06-30 | 2021-06-30 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_classification_transformer>Timeseries classification with a Transformer model</a>                    | 2021-06-25 | 2021-08-05 |
| rl              | <a href=https://keras.io/examples/rl/ppo_cartpole>Proximal Policy Optimization</a>                                                                           | 2021-06-24 | 2021-06-24 |
| nlp             | <a href=https://keras.io/examples/nlp/ner_transformers>Named Entity Recognition using Transformers</a>                                                       | 2021-06-23 | 2021-06-24 |
| vision          | <a href=https://keras.io/examples/vision/adamatch>Semi-supervision and domain adaptation with AdaMatch</a>                                                   | 2021-06-19 | 2021-06-19 |
| vision          | <a href=https://keras.io/examples/vision/gradient_centralization>Gradient Centralization for Better Training Performance</a>                                 | 2021-06-18 | 2021-06-18 |
| vision          | <a href=https://keras.io/examples/vision/video_transformers>Video Classification with Transformers</a>                                                       | 2021-06-08 | 2021-06-08 |
| vision          | <a href=https://keras.io/examples/vision/cutmix>CutMix data augmentation for image classification</a>                                                        | 2021-06-08 | 2021-06-08 |
| vision          | <a href=https://keras.io/examples/vision/conv_lstm>Next-Frame Video Prediction with Convolutional LSTMs</a>                                                  | 2021-06-02 | 2021-06-05 |
| vision          | <a href=https://keras.io/examples/vision/mlp_image_classification>Image classification with modern MLP models</a>                                            | 2021-05-30 | 2021-05-30 |
| graph           | <a href=https://keras.io/examples/graph/gnn_citations>Node Classification with Graph Neural Networks</a>                                                     | 2021-05-30 | 2021-05-30 |
| vision          | <a href=https://keras.io/examples/vision/image_captioning>Image Captioning</a>                                                                               | 2021-05-29 | 2021-10-31 |
| vision          | <a href=https://keras.io/examples/vision/video_classification>Video Classification with a CNN-RNN Architecture</a>                                           | 2021-05-28 | 2021-06-05 |
| nlp             | <a href=https://keras.io/examples/nlp/neural_machine_translation_with_transformer>English-to-Spanish translation with a sequence-to-sequence Transformer</a> | 2021-05-26 | 2021-05-26 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/sample_size_estimate>Estimating required sample size for model training</a>                                  | 2021-05-20 | 2021-06-06 |
| graph           | <a href=https://keras.io/examples/graph/node2vec_movielens>Graph representation learning with node2vec</a>                                                   | 2021-05-15 | 2021-05-15 |
| vision          | <a href=https://keras.io/examples/vision/siamese_contrastive>Image similarity estimation using a Siamese Network with a contrastive loss</a>                 | 2021-05-06 | 2021-05-06 |
| vision          | <a href=https://keras.io/examples/vision/keypoint_detection>Keypoint Detection with Transfer Learning</a>                                                    | 2021-05-02 | 2021-05-02 |
| vision          | <a href=https://keras.io/examples/vision/learnable_resizer>Learning to Resize in Computer Vision</a>                                                         | 2021-04-30 | 2021-05-13 |
| vision          | <a href=https://keras.io/examples/vision/perceiver_image_classification>Image classification with Perceiver</a>                                              | 2021-04-30 | 2021-01-30 |
| vision          | <a href=https://keras.io/examples/vision/semisupervised_simclr>Semi-supervised image classification using contrastive pretraining with SimCLR</a>            | 2021-04-24 | 2021-04-24 |
| vision          | <a href=https://keras.io/examples/vision/consistency_training>Consistency training with supervision</a>                                                      | 2021-04-13 | 2021-04-19 |
| vision          | <a href=https://keras.io/examples/vision/siamese_network>Image similarity estimation using a Siamese Network with a triplet loss</a>                         | 2021-03-25 | 2021-03-25 |
| vision          | <a href=https://keras.io/examples/vision/simsiam>Self-supervised contrastive learning with SimSiam</a>                                                       | 2021-03-19 | 2021-03-20 |
| vision          | <a href=https://keras.io/examples/vision/randaugment>RandAugment for Image Classification for Improved Robustness</a>                                        | 2021-03-13 | 2021-03-17 |
| vision          | <a href=https://keras.io/examples/vision/mixup>MixUp augmentation for image classification</a>                                                               | 2021-03-06 | 2021-03-06 |
| vision          | <a href=https://keras.io/examples/vision/autoencoder>Convolutional autoencoder for image denoising</a>                                                       | 2021-03-01 | 2021-03-01 |
| vision          | <a href=https://keras.io/examples/vision/semantic_image_clustering>Semantic Image Clustering</a>                                                             | 2021-02-28 | 2021-02-28 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/creating_tfrecords>Creating TFRecords</a>                                                                    | 2021-02-27 | 2021-02-27 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/memory_efficient_embeddings>Memory-efficient embeddings for recommendation systems</a>                       | 2021-02-15 | 2021-02-15 |
| structured_data | <a href=https://keras.io/examples/structured_data/classification_with_grn_and_vsn>Classification with Gated Residual and Variable Selection Networks</a>     | 2021-02-10 | 2021-02-10 |
| audio           | <a href=https://keras.io/examples/audio/melgan_spectrogram_inversion>MelGAN-based spectrogram inversion using feature matching</a>                           | 2021-02-09 | 2021-09-15 |
| nlp             | <a href=https://keras.io/examples/nlp/nl_image_search>Natural language image search with a Dual Encoder</a>                                                  | 2021-01-30 | 2021-01-30 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_with_vision_transformer>Image classification with Vision Transformer</a>                       | 2021-01-18 | 2021-01-18 |
| structured_data | <a href=https://keras.io/examples/structured_data/deep_neural_decision_forests>Classification with Neural Decision Forests</a>                               | 2021-01-15 | 2021-01-15 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/bayesian_neural_networks>Probabilistic Bayesian Neural Networks</a>                                          | 2021-01-15 | 2021-01-15 |
| audio           | <a href=https://keras.io/examples/audio/transformer_asr>Automatic Speech Recognition with Transformer</a>                                                    | 2021-01-13 | 2021-01-13 |
| structured_data | <a href=https://keras.io/examples/structured_data/wide_deep_cross_networks>Structured data learning with Wide, Deep, and Cross networks</a>                  | 2020-12-31 | 2021-05-05 |
| structured_data | <a href=https://keras.io/examples/structured_data/movielens_recommendations_transformers>A Transformer-based recommendation system</a>                       | 2020-12-30 | 2020-12-30 |
| vision          | <a href=https://keras.io/examples/vision/supervised-contrastive-learning>Supervised Contrastive Learning</a>                                                 | 2020-11-30 | 2020-11-30 |
| vision          | <a href=https://keras.io/examples/vision/pointnet_segmentation>Point cloud segmentation with PointNet</a>                                                    | 2020-10-23 | 2020-10-24 |
| nlp             | <a href=https://keras.io/examples/nlp/multi_label_classification>Large-scale multi-label text classification</a>                                             | 2020-09-25 | 2020-12-23 |
| vision          | <a href=https://keras.io/examples/vision/3D_image_classification>3D image classification from CT scans</a>                                                   | 2020-09-23 | 2020-09-23 |
| nlp             | <a href=https://keras.io/examples/nlp/masked_language_modeling>End-to-end Masked Language Modeling with BERT</a>                                             | 2020-09-18 | 2020-09-18 |
| vision          | <a href=https://keras.io/examples/vision/knowledge_distillation>Knowledge Distillation</a>                                                                   | 2020-09-01 | 2020-09-01 |
| nlp             | <a href=https://keras.io/examples/nlp/semantic_similarity_with_bert>Semantic Similarity with BERT</a>                                                        | 2020-08-15 | 2020-08-29 |
| generative      | <a href=https://keras.io/examples/generative/cyclegan>CycleGAN</a>                                                                                           | 2020-08-12 | 2020-08-12 |
| generative      | <a href=https://keras.io/examples/generative/real_nvp>Density estimation using Real NVP</a>                                                                  | 2020-08-10 | 2020-08-10 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/tfrecord>How to train a Keras model on TFRecord files</a>                                                    | 2020-07-29 | 2020-08-07 |
| vision          | <a href=https://keras.io/examples/vision/xray_classification_with_tpus>Pneumonia Classification on TPU</a>                                                   | 2020-07-28 | 2020-08-24 |
| vision          | <a href=https://keras.io/examples/vision/super_resolution_sub_pixel>Image Super-Resolution using an Efficient Sub-Pixel CNN</a>                              | 2020-07-28 | 2020-08-27 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_classification_from_scratch>Timeseries classification from scratch</a>                               | 2020-07-21 | 2021-07-16 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning>Image classification via fine-tuning with EfficientNet</a>            | 2020-06-30 | 2020-07-16 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_weather_forecasting>Timeseries forecasting for weather prediction</a>                                | 2020-06-23 | 2020-07-20 |
| vision          | <a href=https://keras.io/examples/vision/captcha_ocr>OCR model for reading Captchas</a>                                                                      | 2020-06-14 | 2020-06-26 |
| audio           | <a href=https://keras.io/examples/audio/speaker_recognition_using_cnn>Speaker Recognition</a>                                                                | 2020-06-14 | 2020-03-07 |
| structured_data | <a href=https://keras.io/examples/structured_data/structured_data_classification_from_scratch>Structured data classification from scratch</a>                | 2020-06-09 | 2020-06-09 |
| vision          | <a href=https://keras.io/examples/vision/metric_learning>Metric learning for image similarity search</a>                                                     | 2020-06-05 | 2020-06-09 |
| rl              | <a href=https://keras.io/examples/rl/ddpg_pendulum>Deep Deterministic Policy Gradient (DDPG)</a>                                                             | 2020-06-04 | 2020-09-21 |
| vision          | <a href=https://keras.io/examples/vision/integrated_gradients>Model interpretability with Integrated Gradients</a>                                           | 2020-06-02 | 2020-06-02 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_anomaly_detection>Timeseries anomaly detection using an Autoencoder</a>                              | 2020-05-31 | 2020-05-31 |
| generative      | <a href=https://keras.io/examples/generative/text_generation_with_miniature_gpt>Text generation with a miniature GPT</a>                                     | 2020-05-29 | 2020-05-29 |
| vision          | <a href=https://keras.io/examples/vision/visualizing_what_convnets_learn>Visualizing what convnets learn</a>                                                 | 2020-05-29 | 2020-05-29 |
| vision          | <a href=https://keras.io/examples/vision/pointnet>Point cloud classification with PointNet</a>                                                               | 2020-05-25 | 2020-05-26 |
| structured_data | <a href=https://keras.io/examples/structured_data/collaborative_filtering_movielens>Collaborative Filtering for Movie Recommendations</a>                    | 2020-05-24 | 2020-05-24 |
| rl              | <a href=https://keras.io/examples/rl/deep_q_network_breakout>Deep Q-Learning for Atari Breakout</a>                                                          | 2020-05-23 | 2020-06-17 |
| nlp             | <a href=https://keras.io/examples/nlp/text_extraction_with_bert>Text Extraction with BERT</a>                                                                | 2020-05-23 | 2020-05-23 |
| vision          | <a href=https://keras.io/examples/vision/reptile>Few-Shot learning with Reptile</a>                                                                          | 2020-05-21 | 2020-05-30 |
| vision          | <a href=https://keras.io/examples/vision/retinanet>Object Detection with RetinaNet</a>                                                                       | 2020-05-17 | 2020-07-14 |
| generative      | <a href=https://keras.io/examples/generative/pixelcnn>PixelCNN</a>                                                                                           | 2020-05-17 | 2020-05-23 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/debugging_tips>Keras debugging tips</a>                                                                      | 2020-05-16 | 2020-05-16 |
| rl              | <a href=https://keras.io/examples/rl/actor_critic_cartpole>Actor Critic Method</a>                                                                           | 2020-05-13 | 2020-05-13 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_with_switch_transformer>Text classification with Switch Transformer</a>                            | 2020-05-10 | 2021-02-15 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_with_transformer>Text classification with Transformer</a>                                          | 2020-05-10 | 2020-05-10 |
| generative      | <a href=https://keras.io/examples/generative/wgan_gp>WGAN-GP overriding `Model.train_step`</a>                                                               | 2020-05-09 | 2020-05-09 |
| nlp             | <a href=https://keras.io/examples/nlp/pretrained_word_embeddings>Using pre-trained word embeddings</a>                                                       | 2020-05-05 | 2020-05-05 |
| generative      | <a href=https://keras.io/examples/generative/vae>Variational AutoEncoder</a>                                                                                 | 2020-05-03 | 2020-05-03 |
| nlp             | <a href=https://keras.io/examples/nlp/bidirectional_lstm_imdb>Bidirectional LSTM on IMDB</a>                                                                 | 2020-05-03 | 2020-05-03 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_from_scratch>Image classification from scratch</a>                                             | 2020-04-27 | 2020-04-28 |
| vision          | <a href=https://keras.io/examples/vision/grad_cam>Grad-CAM class activation visualization</a>                                                                | 2020-04-26 | 2021-03-07 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/quasi_svm>A Quasi-SVM in Keras</a>                                                                           | 2020-04-17 | 2020-04-17 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_from_scratch>Text classification from scratch</a>                                                  | 2019-11-06 | 2020-05-17 |
| structured_data | <a href=https://keras.io/examples/structured_data/imbalanced_classification>Imbalanced classification: credit card fraud detection</a>                       | 2019-05-28 | 2020-04-17 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/endpoint_layer_pattern>Endpoint layer pattern</a>                                                            | 2019-05-10 | 2019-05-10 |
| generative      | <a href=https://keras.io/examples/generative/dcgan_overriding_train_step>DCGAN to generate face images</a>                                                   | 2019-04-29 | 2021-01-01 |
| vision          | <a href=https://keras.io/examples/vision/oxford_pets_image_segmentation>Image segmentation with a U-Net-like architecture</a>                                | 2019-03-20 | 2020-04-20 |
| nlp             | <a href=https://keras.io/examples/nlp/lstm_seq2seq>Character-level recurrent sequence-to-sequence model</a>                                                  | 2017-09-29 | 2020-04-26 |
| generative      | <a href=https://keras.io/examples/generative/deep_dream>Deep Dream</a>                                                                                       | 2016-01-13 | 2020-05-02 |
| generative      | <a href=https://keras.io/examples/generative/neural_style_transfer>Neural style transfer</a>                                                                 | 2016-01-11 | 2020-05-02 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/antirectifier>Simple custom layer example: Antirectifier</a>                                                 | 2016-01-06 | 2020-04-20 |
| nlp             | <a href=https://keras.io/examples/nlp/addition_rnn>Sequence to sequence learning for performing number addition</a>                                          | 2015-08-17 | 2020-04-17 |
| vision          | <a href=https://keras.io/examples/vision/mnist_convnet>Simple MNIST convnet</a>                                                                              | 2015-06-19 | 2020-04-21 |
| generative      | <a href=https://keras.io/examples/generative/lstm_character_level_text_generation>Character-level text generation with LSTM</a>                              | 2015-06-15 | 2020-04-30 |


```python
# sorted by 'Last modified'

sorted_report_df = report_df.sort_values(by=['Last modified'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```


| Category        | Title                                                                                                                                                        | Date created        | Last modified       |
|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:--------------------|
| structured_data | <a href=https://keras.io/examples/structured_data/classification_with_tfdf>Classification with TensorFlow Decision Forests</a>                               | 2022-01-25 | 2022-01-25 |
| vision          | <a href=https://keras.io/examples/vision/patch_convnet>Augmenting convnets with aggregated attention</a>                                                     | 2022-01-22 | 2022-01-22 |
| structured_data | <a href=https://keras.io/examples/structured_data/tabtransformer>Structured data learning with TabTransformer</a>                                            | 2022-01-18 | 2022-01-18 |
| nlp             | <a href=https://keras.io/examples/nlp/question_answering>Question Answering with Hugging Face Transformers</a>                                               | 2022-01-13 | 2022-01-13 |
| vision          | <a href=https://keras.io/examples/vision/vivit>Video Vision Transformer</a>                                                                                  | 2022-01-12 | 2022-01-12 |
| vision          | <a href=https://keras.io/examples/vision/vit_small_ds>Train a Vision Transformer on small datasets</a>                                                       | 2022-01-07 | 2022-01-10 |
| generative      | <a href=https://keras.io/examples/generative/gaugan>GauGAN for conditional image generation</a>                                                              | 2021-12-26 | 2022-01-03 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_traffic_forecasting>Traffic forecasting using graph neural networks and LSTM</a>                     | 2021-12-28 | 2021-12-28 |
| graph           | <a href=https://keras.io/examples/graph/mpnn-molecular-graphs>Message-passing neural network (MPNN) for molecular property prediction</a>                    | 2021-08-16 | 2021-12-27 |
| graph           | <a href=https://keras.io/examples/graph/gat_node_classification>Graph attention network (GAT) for node classification</a>                                    | 2021-09-13 | 2021-12-26 |
| vision          | <a href=https://keras.io/examples/vision/masked_image_modeling>Masked image modeling with Autoencoders</a>                                                   | 2021-12-20 | 2021-12-21 |
| vision          | <a href=https://keras.io/examples/vision/barlow_twins>Barlow Twins for Contrastive SSL</a>                                                                   | 2021-11-04 | 2021-12-20 |
| vision          | <a href=https://keras.io/examples/vision/token_learner>Learning to tokenize in Vision Transformers</a>                                                       | 2021-12-10 | 2021-12-15 |
| vision          | <a href=https://keras.io/examples/vision/attention_mil_classification>Classification using Attention-based Deep Multiple Instance Learning (MIL).</a>        | 2021-08-16 | 2021-11-25 |
| generative      | <a href=https://keras.io/examples/generative/adain>Neural Style Transfer with AdaIN</a>                                                                      | 2021-11-08 | 2021-11-08 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/subclassing_conv_layers>Customizing the convolution operation of a Conv2D layer</a>                          | 2021-11-03 | 2021-11-03 |
| vision          | <a href=https://keras.io/examples/vision/image_captioning>Image Captioning</a>                                                                               | 2021-05-29 | 2021-10-31 |
| nlp             | <a href=https://keras.io/examples/nlp/active_learning_review_classification>Review Classification using Active Learning</a>                                  | 2021-10-29 | 2021-10-29 |
| generative      | <a href=https://keras.io/examples/generative/gan_ada>Data-efficient GANs with Adaptive Discriminator Augmentation</a>                                        | 2021-10-28 | 2021-10-28 |
| vision          | <a href=https://keras.io/examples/vision/mobilevit>MobileViT: A mobile-friendly Transformer-based model for image classification</a>                         | 2021-10-20 | 2021-10-20 |
| vision          | <a href=https://keras.io/examples/vision/eanet>Image classification with EANet (External Attention Transformer)</a>                                          | 2021-10-19 | 2021-10-19 |
| vision          | <a href=https://keras.io/examples/vision/convmixer>Image classification with ConvMixer</a>                                                                   | 2021-10-12 | 2021-10-12 |
| vision          | <a href=https://keras.io/examples/vision/fixres>FixRes: Fixing train-test resolution discrepancy</a>                                                         | 2021-10-08 | 2021-10-10 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/sklearn_metric_callbacks>Evaluating and exporting scikit-learn metrics in a Keras callback</a>               | 2021-10-07 | 2021-10-07 |
| nlp             | <a href=https://keras.io/examples/nlp/text_generation_fnet>Text Generation using FNet</a>                                                                    | 2021-10-05 | 2021-10-05 |
| vision          | <a href=https://keras.io/examples/vision/metric_learning_tf_similarity>Metric learning for image similarity search using TensorFlow Similarity</a>           | 2021-09-30 | 2021-09-30 |
| audio           | <a href=https://keras.io/examples/audio/ctc_asr>Automatic Speech Recognition using CTC</a>                                                                   | 2021-09-26 | 2021-09-26 |
| vision          | <a href=https://keras.io/examples/vision/bit>Image Classification using BigTransfer (BiT)</a>                                                                | 2021-09-24 | 2021-09-24 |
| vision          | <a href=https://keras.io/examples/vision/zero_dce>Zero-DCE for low-light image enhancement</a>                                                               | 2021-09-18 | 2021-09-19 |
| audio           | <a href=https://keras.io/examples/audio/melgan_spectrogram_inversion>MelGAN-based spectrogram inversion using feature matching</a>                           | 2021-02-09 | 2021-09-15 |
| vision          | <a href=https://keras.io/examples/vision/mirnet>Low-light image enhancement using MIRNet</a>                                                                 | 2021-09-11 | 2021-09-15 |
| vision          | <a href=https://keras.io/examples/vision/nnclr>Self-supervised contrastive learning with NNCLR</a>                                                           | 2021-09-13 | 2021-09-13 |
| vision          | <a href=https://keras.io/examples/vision/near_dup_search>Near-duplicate image search</a>                                                                     | 2021-09-10 | 2021-09-10 |
| vision          | <a href=https://keras.io/examples/vision/swin_transformers>Image classification with Swin Transformers</a>                                                   | 2021-09-08 | 2021-09-08 |
| vision          | <a href=https://keras.io/examples/vision/deeplabv3_plus>Multiclass semantic segmentation using DeepLabV3+</a>                                                | 2021-08-31 | 2021-09-01 |
| vision          | <a href=https://keras.io/examples/vision/depth_estimation>Monocular depth estimation</a>                                                                     | 2021-08-30 | 2021-08-30 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/tensorflow_nu_models>Writing Keras Models With TensorFlow NumPy</a>                                          | 2021-08-28 | 2021-08-28 |
| vision          | <a href=https://keras.io/examples/vision/handwriting_recognition>Handwriting recognition</a>                                                                 | 2021-08-16 | 2021-08-16 |
| nlp             | <a href=https://keras.io/examples/nlp/multimodal_entailment>Multimodal entailment</a>                                                                        | 2021-08-08 | 2021-08-15 |
| vision          | <a href=https://keras.io/examples/vision/nerf>3D volumetric rendering with NeRF</a>                                                                          | 2021-08-09 | 2021-08-09 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_classification_transformer>Timeseries classification with a Transformer model</a>                    | 2021-06-25 | 2021-08-05 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/better_knowledge_distillation>Knowledge distillation recipes</a>                                             | 2021-08-01 | 2021-08-01 |
| vision          | <a href=https://keras.io/examples/vision/involution>Involutional neural networks</a>                                                                         | 2021-07-25 | 2021-07-25 |
| generative      | <a href=https://keras.io/examples/generative/vq_vae>Vector-Quantized Variational Autoencoders</a>                                                            | 2021-07-21 | 2021-07-21 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_classification_from_scratch>Timeseries classification from scratch</a>                               | 2020-07-21 | 2021-07-16 |
| generative      | <a href=https://keras.io/examples/generative/conditional_gan>Conditional GAN</a>                                                                             | 2021-07-13 | 2021-07-15 |
| generative      | <a href=https://keras.io/examples/generative/stylegan>Face image generation with StyleGAN</a>                                                                | 2021-07-01 | 2021-07-01 |
| generative      | <a href=https://keras.io/examples/generative/wgan-graphs>WGAN-GP with R-GCN for the generation of small molecular graphs</a>                                 | 2021-06-30 | 2021-06-30 |
| vision          | <a href=https://keras.io/examples/vision/cct>Compact Convolutional Transformers</a>                                                                          | 2021-06-30 | 2021-06-30 |
| rl              | <a href=https://keras.io/examples/rl/ppo_cartpole>Proximal Policy Optimization</a>                                                                           | 2021-06-24 | 2021-06-24 |
| nlp             | <a href=https://keras.io/examples/nlp/ner_transformers>Named Entity Recognition using Transformers</a>                                                       | 2021-06-23 | 2021-06-24 |
| vision          | <a href=https://keras.io/examples/vision/adamatch>Semi-supervision and domain adaptation with AdaMatch</a>                                                   | 2021-06-19 | 2021-06-19 |
| vision          | <a href=https://keras.io/examples/vision/gradient_centralization>Gradient Centralization for Better Training Performance</a>                                 | 2021-06-18 | 2021-06-18 |
| vision          | <a href=https://keras.io/examples/vision/cutmix>CutMix data augmentation for image classification</a>                                                        | 2021-06-08 | 2021-06-08 |
| vision          | <a href=https://keras.io/examples/vision/video_transformers>Video Classification with Transformers</a>                                                       | 2021-06-08 | 2021-06-08 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/sample_size_estimate>Estimating required sample size for model training</a>                                  | 2021-05-20 | 2021-06-06 |
| vision          | <a href=https://keras.io/examples/vision/video_classification>Video Classification with a CNN-RNN Architecture</a>                                           | 2021-05-28 | 2021-06-05 |
| vision          | <a href=https://keras.io/examples/vision/conv_lstm>Next-Frame Video Prediction with Convolutional LSTMs</a>                                                  | 2021-06-02 | 2021-06-05 |
| graph           | <a href=https://keras.io/examples/graph/gnn_citations>Node Classification with Graph Neural Networks</a>                                                     | 2021-05-30 | 2021-05-30 |
| vision          | <a href=https://keras.io/examples/vision/mlp_image_classification>Image classification with modern MLP models</a>                                            | 2021-05-30 | 2021-05-30 |
| nlp             | <a href=https://keras.io/examples/nlp/neural_machine_translation_with_transformer>English-to-Spanish translation with a sequence-to-sequence Transformer</a> | 2021-05-26 | 2021-05-26 |
| graph           | <a href=https://keras.io/examples/graph/node2vec_movielens>Graph representation learning with node2vec</a>                                                   | 2021-05-15 | 2021-05-15 |
| vision          | <a href=https://keras.io/examples/vision/learnable_resizer>Learning to Resize in Computer Vision</a>                                                         | 2021-04-30 | 2021-05-13 |
| vision          | <a href=https://keras.io/examples/vision/siamese_contrastive>Image similarity estimation using a Siamese Network with a contrastive loss</a>                 | 2021-05-06 | 2021-05-06 |
| structured_data | <a href=https://keras.io/examples/structured_data/wide_deep_cross_networks>Structured data learning with Wide, Deep, and Cross networks</a>                  | 2020-12-31 | 2021-05-05 |
| vision          | <a href=https://keras.io/examples/vision/keypoint_detection>Keypoint Detection with Transfer Learning</a>                                                    | 2021-05-02 | 2021-05-02 |
| vision          | <a href=https://keras.io/examples/vision/semisupervised_simclr>Semi-supervised image classification using contrastive pretraining with SimCLR</a>            | 2021-04-24 | 2021-04-24 |
| vision          | <a href=https://keras.io/examples/vision/consistency_training>Consistency training with supervision</a>                                                      | 2021-04-13 | 2021-04-19 |
| vision          | <a href=https://keras.io/examples/vision/siamese_network>Image similarity estimation using a Siamese Network with a triplet loss</a>                         | 2021-03-25 | 2021-03-25 |
| vision          | <a href=https://keras.io/examples/vision/simsiam>Self-supervised contrastive learning with SimSiam</a>                                                       | 2021-03-19 | 2021-03-20 |
| vision          | <a href=https://keras.io/examples/vision/randaugment>RandAugment for Image Classification for Improved Robustness</a>                                        | 2021-03-13 | 2021-03-17 |
| vision          | <a href=https://keras.io/examples/vision/grad_cam>Grad-CAM class activation visualization</a>                                                                | 2020-04-26 | 2021-03-07 |
| vision          | <a href=https://keras.io/examples/vision/mixup>MixUp augmentation for image classification</a>                                                               | 2021-03-06 | 2021-03-06 |
| vision          | <a href=https://keras.io/examples/vision/autoencoder>Convolutional autoencoder for image denoising</a>                                                       | 2021-03-01 | 2021-03-01 |
| vision          | <a href=https://keras.io/examples/vision/semantic_image_clustering>Semantic Image Clustering</a>                                                             | 2021-02-28 | 2021-02-28 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/creating_tfrecords>Creating TFRecords</a>                                                                    | 2021-02-27 | 2021-02-27 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/memory_efficient_embeddings>Memory-efficient embeddings for recommendation systems</a>                       | 2021-02-15 | 2021-02-15 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_with_switch_transformer>Text classification with Switch Transformer</a>                            | 2020-05-10 | 2021-02-15 |
| structured_data | <a href=https://keras.io/examples/structured_data/classification_with_grn_and_vsn>Classification with Gated Residual and Variable Selection Networks</a>     | 2021-02-10 | 2021-02-10 |
| nlp             | <a href=https://keras.io/examples/nlp/nl_image_search>Natural language image search with a Dual Encoder</a>                                                  | 2021-01-30 | 2021-01-30 |
| vision          | <a href=https://keras.io/examples/vision/perceiver_image_classification>Image classification with Perceiver</a>                                              | 2021-04-30 | 2021-01-30 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_with_vision_transformer>Image classification with Vision Transformer</a>                       | 2021-01-18 | 2021-01-18 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/bayesian_neural_networks>Probabilistic Bayesian Neural Networks</a>                                          | 2021-01-15 | 2021-01-15 |
| structured_data | <a href=https://keras.io/examples/structured_data/deep_neural_decision_forests>Classification with Neural Decision Forests</a>                               | 2021-01-15 | 2021-01-15 |
| audio           | <a href=https://keras.io/examples/audio/transformer_asr>Automatic Speech Recognition with Transformer</a>                                                    | 2021-01-13 | 2021-01-13 |
| generative      | <a href=https://keras.io/examples/generative/dcgan_overriding_train_step>DCGAN to generate face images</a>                                                   | 2019-04-29 | 2021-01-01 |
| structured_data | <a href=https://keras.io/examples/structured_data/movielens_recommendations_transformers>A Transformer-based recommendation system</a>                       | 2020-12-30 | 2020-12-30 |
| nlp             | <a href=https://keras.io/examples/nlp/multi_label_classification>Large-scale multi-label text classification</a>                                             | 2020-09-25 | 2020-12-23 |
| vision          | <a href=https://keras.io/examples/vision/supervised-contrastive-learning>Supervised Contrastive Learning</a>                                                 | 2020-11-30 | 2020-11-30 |
| vision          | <a href=https://keras.io/examples/vision/pointnet_segmentation>Point cloud segmentation with PointNet</a>                                                    | 2020-10-23 | 2020-10-24 |
| vision          | <a href=https://keras.io/examples/vision/3D_image_classification>3D image classification from CT scans</a>                                                   | 2020-09-23 | 2020-09-23 |
| rl              | <a href=https://keras.io/examples/rl/ddpg_pendulum>Deep Deterministic Policy Gradient (DDPG)</a>                                                             | 2020-06-04 | 2020-09-21 |
| nlp             | <a href=https://keras.io/examples/nlp/masked_language_modeling>End-to-end Masked Language Modeling with BERT</a>                                             | 2020-09-18 | 2020-09-18 |
| vision          | <a href=https://keras.io/examples/vision/knowledge_distillation>Knowledge Distillation</a>                                                                   | 2020-09-01 | 2020-09-01 |
| nlp             | <a href=https://keras.io/examples/nlp/semantic_similarity_with_bert>Semantic Similarity with BERT</a>                                                        | 2020-08-15 | 2020-08-29 |
| vision          | <a href=https://keras.io/examples/vision/super_resolution_sub_pixel>Image Super-Resolution using an Efficient Sub-Pixel CNN</a>                              | 2020-07-28 | 2020-08-27 |
| vision          | <a href=https://keras.io/examples/vision/xray_classification_with_tpus>Pneumonia Classification on TPU</a>                                                   | 2020-07-28 | 2020-08-24 |
| generative      | <a href=https://keras.io/examples/generative/cyclegan>CycleGAN</a>                                                                                           | 2020-08-12 | 2020-08-12 |
| generative      | <a href=https://keras.io/examples/generative/real_nvp>Density estimation using Real NVP</a>                                                                  | 2020-08-10 | 2020-08-10 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/tfrecord>How to train a Keras model on TFRecord files</a>                                                    | 2020-07-29 | 2020-08-07 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_weather_forecasting>Timeseries forecasting for weather prediction</a>                                | 2020-06-23 | 2020-07-20 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning>Image classification via fine-tuning with EfficientNet</a>            | 2020-06-30 | 2020-07-16 |
| vision          | <a href=https://keras.io/examples/vision/retinanet>Object Detection with RetinaNet</a>                                                                       | 2020-05-17 | 2020-07-14 |
| vision          | <a href=https://keras.io/examples/vision/captcha_ocr>OCR model for reading Captchas</a>                                                                      | 2020-06-14 | 2020-06-26 |
| rl              | <a href=https://keras.io/examples/rl/deep_q_network_breakout>Deep Q-Learning for Atari Breakout</a>                                                          | 2020-05-23 | 2020-06-17 |
| vision          | <a href=https://keras.io/examples/vision/metric_learning>Metric learning for image similarity search</a>                                                     | 2020-06-05 | 2020-06-09 |
| structured_data | <a href=https://keras.io/examples/structured_data/structured_data_classification_from_scratch>Structured data classification from scratch</a>                | 2020-06-09 | 2020-06-09 |
| vision          | <a href=https://keras.io/examples/vision/integrated_gradients>Model interpretability with Integrated Gradients</a>                                           | 2020-06-02 | 2020-06-02 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_anomaly_detection>Timeseries anomaly detection using an Autoencoder</a>                              | 2020-05-31 | 2020-05-31 |
| vision          | <a href=https://keras.io/examples/vision/reptile>Few-Shot learning with Reptile</a>                                                                          | 2020-05-21 | 2020-05-30 |
| vision          | <a href=https://keras.io/examples/vision/visualizing_what_convnets_learn>Visualizing what convnets learn</a>                                                 | 2020-05-29 | 2020-05-29 |
| generative      | <a href=https://keras.io/examples/generative/text_generation_with_miniature_gpt>Text generation with a miniature GPT</a>                                     | 2020-05-29 | 2020-05-29 |
| vision          | <a href=https://keras.io/examples/vision/pointnet>Point cloud classification with PointNet</a>                                                               | 2020-05-25 | 2020-05-26 |
| structured_data | <a href=https://keras.io/examples/structured_data/collaborative_filtering_movielens>Collaborative Filtering for Movie Recommendations</a>                    | 2020-05-24 | 2020-05-24 |
| generative      | <a href=https://keras.io/examples/generative/pixelcnn>PixelCNN</a>                                                                                           | 2020-05-17 | 2020-05-23 |
| nlp             | <a href=https://keras.io/examples/nlp/text_extraction_with_bert>Text Extraction with BERT</a>                                                                | 2020-05-23 | 2020-05-23 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_from_scratch>Text classification from scratch</a>                                                  | 2019-11-06 | 2020-05-17 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/debugging_tips>Keras debugging tips</a>                                                                      | 2020-05-16 | 2020-05-16 |
| rl              | <a href=https://keras.io/examples/rl/actor_critic_cartpole>Actor Critic Method</a>                                                                           | 2020-05-13 | 2020-05-13 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_with_transformer>Text classification with Transformer</a>                                          | 2020-05-10 | 2020-05-10 |
| generative      | <a href=https://keras.io/examples/generative/wgan_gp>WGAN-GP overriding `Model.train_step`</a>                                                               | 2020-05-09 | 2020-05-09 |
| nlp             | <a href=https://keras.io/examples/nlp/pretrained_word_embeddings>Using pre-trained word embeddings</a>                                                       | 2020-05-05 | 2020-05-05 |
| nlp             | <a href=https://keras.io/examples/nlp/bidirectional_lstm_imdb>Bidirectional LSTM on IMDB</a>                                                                 | 2020-05-03 | 2020-05-03 |
| generative      | <a href=https://keras.io/examples/generative/vae>Variational AutoEncoder</a>                                                                                 | 2020-05-03 | 2020-05-03 |
| generative      | <a href=https://keras.io/examples/generative/neural_style_transfer>Neural style transfer</a>                                                                 | 2016-01-11 | 2020-05-02 |
| generative      | <a href=https://keras.io/examples/generative/deep_dream>Deep Dream</a>                                                                                       | 2016-01-13 | 2020-05-02 |
| generative      | <a href=https://keras.io/examples/generative/lstm_character_level_text_generation>Character-level text generation with LSTM</a>                              | 2015-06-15 | 2020-04-30 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_from_scratch>Image classification from scratch</a>                                             | 2020-04-27 | 2020-04-28 |
| nlp             | <a href=https://keras.io/examples/nlp/lstm_seq2seq>Character-level recurrent sequence-to-sequence model</a>                                                  | 2017-09-29 | 2020-04-26 |
| vision          | <a href=https://keras.io/examples/vision/mnist_convnet>Simple MNIST convnet</a>                                                                              | 2015-06-19 | 2020-04-21 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/antirectifier>Simple custom layer example: Antirectifier</a>                                                 | 2016-01-06 | 2020-04-20 |
| vision          | <a href=https://keras.io/examples/vision/oxford_pets_image_segmentation>Image segmentation with a U-Net-like architecture</a>                                | 2019-03-20 | 2020-04-20 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/quasi_svm>A Quasi-SVM in Keras</a>                                                                           | 2020-04-17 | 2020-04-17 |
| structured_data | <a href=https://keras.io/examples/structured_data/imbalanced_classification>Imbalanced classification: credit card fraud detection</a>                       | 2019-05-28 | 2020-04-17 |
| nlp             | <a href=https://keras.io/examples/nlp/addition_rnn>Sequence to sequence learning for performing number addition</a>                                          | 2015-08-17 | 2020-04-17 |
| audio           | <a href=https://keras.io/examples/audio/speaker_recognition_using_cnn>Speaker Recognition</a>                                                                | 2020-06-14 | 2020-03-07 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/endpoint_layer_pattern>Endpoint layer pattern</a>                                                            | 2019-05-10 | 2019-05-10 |




```python
# sorted by 'Category' and 'Date created'

sorted_report_df = report_df.sort_values(by=['Category', 'Date created'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```


| Category        | Title                                                                                                                                                        | Date created        | Last modified       |
|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:--------------------|
| vision          | <a href=https://keras.io/examples/vision/patch_convnet>Augmenting convnets with aggregated attention</a>                                                     | 2022-01-22 | 2022-01-22 |
| vision          | <a href=https://keras.io/examples/vision/vivit>Video Vision Transformer</a>                                                                                  | 2022-01-12 | 2022-01-12 |
| vision          | <a href=https://keras.io/examples/vision/vit_small_ds>Train a Vision Transformer on small datasets</a>                                                       | 2022-01-07 | 2022-01-10 |
| vision          | <a href=https://keras.io/examples/vision/masked_image_modeling>Masked image modeling with Autoencoders</a>                                                   | 2021-12-20 | 2021-12-21 |
| vision          | <a href=https://keras.io/examples/vision/token_learner>Learning to tokenize in Vision Transformers</a>                                                       | 2021-12-10 | 2021-12-15 |
| vision          | <a href=https://keras.io/examples/vision/barlow_twins>Barlow Twins for Contrastive SSL</a>                                                                   | 2021-11-04 | 2021-12-20 |
| vision          | <a href=https://keras.io/examples/vision/mobilevit>MobileViT: A mobile-friendly Transformer-based model for image classification</a>                         | 2021-10-20 | 2021-10-20 |
| vision          | <a href=https://keras.io/examples/vision/eanet>Image classification with EANet (External Attention Transformer)</a>                                          | 2021-10-19 | 2021-10-19 |
| vision          | <a href=https://keras.io/examples/vision/convmixer>Image classification with ConvMixer</a>                                                                   | 2021-10-12 | 2021-10-12 |
| vision          | <a href=https://keras.io/examples/vision/fixres>FixRes: Fixing train-test resolution discrepancy</a>                                                         | 2021-10-08 | 2021-10-10 |
| vision          | <a href=https://keras.io/examples/vision/metric_learning_tf_similarity>Metric learning for image similarity search using TensorFlow Similarity</a>           | 2021-09-30 | 2021-09-30 |
| vision          | <a href=https://keras.io/examples/vision/bit>Image Classification using BigTransfer (BiT)</a>                                                                | 2021-09-24 | 2021-09-24 |
| vision          | <a href=https://keras.io/examples/vision/zero_dce>Zero-DCE for low-light image enhancement</a>                                                               | 2021-09-18 | 2021-09-19 |
| vision          | <a href=https://keras.io/examples/vision/nnclr>Self-supervised contrastive learning with NNCLR</a>                                                           | 2021-09-13 | 2021-09-13 |
| vision          | <a href=https://keras.io/examples/vision/mirnet>Low-light image enhancement using MIRNet</a>                                                                 | 2021-09-11 | 2021-09-15 |
| vision          | <a href=https://keras.io/examples/vision/near_dup_search>Near-duplicate image search</a>                                                                     | 2021-09-10 | 2021-09-10 |
| vision          | <a href=https://keras.io/examples/vision/swin_transformers>Image classification with Swin Transformers</a>                                                   | 2021-09-08 | 2021-09-08 |
| vision          | <a href=https://keras.io/examples/vision/deeplabv3_plus>Multiclass semantic segmentation using DeepLabV3+</a>                                                | 2021-08-31 | 2021-09-01 |
| vision          | <a href=https://keras.io/examples/vision/depth_estimation>Monocular depth estimation</a>                                                                     | 2021-08-30 | 2021-08-30 |
| vision          | <a href=https://keras.io/examples/vision/handwriting_recognition>Handwriting recognition</a>                                                                 | 2021-08-16 | 2021-08-16 |
| vision          | <a href=https://keras.io/examples/vision/attention_mil_classification>Classification using Attention-based Deep Multiple Instance Learning (MIL).</a>        | 2021-08-16 | 2021-11-25 |
| vision          | <a href=https://keras.io/examples/vision/nerf>3D volumetric rendering with NeRF</a>                                                                          | 2021-08-09 | 2021-08-09 |
| vision          | <a href=https://keras.io/examples/vision/involution>Involutional neural networks</a>                                                                         | 2021-07-25 | 2021-07-25 |
| vision          | <a href=https://keras.io/examples/vision/cct>Compact Convolutional Transformers</a>                                                                          | 2021-06-30 | 2021-06-30 |
| vision          | <a href=https://keras.io/examples/vision/adamatch>Semi-supervision and domain adaptation with AdaMatch</a>                                                   | 2021-06-19 | 2021-06-19 |
| vision          | <a href=https://keras.io/examples/vision/gradient_centralization>Gradient Centralization for Better Training Performance</a>                                 | 2021-06-18 | 2021-06-18 |
| vision          | <a href=https://keras.io/examples/vision/cutmix>CutMix data augmentation for image classification</a>                                                        | 2021-06-08 | 2021-06-08 |
| vision          | <a href=https://keras.io/examples/vision/video_transformers>Video Classification with Transformers</a>                                                       | 2021-06-08 | 2021-06-08 |
| vision          | <a href=https://keras.io/examples/vision/conv_lstm>Next-Frame Video Prediction with Convolutional LSTMs</a>                                                  | 2021-06-02 | 2021-06-05 |
| vision          | <a href=https://keras.io/examples/vision/mlp_image_classification>Image classification with modern MLP models</a>                                            | 2021-05-30 | 2021-05-30 |
| vision          | <a href=https://keras.io/examples/vision/image_captioning>Image Captioning</a>                                                                               | 2021-05-29 | 2021-10-31 |
| vision          | <a href=https://keras.io/examples/vision/video_classification>Video Classification with a CNN-RNN Architecture</a>                                           | 2021-05-28 | 2021-06-05 |
| vision          | <a href=https://keras.io/examples/vision/siamese_contrastive>Image similarity estimation using a Siamese Network with a contrastive loss</a>                 | 2021-05-06 | 2021-05-06 |
| vision          | <a href=https://keras.io/examples/vision/keypoint_detection>Keypoint Detection with Transfer Learning</a>                                                    | 2021-05-02 | 2021-05-02 |
| vision          | <a href=https://keras.io/examples/vision/perceiver_image_classification>Image classification with Perceiver</a>                                              | 2021-04-30 | 2021-01-30 |
| vision          | <a href=https://keras.io/examples/vision/learnable_resizer>Learning to Resize in Computer Vision</a>                                                         | 2021-04-30 | 2021-05-13 |
| vision          | <a href=https://keras.io/examples/vision/semisupervised_simclr>Semi-supervised image classification using contrastive pretraining with SimCLR</a>            | 2021-04-24 | 2021-04-24 |
| vision          | <a href=https://keras.io/examples/vision/consistency_training>Consistency training with supervision</a>                                                      | 2021-04-13 | 2021-04-19 |
| vision          | <a href=https://keras.io/examples/vision/siamese_network>Image similarity estimation using a Siamese Network with a triplet loss</a>                         | 2021-03-25 | 2021-03-25 |
| vision          | <a href=https://keras.io/examples/vision/simsiam>Self-supervised contrastive learning with SimSiam</a>                                                       | 2021-03-19 | 2021-03-20 |
| vision          | <a href=https://keras.io/examples/vision/randaugment>RandAugment for Image Classification for Improved Robustness</a>                                        | 2021-03-13 | 2021-03-17 |
| vision          | <a href=https://keras.io/examples/vision/mixup>MixUp augmentation for image classification</a>                                                               | 2021-03-06 | 2021-03-06 |
| vision          | <a href=https://keras.io/examples/vision/autoencoder>Convolutional autoencoder for image denoising</a>                                                       | 2021-03-01 | 2021-03-01 |
| vision          | <a href=https://keras.io/examples/vision/semantic_image_clustering>Semantic Image Clustering</a>                                                             | 2021-02-28 | 2021-02-28 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_with_vision_transformer>Image classification with Vision Transformer</a>                       | 2021-01-18 | 2021-01-18 |
| vision          | <a href=https://keras.io/examples/vision/supervised-contrastive-learning>Supervised Contrastive Learning</a>                                                 | 2020-11-30 | 2020-11-30 |
| vision          | <a href=https://keras.io/examples/vision/pointnet_segmentation>Point cloud segmentation with PointNet</a>                                                    | 2020-10-23 | 2020-10-24 |
| vision          | <a href=https://keras.io/examples/vision/3D_image_classification>3D image classification from CT scans</a>                                                   | 2020-09-23 | 2020-09-23 |
| vision          | <a href=https://keras.io/examples/vision/knowledge_distillation>Knowledge Distillation</a>                                                                   | 2020-09-01 | 2020-09-01 |
| vision          | <a href=https://keras.io/examples/vision/xray_classification_with_tpus>Pneumonia Classification on TPU</a>                                                   | 2020-07-28 | 2020-08-24 |
| vision          | <a href=https://keras.io/examples/vision/super_resolution_sub_pixel>Image Super-Resolution using an Efficient Sub-Pixel CNN</a>                              | 2020-07-28 | 2020-08-27 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning>Image classification via fine-tuning with EfficientNet</a>            | 2020-06-30 | 2020-07-16 |
| vision          | <a href=https://keras.io/examples/vision/captcha_ocr>OCR model for reading Captchas</a>                                                                      | 2020-06-14 | 2020-06-26 |
| vision          | <a href=https://keras.io/examples/vision/metric_learning>Metric learning for image similarity search</a>                                                     | 2020-06-05 | 2020-06-09 |
| vision          | <a href=https://keras.io/examples/vision/integrated_gradients>Model interpretability with Integrated Gradients</a>                                           | 2020-06-02 | 2020-06-02 |
| vision          | <a href=https://keras.io/examples/vision/visualizing_what_convnets_learn>Visualizing what convnets learn</a>                                                 | 2020-05-29 | 2020-05-29 |
| vision          | <a href=https://keras.io/examples/vision/pointnet>Point cloud classification with PointNet</a>                                                               | 2020-05-25 | 2020-05-26 |
| vision          | <a href=https://keras.io/examples/vision/reptile>Few-Shot learning with Reptile</a>                                                                          | 2020-05-21 | 2020-05-30 |
| vision          | <a href=https://keras.io/examples/vision/retinanet>Object Detection with RetinaNet</a>                                                                       | 2020-05-17 | 2020-07-14 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_from_scratch>Image classification from scratch</a>                                             | 2020-04-27 | 2020-04-28 |
| vision          | <a href=https://keras.io/examples/vision/grad_cam>Grad-CAM class activation visualization</a>                                                                | 2020-04-26 | 2021-03-07 |
| vision          | <a href=https://keras.io/examples/vision/oxford_pets_image_segmentation>Image segmentation with a U-Net-like architecture</a>                                | 2019-03-20 | 2020-04-20 |
| vision          | <a href=https://keras.io/examples/vision/mnist_convnet>Simple MNIST convnet</a>                                                                              | 2015-06-19 | 2020-04-21 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_traffic_forecasting>Traffic forecasting using graph neural networks and LSTM</a>                     | 2021-12-28 | 2021-12-28 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_classification_transformer>Timeseries classification with a Transformer model</a>                    | 2021-06-25 | 2021-08-05 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_classification_from_scratch>Timeseries classification from scratch</a>                               | 2020-07-21 | 2021-07-16 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_weather_forecasting>Timeseries forecasting for weather prediction</a>                                | 2020-06-23 | 2020-07-20 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_anomaly_detection>Timeseries anomaly detection using an Autoencoder</a>                              | 2020-05-31 | 2020-05-31 |
| structured_data | <a href=https://keras.io/examples/structured_data/classification_with_tfdf>Classification with TensorFlow Decision Forests</a>                               | 2022-01-25 | 2022-01-25 |
| structured_data | <a href=https://keras.io/examples/structured_data/tabtransformer>Structured data learning with TabTransformer</a>                                            | 2022-01-18 | 2022-01-18 |
| structured_data | <a href=https://keras.io/examples/structured_data/classification_with_grn_and_vsn>Classification with Gated Residual and Variable Selection Networks</a>     | 2021-02-10 | 2021-02-10 |
| structured_data | <a href=https://keras.io/examples/structured_data/deep_neural_decision_forests>Classification with Neural Decision Forests</a>                               | 2021-01-15 | 2021-01-15 |
| structured_data | <a href=https://keras.io/examples/structured_data/wide_deep_cross_networks>Structured data learning with Wide, Deep, and Cross networks</a>                  | 2020-12-31 | 2021-05-05 |
| structured_data | <a href=https://keras.io/examples/structured_data/movielens_recommendations_transformers>A Transformer-based recommendation system</a>                       | 2020-12-30 | 2020-12-30 |
| structured_data | <a href=https://keras.io/examples/structured_data/structured_data_classification_from_scratch>Structured data classification from scratch</a>                | 2020-06-09 | 2020-06-09 |
| structured_data | <a href=https://keras.io/examples/structured_data/collaborative_filtering_movielens>Collaborative Filtering for Movie Recommendations</a>                    | 2020-05-24 | 2020-05-24 |
| structured_data | <a href=https://keras.io/examples/structured_data/imbalanced_classification>Imbalanced classification: credit card fraud detection</a>                       | 2019-05-28 | 2020-04-17 |
| rl              | <a href=https://keras.io/examples/rl/ppo_cartpole>Proximal Policy Optimization</a>                                                                           | 2021-06-24 | 2021-06-24 |
| rl              | <a href=https://keras.io/examples/rl/ddpg_pendulum>Deep Deterministic Policy Gradient (DDPG)</a>                                                             | 2020-06-04 | 2020-09-21 |
| rl              | <a href=https://keras.io/examples/rl/deep_q_network_breakout>Deep Q-Learning for Atari Breakout</a>                                                          | 2020-05-23 | 2020-06-17 |
| rl              | <a href=https://keras.io/examples/rl/actor_critic_cartpole>Actor Critic Method</a>                                                                           | 2020-05-13 | 2020-05-13 |
| nlp             | <a href=https://keras.io/examples/nlp/question_answering>Question Answering with Hugging Face Transformers</a>                                               | 2022-01-13 | 2022-01-13 |
| nlp             | <a href=https://keras.io/examples/nlp/active_learning_review_classification>Review Classification using Active Learning</a>                                  | 2021-10-29 | 2021-10-29 |
| nlp             | <a href=https://keras.io/examples/nlp/text_generation_fnet>Text Generation using FNet</a>                                                                    | 2021-10-05 | 2021-10-05 |
| nlp             | <a href=https://keras.io/examples/nlp/multimodal_entailment>Multimodal entailment</a>                                                                        | 2021-08-08 | 2021-08-15 |
| nlp             | <a href=https://keras.io/examples/nlp/ner_transformers>Named Entity Recognition using Transformers</a>                                                       | 2021-06-23 | 2021-06-24 |
| nlp             | <a href=https://keras.io/examples/nlp/neural_machine_translation_with_transformer>English-to-Spanish translation with a sequence-to-sequence Transformer</a> | 2021-05-26 | 2021-05-26 |
| nlp             | <a href=https://keras.io/examples/nlp/nl_image_search>Natural language image search with a Dual Encoder</a>                                                  | 2021-01-30 | 2021-01-30 |
| nlp             | <a href=https://keras.io/examples/nlp/multi_label_classification>Large-scale multi-label text classification</a>                                             | 2020-09-25 | 2020-12-23 |
| nlp             | <a href=https://keras.io/examples/nlp/masked_language_modeling>End-to-end Masked Language Modeling with BERT</a>                                             | 2020-09-18 | 2020-09-18 |
| nlp             | <a href=https://keras.io/examples/nlp/semantic_similarity_with_bert>Semantic Similarity with BERT</a>                                                        | 2020-08-15 | 2020-08-29 |
| nlp             | <a href=https://keras.io/examples/nlp/text_extraction_with_bert>Text Extraction with BERT</a>                                                                | 2020-05-23 | 2020-05-23 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_with_switch_transformer>Text classification with Switch Transformer</a>                            | 2020-05-10 | 2021-02-15 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_with_transformer>Text classification with Transformer</a>                                          | 2020-05-10 | 2020-05-10 |
| nlp             | <a href=https://keras.io/examples/nlp/pretrained_word_embeddings>Using pre-trained word embeddings</a>                                                       | 2020-05-05 | 2020-05-05 |
| nlp             | <a href=https://keras.io/examples/nlp/bidirectional_lstm_imdb>Bidirectional LSTM on IMDB</a>                                                                 | 2020-05-03 | 2020-05-03 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_from_scratch>Text classification from scratch</a>                                                  | 2019-11-06 | 2020-05-17 |
| nlp             | <a href=https://keras.io/examples/nlp/lstm_seq2seq>Character-level recurrent sequence-to-sequence model</a>                                                  | 2017-09-29 | 2020-04-26 |
| nlp             | <a href=https://keras.io/examples/nlp/addition_rnn>Sequence to sequence learning for performing number addition</a>                                          | 2015-08-17 | 2020-04-17 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/subclassing_conv_layers>Customizing the convolution operation of a Conv2D layer</a>                          | 2021-11-03 | 2021-11-03 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/sklearn_metric_callbacks>Evaluating and exporting scikit-learn metrics in a Keras callback</a>               | 2021-10-07 | 2021-10-07 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/tensorflow_nu_models>Writing Keras Models With TensorFlow NumPy</a>                                          | 2021-08-28 | 2021-08-28 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/better_knowledge_distillation>Knowledge distillation recipes</a>                                             | 2021-08-01 | 2021-08-01 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/sample_size_estimate>Estimating required sample size for model training</a>                                  | 2021-05-20 | 2021-06-06 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/creating_tfrecords>Creating TFRecords</a>                                                                    | 2021-02-27 | 2021-02-27 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/memory_efficient_embeddings>Memory-efficient embeddings for recommendation systems</a>                       | 2021-02-15 | 2021-02-15 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/bayesian_neural_networks>Probabilistic Bayesian Neural Networks</a>                                          | 2021-01-15 | 2021-01-15 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/tfrecord>How to train a Keras model on TFRecord files</a>                                                    | 2020-07-29 | 2020-08-07 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/debugging_tips>Keras debugging tips</a>                                                                      | 2020-05-16 | 2020-05-16 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/quasi_svm>A Quasi-SVM in Keras</a>                                                                           | 2020-04-17 | 2020-04-17 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/endpoint_layer_pattern>Endpoint layer pattern</a>                                                            | 2019-05-10 | 2019-05-10 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/antirectifier>Simple custom layer example: Antirectifier</a>                                                 | 2016-01-06 | 2020-04-20 |
| graph           | <a href=https://keras.io/examples/graph/gat_node_classification>Graph attention network (GAT) for node classification</a>                                    | 2021-09-13 | 2021-12-26 |
| graph           | <a href=https://keras.io/examples/graph/mpnn-molecular-graphs>Message-passing neural network (MPNN) for molecular property prediction</a>                    | 2021-08-16 | 2021-12-27 |
| graph           | <a href=https://keras.io/examples/graph/gnn_citations>Node Classification with Graph Neural Networks</a>                                                     | 2021-05-30 | 2021-05-30 |
| graph           | <a href=https://keras.io/examples/graph/node2vec_movielens>Graph representation learning with node2vec</a>                                                   | 2021-05-15 | 2021-05-15 |
| generative      | <a href=https://keras.io/examples/generative/gaugan>GauGAN for conditional image generation</a>                                                              | 2021-12-26 | 2022-01-03 |
| generative      | <a href=https://keras.io/examples/generative/adain>Neural Style Transfer with AdaIN</a>                                                                      | 2021-11-08 | 2021-11-08 |
| generative      | <a href=https://keras.io/examples/generative/gan_ada>Data-efficient GANs with Adaptive Discriminator Augmentation</a>                                        | 2021-10-28 | 2021-10-28 |
| generative      | <a href=https://keras.io/examples/generative/vq_vae>Vector-Quantized Variational Autoencoders</a>                                                            | 2021-07-21 | 2021-07-21 |
| generative      | <a href=https://keras.io/examples/generative/conditional_gan>Conditional GAN</a>                                                                             | 2021-07-13 | 2021-07-15 |
| generative      | <a href=https://keras.io/examples/generative/stylegan>Face image generation with StyleGAN</a>                                                                | 2021-07-01 | 2021-07-01 |
| generative      | <a href=https://keras.io/examples/generative/wgan-graphs>WGAN-GP with R-GCN for the generation of small molecular graphs</a>                                 | 2021-06-30 | 2021-06-30 |
| generative      | <a href=https://keras.io/examples/generative/cyclegan>CycleGAN</a>                                                                                           | 2020-08-12 | 2020-08-12 |
| generative      | <a href=https://keras.io/examples/generative/real_nvp>Density estimation using Real NVP</a>                                                                  | 2020-08-10 | 2020-08-10 |
| generative      | <a href=https://keras.io/examples/generative/text_generation_with_miniature_gpt>Text generation with a miniature GPT</a>                                     | 2020-05-29 | 2020-05-29 |
| generative      | <a href=https://keras.io/examples/generative/pixelcnn>PixelCNN</a>                                                                                           | 2020-05-17 | 2020-05-23 |
| generative      | <a href=https://keras.io/examples/generative/wgan_gp>WGAN-GP overriding `Model.train_step`</a>                                                               | 2020-05-09 | 2020-05-09 |
| generative      | <a href=https://keras.io/examples/generative/vae>Variational AutoEncoder</a>                                                                                 | 2020-05-03 | 2020-05-03 |
| generative      | <a href=https://keras.io/examples/generative/dcgan_overriding_train_step>DCGAN to generate face images</a>                                                   | 2019-04-29 | 2021-01-01 |
| generative      | <a href=https://keras.io/examples/generative/deep_dream>Deep Dream</a>                                                                                       | 2016-01-13 | 2020-05-02 |
| generative      | <a href=https://keras.io/examples/generative/neural_style_transfer>Neural style transfer</a>                                                                 | 2016-01-11 | 2020-05-02 |
| generative      | <a href=https://keras.io/examples/generative/lstm_character_level_text_generation>Character-level text generation with LSTM</a>                              | 2015-06-15 | 2020-04-30 |
| audio           | <a href=https://keras.io/examples/audio/ctc_asr>Automatic Speech Recognition using CTC</a>                                                                   | 2021-09-26 | 2021-09-26 |
| audio           | <a href=https://keras.io/examples/audio/melgan_spectrogram_inversion>MelGAN-based spectrogram inversion using feature matching</a>                           | 2021-02-09 | 2021-09-15 |
| audio           | <a href=https://keras.io/examples/audio/transformer_asr>Automatic Speech Recognition with Transformer</a>                                                    | 2021-01-13 | 2021-01-13 |
| audio           | <a href=https://keras.io/examples/audio/speaker_recognition_using_cnn>Speaker Recognition</a>                                                                | 2020-06-14 | 2020-03-07 |


```python
# sorted by 'Category' and 'Last modified'

sorted_report_df = report_df.sort_values(by=['Category', 'Last modified'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```

| Category        | Title                                                                                                                                                        | Date created        | Last modified       |
|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:--------------------|
| vision          | <a href=https://keras.io/examples/vision/patch_convnet>Augmenting convnets with aggregated attention</a>                                                     | 2022-01-22 | 2022-01-22 |
| vision          | <a href=https://keras.io/examples/vision/vivit>Video Vision Transformer</a>                                                                                  | 2022-01-12 | 2022-01-12 |
| vision          | <a href=https://keras.io/examples/vision/vit_small_ds>Train a Vision Transformer on small datasets</a>                                                       | 2022-01-07 | 2022-01-10 |
| vision          | <a href=https://keras.io/examples/vision/masked_image_modeling>Masked image modeling with Autoencoders</a>                                                   | 2021-12-20 | 2021-12-21 |
| vision          | <a href=https://keras.io/examples/vision/barlow_twins>Barlow Twins for Contrastive SSL</a>                                                                   | 2021-11-04 | 2021-12-20 |
| vision          | <a href=https://keras.io/examples/vision/token_learner>Learning to tokenize in Vision Transformers</a>                                                       | 2021-12-10 | 2021-12-15 |
| vision          | <a href=https://keras.io/examples/vision/attention_mil_classification>Classification using Attention-based Deep Multiple Instance Learning (MIL).</a>        | 2021-08-16 | 2021-11-25 |
| vision          | <a href=https://keras.io/examples/vision/image_captioning>Image Captioning</a>                                                                               | 2021-05-29 | 2021-10-31 |
| vision          | <a href=https://keras.io/examples/vision/mobilevit>MobileViT: A mobile-friendly Transformer-based model for image classification</a>                         | 2021-10-20 | 2021-10-20 |
| vision          | <a href=https://keras.io/examples/vision/eanet>Image classification with EANet (External Attention Transformer)</a>                                          | 2021-10-19 | 2021-10-19 |
| vision          | <a href=https://keras.io/examples/vision/convmixer>Image classification with ConvMixer</a>                                                                   | 2021-10-12 | 2021-10-12 |
| vision          | <a href=https://keras.io/examples/vision/fixres>FixRes: Fixing train-test resolution discrepancy</a>                                                         | 2021-10-08 | 2021-10-10 |
| vision          | <a href=https://keras.io/examples/vision/metric_learning_tf_similarity>Metric learning for image similarity search using TensorFlow Similarity</a>           | 2021-09-30 | 2021-09-30 |
| vision          | <a href=https://keras.io/examples/vision/bit>Image Classification using BigTransfer (BiT)</a>                                                                | 2021-09-24 | 2021-09-24 |
| vision          | <a href=https://keras.io/examples/vision/zero_dce>Zero-DCE for low-light image enhancement</a>                                                               | 2021-09-18 | 2021-09-19 |
| vision          | <a href=https://keras.io/examples/vision/mirnet>Low-light image enhancement using MIRNet</a>                                                                 | 2021-09-11 | 2021-09-15 |
| vision          | <a href=https://keras.io/examples/vision/nnclr>Self-supervised contrastive learning with NNCLR</a>                                                           | 2021-09-13 | 2021-09-13 |
| vision          | <a href=https://keras.io/examples/vision/near_dup_search>Near-duplicate image search</a>                                                                     | 2021-09-10 | 2021-09-10 |
| vision          | <a href=https://keras.io/examples/vision/swin_transformers>Image classification with Swin Transformers</a>                                                   | 2021-09-08 | 2021-09-08 |
| vision          | <a href=https://keras.io/examples/vision/deeplabv3_plus>Multiclass semantic segmentation using DeepLabV3+</a>                                                | 2021-08-31 | 2021-09-01 |
| vision          | <a href=https://keras.io/examples/vision/depth_estimation>Monocular depth estimation</a>                                                                     | 2021-08-30 | 2021-08-30 |
| vision          | <a href=https://keras.io/examples/vision/handwriting_recognition>Handwriting recognition</a>                                                                 | 2021-08-16 | 2021-08-16 |
| vision          | <a href=https://keras.io/examples/vision/nerf>3D volumetric rendering with NeRF</a>                                                                          | 2021-08-09 | 2021-08-09 |
| vision          | <a href=https://keras.io/examples/vision/involution>Involutional neural networks</a>                                                                         | 2021-07-25 | 2021-07-25 |
| vision          | <a href=https://keras.io/examples/vision/cct>Compact Convolutional Transformers</a>                                                                          | 2021-06-30 | 2021-06-30 |
| vision          | <a href=https://keras.io/examples/vision/adamatch>Semi-supervision and domain adaptation with AdaMatch</a>                                                   | 2021-06-19 | 2021-06-19 |
| vision          | <a href=https://keras.io/examples/vision/gradient_centralization>Gradient Centralization for Better Training Performance</a>                                 | 2021-06-18 | 2021-06-18 |
| vision          | <a href=https://keras.io/examples/vision/cutmix>CutMix data augmentation for image classification</a>                                                        | 2021-06-08 | 2021-06-08 |
| vision          | <a href=https://keras.io/examples/vision/video_transformers>Video Classification with Transformers</a>                                                       | 2021-06-08 | 2021-06-08 |
| vision          | <a href=https://keras.io/examples/vision/video_classification>Video Classification with a CNN-RNN Architecture</a>                                           | 2021-05-28 | 2021-06-05 |
| vision          | <a href=https://keras.io/examples/vision/conv_lstm>Next-Frame Video Prediction with Convolutional LSTMs</a>                                                  | 2021-06-02 | 2021-06-05 |
| vision          | <a href=https://keras.io/examples/vision/mlp_image_classification>Image classification with modern MLP models</a>                                            | 2021-05-30 | 2021-05-30 |
| vision          | <a href=https://keras.io/examples/vision/learnable_resizer>Learning to Resize in Computer Vision</a>                                                         | 2021-04-30 | 2021-05-13 |
| vision          | <a href=https://keras.io/examples/vision/siamese_contrastive>Image similarity estimation using a Siamese Network with a contrastive loss</a>                 | 2021-05-06 | 2021-05-06 |
| vision          | <a href=https://keras.io/examples/vision/keypoint_detection>Keypoint Detection with Transfer Learning</a>                                                    | 2021-05-02 | 2021-05-02 |
| vision          | <a href=https://keras.io/examples/vision/semisupervised_simclr>Semi-supervised image classification using contrastive pretraining with SimCLR</a>            | 2021-04-24 | 2021-04-24 |
| vision          | <a href=https://keras.io/examples/vision/consistency_training>Consistency training with supervision</a>                                                      | 2021-04-13 | 2021-04-19 |
| vision          | <a href=https://keras.io/examples/vision/siamese_network>Image similarity estimation using a Siamese Network with a triplet loss</a>                         | 2021-03-25 | 2021-03-25 |
| vision          | <a href=https://keras.io/examples/vision/simsiam>Self-supervised contrastive learning with SimSiam</a>                                                       | 2021-03-19 | 2021-03-20 |
| vision          | <a href=https://keras.io/examples/vision/randaugment>RandAugment for Image Classification for Improved Robustness</a>                                        | 2021-03-13 | 2021-03-17 |
| vision          | <a href=https://keras.io/examples/vision/grad_cam>Grad-CAM class activation visualization</a>                                                                | 2020-04-26 | 2021-03-07 |
| vision          | <a href=https://keras.io/examples/vision/mixup>MixUp augmentation for image classification</a>                                                               | 2021-03-06 | 2021-03-06 |
| vision          | <a href=https://keras.io/examples/vision/autoencoder>Convolutional autoencoder for image denoising</a>                                                       | 2021-03-01 | 2021-03-01 |
| vision          | <a href=https://keras.io/examples/vision/semantic_image_clustering>Semantic Image Clustering</a>                                                             | 2021-02-28 | 2021-02-28 |
| vision          | <a href=https://keras.io/examples/vision/perceiver_image_classification>Image classification with Perceiver</a>                                              | 2021-04-30 | 2021-01-30 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_with_vision_transformer>Image classification with Vision Transformer</a>                       | 2021-01-18 | 2021-01-18 |
| vision          | <a href=https://keras.io/examples/vision/supervised-contrastive-learning>Supervised Contrastive Learning</a>                                                 | 2020-11-30 | 2020-11-30 |
| vision          | <a href=https://keras.io/examples/vision/pointnet_segmentation>Point cloud segmentation with PointNet</a>                                                    | 2020-10-23 | 2020-10-24 |
| vision          | <a href=https://keras.io/examples/vision/3D_image_classification>3D image classification from CT scans</a>                                                   | 2020-09-23 | 2020-09-23 |
| vision          | <a href=https://keras.io/examples/vision/knowledge_distillation>Knowledge Distillation</a>                                                                   | 2020-09-01 | 2020-09-01 |
| vision          | <a href=https://keras.io/examples/vision/super_resolution_sub_pixel>Image Super-Resolution using an Efficient Sub-Pixel CNN</a>                              | 2020-07-28 | 2020-08-27 |
| vision          | <a href=https://keras.io/examples/vision/xray_classification_with_tpus>Pneumonia Classification on TPU</a>                                                   | 2020-07-28 | 2020-08-24 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning>Image classification via fine-tuning with EfficientNet</a>            | 2020-06-30 | 2020-07-16 |
| vision          | <a href=https://keras.io/examples/vision/retinanet>Object Detection with RetinaNet</a>                                                                       | 2020-05-17 | 2020-07-14 |
| vision          | <a href=https://keras.io/examples/vision/captcha_ocr>OCR model for reading Captchas</a>                                                                      | 2020-06-14 | 2020-06-26 |
| vision          | <a href=https://keras.io/examples/vision/metric_learning>Metric learning for image similarity search</a>                                                     | 2020-06-05 | 2020-06-09 |
| vision          | <a href=https://keras.io/examples/vision/integrated_gradients>Model interpretability with Integrated Gradients</a>                                           | 2020-06-02 | 2020-06-02 |
| vision          | <a href=https://keras.io/examples/vision/reptile>Few-Shot learning with Reptile</a>                                                                          | 2020-05-21 | 2020-05-30 |
| vision          | <a href=https://keras.io/examples/vision/visualizing_what_convnets_learn>Visualizing what convnets learn</a>                                                 | 2020-05-29 | 2020-05-29 |
| vision          | <a href=https://keras.io/examples/vision/pointnet>Point cloud classification with PointNet</a>                                                               | 2020-05-25 | 2020-05-26 |
| vision          | <a href=https://keras.io/examples/vision/image_classification_from_scratch>Image classification from scratch</a>                                             | 2020-04-27 | 2020-04-28 |
| vision          | <a href=https://keras.io/examples/vision/mnist_convnet>Simple MNIST convnet</a>                                                                              | 2015-06-19 | 2020-04-21 |
| vision          | <a href=https://keras.io/examples/vision/oxford_pets_image_segmentation>Image segmentation with a U-Net-like architecture</a>                                | 2019-03-20 | 2020-04-20 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_traffic_forecasting>Traffic forecasting using graph neural networks and LSTM</a>                     | 2021-12-28 | 2021-12-28 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_classification_transformer>Timeseries classification with a Transformer model</a>                    | 2021-06-25 | 2021-08-05 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_classification_from_scratch>Timeseries classification from scratch</a>                               | 2020-07-21 | 2021-07-16 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_weather_forecasting>Timeseries forecasting for weather prediction</a>                                | 2020-06-23 | 2020-07-20 |
| timeseries      | <a href=https://keras.io/examples/timeseries/timeseries_anomaly_detection>Timeseries anomaly detection using an Autoencoder</a>                              | 2020-05-31 | 2020-05-31 |
| structured_data | <a href=https://keras.io/examples/structured_data/classification_with_tfdf>Classification with TensorFlow Decision Forests</a>                               | 2022-01-25 | 2022-01-25 |
| structured_data | <a href=https://keras.io/examples/structured_data/tabtransformer>Structured data learning with TabTransformer</a>                                            | 2022-01-18 | 2022-01-18 |
| structured_data | <a href=https://keras.io/examples/structured_data/wide_deep_cross_networks>Structured data learning with Wide, Deep, and Cross networks</a>                  | 2020-12-31 | 2021-05-05 |
| structured_data | <a href=https://keras.io/examples/structured_data/classification_with_grn_and_vsn>Classification with Gated Residual and Variable Selection Networks</a>     | 2021-02-10 | 2021-02-10 |
| structured_data | <a href=https://keras.io/examples/structured_data/deep_neural_decision_forests>Classification with Neural Decision Forests</a>                               | 2021-01-15 | 2021-01-15 |
| structured_data | <a href=https://keras.io/examples/structured_data/movielens_recommendations_transformers>A Transformer-based recommendation system</a>                       | 2020-12-30 | 2020-12-30 |
| structured_data | <a href=https://keras.io/examples/structured_data/structured_data_classification_from_scratch>Structured data classification from scratch</a>                | 2020-06-09 | 2020-06-09 |
| structured_data | <a href=https://keras.io/examples/structured_data/collaborative_filtering_movielens>Collaborative Filtering for Movie Recommendations</a>                    | 2020-05-24 | 2020-05-24 |
| structured_data | <a href=https://keras.io/examples/structured_data/imbalanced_classification>Imbalanced classification: credit card fraud detection</a>                       | 2019-05-28 | 2020-04-17 |
| rl              | <a href=https://keras.io/examples/rl/ppo_cartpole>Proximal Policy Optimization</a>                                                                           | 2021-06-24 | 2021-06-24 |
| rl              | <a href=https://keras.io/examples/rl/ddpg_pendulum>Deep Deterministic Policy Gradient (DDPG)</a>                                                             | 2020-06-04 | 2020-09-21 |
| rl              | <a href=https://keras.io/examples/rl/deep_q_network_breakout>Deep Q-Learning for Atari Breakout</a>                                                          | 2020-05-23 | 2020-06-17 |
| rl              | <a href=https://keras.io/examples/rl/actor_critic_cartpole>Actor Critic Method</a>                                                                           | 2020-05-13 | 2020-05-13 |
| nlp             | <a href=https://keras.io/examples/nlp/question_answering>Question Answering with Hugging Face Transformers</a>                                               | 2022-01-13 | 2022-01-13 |
| nlp             | <a href=https://keras.io/examples/nlp/active_learning_review_classification>Review Classification using Active Learning</a>                                  | 2021-10-29 | 2021-10-29 |
| nlp             | <a href=https://keras.io/examples/nlp/text_generation_fnet>Text Generation using FNet</a>                                                                    | 2021-10-05 | 2021-10-05 |
| nlp             | <a href=https://keras.io/examples/nlp/multimodal_entailment>Multimodal entailment</a>                                                                        | 2021-08-08 | 2021-08-15 |
| nlp             | <a href=https://keras.io/examples/nlp/ner_transformers>Named Entity Recognition using Transformers</a>                                                       | 2021-06-23 | 2021-06-24 |
| nlp             | <a href=https://keras.io/examples/nlp/neural_machine_translation_with_transformer>English-to-Spanish translation with a sequence-to-sequence Transformer</a> | 2021-05-26 | 2021-05-26 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_with_switch_transformer>Text classification with Switch Transformer</a>                            | 2020-05-10 | 2021-02-15 |
| nlp             | <a href=https://keras.io/examples/nlp/nl_image_search>Natural language image search with a Dual Encoder</a>                                                  | 2021-01-30 | 2021-01-30 |
| nlp             | <a href=https://keras.io/examples/nlp/multi_label_classification>Large-scale multi-label text classification</a>                                             | 2020-09-25 | 2020-12-23 |
| nlp             | <a href=https://keras.io/examples/nlp/masked_language_modeling>End-to-end Masked Language Modeling with BERT</a>                                             | 2020-09-18 | 2020-09-18 |
| nlp             | <a href=https://keras.io/examples/nlp/semantic_similarity_with_bert>Semantic Similarity with BERT</a>                                                        | 2020-08-15 | 2020-08-29 |
| nlp             | <a href=https://keras.io/examples/nlp/text_extraction_with_bert>Text Extraction with BERT</a>                                                                | 2020-05-23 | 2020-05-23 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_from_scratch>Text classification from scratch</a>                                                  | 2019-11-06 | 2020-05-17 |
| nlp             | <a href=https://keras.io/examples/nlp/text_classification_with_transformer>Text classification with Transformer</a>                                          | 2020-05-10 | 2020-05-10 |
| nlp             | <a href=https://keras.io/examples/nlp/pretrained_word_embeddings>Using pre-trained word embeddings</a>                                                       | 2020-05-05 | 2020-05-05 |
| nlp             | <a href=https://keras.io/examples/nlp/bidirectional_lstm_imdb>Bidirectional LSTM on IMDB</a>                                                                 | 2020-05-03 | 2020-05-03 |
| nlp             | <a href=https://keras.io/examples/nlp/lstm_seq2seq>Character-level recurrent sequence-to-sequence model</a>                                                  | 2017-09-29 | 2020-04-26 |
| nlp             | <a href=https://keras.io/examples/nlp/addition_rnn>Sequence to sequence learning for performing number addition</a>                                          | 2015-08-17 | 2020-04-17 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/subclassing_conv_layers>Customizing the convolution operation of a Conv2D layer</a>                          | 2021-11-03 | 2021-11-03 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/sklearn_metric_callbacks>Evaluating and exporting scikit-learn metrics in a Keras callback</a>               | 2021-10-07 | 2021-10-07 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/tensorflow_nu_models>Writing Keras Models With TensorFlow NumPy</a>                                          | 2021-08-28 | 2021-08-28 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/better_knowledge_distillation>Knowledge distillation recipes</a>                                             | 2021-08-01 | 2021-08-01 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/sample_size_estimate>Estimating required sample size for model training</a>                                  | 2021-05-20 | 2021-06-06 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/creating_tfrecords>Creating TFRecords</a>                                                                    | 2021-02-27 | 2021-02-27 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/memory_efficient_embeddings>Memory-efficient embeddings for recommendation systems</a>                       | 2021-02-15 | 2021-02-15 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/bayesian_neural_networks>Probabilistic Bayesian Neural Networks</a>                                          | 2021-01-15 | 2021-01-15 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/tfrecord>How to train a Keras model on TFRecord files</a>                                                    | 2020-07-29 | 2020-08-07 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/debugging_tips>Keras debugging tips</a>                                                                      | 2020-05-16 | 2020-05-16 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/antirectifier>Simple custom layer example: Antirectifier</a>                                                 | 2016-01-06 | 2020-04-20 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/quasi_svm>A Quasi-SVM in Keras</a>                                                                           | 2020-04-17 | 2020-04-17 |
| keras_recipes   | <a href=https://keras.io/examples/keras_recipes/endpoint_layer_pattern>Endpoint layer pattern</a>                                                            | 2019-05-10 | 2019-05-10 |
| graph           | <a href=https://keras.io/examples/graph/mpnn-molecular-graphs>Message-passing neural network (MPNN) for molecular property prediction</a>                    | 2021-08-16 | 2021-12-27 |
| graph           | <a href=https://keras.io/examples/graph/gat_node_classification>Graph attention network (GAT) for node classification</a>                                    | 2021-09-13 | 2021-12-26 |
| graph           | <a href=https://keras.io/examples/graph/gnn_citations>Node Classification with Graph Neural Networks</a>                                                     | 2021-05-30 | 2021-05-30 |
| graph           | <a href=https://keras.io/examples/graph/node2vec_movielens>Graph representation learning with node2vec</a>                                                   | 2021-05-15 | 2021-05-15 |
| generative      | <a href=https://keras.io/examples/generative/gaugan>GauGAN for conditional image generation</a>                                                              | 2021-12-26 | 2022-01-03 |
| generative      | <a href=https://keras.io/examples/generative/adain>Neural Style Transfer with AdaIN</a>                                                                      | 2021-11-08 | 2021-11-08 |
| generative      | <a href=https://keras.io/examples/generative/gan_ada>Data-efficient GANs with Adaptive Discriminator Augmentation</a>                                        | 2021-10-28 | 2021-10-28 |
| generative      | <a href=https://keras.io/examples/generative/vq_vae>Vector-Quantized Variational Autoencoders</a>                                                            | 2021-07-21 | 2021-07-21 |
| generative      | <a href=https://keras.io/examples/generative/conditional_gan>Conditional GAN</a>                                                                             | 2021-07-13 | 2021-07-15 |
| generative      | <a href=https://keras.io/examples/generative/stylegan>Face image generation with StyleGAN</a>                                                                | 2021-07-01 | 2021-07-01 |
| generative      | <a href=https://keras.io/examples/generative/wgan-graphs>WGAN-GP with R-GCN for the generation of small molecular graphs</a>                                 | 2021-06-30 | 2021-06-30 |
| generative      | <a href=https://keras.io/examples/generative/dcgan_overriding_train_step>DCGAN to generate face images</a>                                                   | 2019-04-29 | 2021-01-01 |
| generative      | <a href=https://keras.io/examples/generative/cyclegan>CycleGAN</a>                                                                                           | 2020-08-12 | 2020-08-12 |
| generative      | <a href=https://keras.io/examples/generative/real_nvp>Density estimation using Real NVP</a>                                                                  | 2020-08-10 | 2020-08-10 |
| generative      | <a href=https://keras.io/examples/generative/text_generation_with_miniature_gpt>Text generation with a miniature GPT</a>                                     | 2020-05-29 | 2020-05-29 |
| generative      | <a href=https://keras.io/examples/generative/pixelcnn>PixelCNN</a>                                                                                           | 2020-05-17 | 2020-05-23 |
| generative      | <a href=https://keras.io/examples/generative/wgan_gp>WGAN-GP overriding `Model.train_step`</a>                                                               | 2020-05-09 | 2020-05-09 |
| generative      | <a href=https://keras.io/examples/generative/vae>Variational AutoEncoder</a>                                                                                 | 2020-05-03 | 2020-05-03 |
| generative      | <a href=https://keras.io/examples/generative/deep_dream>Deep Dream</a>                                                                                       | 2016-01-13 | 2020-05-02 |
| generative      | <a href=https://keras.io/examples/generative/neural_style_transfer>Neural style transfer</a>                                                                 | 2016-01-11 | 2020-05-02 |
| generative      | <a href=https://keras.io/examples/generative/lstm_character_level_text_generation>Character-level text generation with LSTM</a>                              | 2015-06-15 | 2020-04-30 |
| audio           | <a href=https://keras.io/examples/audio/ctc_asr>Automatic Speech Recognition using CTC</a>                                                                   | 2021-09-26 | 2021-09-26 |
| audio           | <a href=https://keras.io/examples/audio/melgan_spectrogram_inversion>MelGAN-based spectrogram inversion using feature matching</a>                           | 2021-02-09 | 2021-09-15 |
| audio           | <a href=https://keras.io/examples/audio/transformer_asr>Automatic Speech Recognition with Transformer</a>                                                    | 2021-01-13 | 2021-01-13 |
| audio           | <a href=https://keras.io/examples/audio/speaker_recognition_using_cnn>Speaker Recognition</a>                                                                | 2020-06-14 | 2020-03-07 |
