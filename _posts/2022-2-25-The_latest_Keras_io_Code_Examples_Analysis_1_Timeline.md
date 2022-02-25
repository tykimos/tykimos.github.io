---
layout: post
title:  "The latest Keras.io Code Examples Analysis 1 - Timeline"
author: Taeyoung Kim
date:   2022-2-25 00:00:00
categories: tech
comments: true
image: http://tykimos.github.io/warehouse/2022-2-25-The_latest_Keras_io_Code_Examples_Analysis_1_Timeline_title.png
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

print(py_files)
print(catagories_path)
```

    ['keras-io/keras-io-master/examples/rl/deep_q_network_breakout.py', 'keras-io/keras-io-master/examples/rl/actor_critic_cartpole.py', 'keras-io/keras-io-master/examples/rl/ddpg_pendulum.py', 'keras-io/keras-io-master/examples/rl/ppo_cartpole.py', 'keras-io/keras-io-master/examples/keras_recipes/subclassing_conv_layers.py', 'keras-io/keras-io-master/examples/keras_recipes/tensorflow_numpy_models.py', 'keras-io/keras-io-master/examples/keras_recipes/memory_efficient_embeddings.py', 'keras-io/keras-io-master/examples/keras_recipes/creating_tfrecords.py', 'keras-io/keras-io-master/examples/keras_recipes/quasi_svm.py', 'keras-io/keras-io-master/examples/keras_recipes/antirectifier.py', 'keras-io/keras-io-master/examples/keras_recipes/bayesian_neural_networks.py', 'keras-io/keras-io-master/examples/keras_recipes/endpoint_layer_pattern.py', 'keras-io/keras-io-master/examples/keras_recipes/tfrecord.py', 'keras-io/keras-io-master/examples/keras_recipes/sample_size_estimate.py', 'keras-io/keras-io-master/examples/keras_recipes/sklearn_metric_callbacks.py', 'keras-io/keras-io-master/examples/keras_recipes/debugging_tips.py', 'keras-io/keras-io-master/examples/keras_recipes/better_knowledge_distillation.py', 'keras-io/keras-io-master/examples/graph/node2vec_movielens.py', 'keras-io/keras-io-master/examples/graph/gat_node_classification.py', 'keras-io/keras-io-master/examples/graph/gnn_citations.py', 'keras-io/keras-io-master/examples/graph/mpnn-molecular-graphs.py', 'keras-io/keras-io-master/examples/structured_data/tabtransformer.py', 'keras-io/keras-io-master/examples/structured_data/wide_deep_cross_networks.py', 'keras-io/keras-io-master/examples/structured_data/imbalanced_classification.py', 'keras-io/keras-io-master/examples/structured_data/collaborative_filtering_movielens.py', 'keras-io/keras-io-master/examples/structured_data/classification_with_grn_and_vsn.py', 'keras-io/keras-io-master/examples/structured_data/deep_neural_decision_forests.py', 'keras-io/keras-io-master/examples/structured_data/structured_data_classification_from_scratch.py', 'keras-io/keras-io-master/examples/structured_data/classification_with_tfdf.py', 'keras-io/keras-io-master/examples/structured_data/movielens_recommendations_transformers.py', 'keras-io/keras-io-master/examples/vision/image_classification_with_vision_transformer.py', 'keras-io/keras-io-master/examples/vision/pointnet_segmentation.py', 'keras-io/keras-io-master/examples/vision/grad_cam.py', 'keras-io/keras-io-master/examples/vision/oxford_pets_image_segmentation.py', 'keras-io/keras-io-master/examples/vision/mirnet.py', 'keras-io/keras-io-master/examples/vision/vit_small_ds.py', 'keras-io/keras-io-master/examples/vision/supervised-contrastive-learning.py', 'keras-io/keras-io-master/examples/vision/knowledge_distillation.py', 'keras-io/keras-io-master/examples/vision/eanet.py', 'keras-io/keras-io-master/examples/vision/nnclr.py', 'keras-io/keras-io-master/examples/vision/xray_classification_with_tpus.py', 'keras-io/keras-io-master/examples/vision/zero_dce.py', 'keras-io/keras-io-master/examples/vision/metric_learning_tf_similarity.py', 'keras-io/keras-io-master/examples/vision/image_classification_efficientnet_fine_tuning.py', 'keras-io/keras-io-master/examples/vision/gradient_centralization.py', 'keras-io/keras-io-master/examples/vision/perceiver_image_classification.py', 'keras-io/keras-io-master/examples/vision/convmixer.py', 'keras-io/keras-io-master/examples/vision/consistency_training.py', 'keras-io/keras-io-master/examples/vision/video_classification.py', 'keras-io/keras-io-master/examples/vision/token_learner.py', 'keras-io/keras-io-master/examples/vision/mnist_convnet.py', 'keras-io/keras-io-master/examples/vision/integrated_gradients.py', 'keras-io/keras-io-master/examples/vision/super_resolution_sub_pixel.py', 'keras-io/keras-io-master/examples/vision/retinanet.py', 'keras-io/keras-io-master/examples/vision/vivit.py', 'keras-io/keras-io-master/examples/vision/siamese_contrastive.py', 'keras-io/keras-io-master/examples/vision/adamatch.py', 'keras-io/keras-io-master/examples/vision/learnable_resizer.py', 'keras-io/keras-io-master/examples/vision/near_dup_search.py', 'keras-io/keras-io-master/examples/vision/visualizing_what_convnets_learn.py', 'keras-io/keras-io-master/examples/vision/siamese_network.py', 'keras-io/keras-io-master/examples/vision/autoencoder.py', 'keras-io/keras-io-master/examples/vision/cutmix.py', 'keras-io/keras-io-master/examples/vision/captcha_ocr.py', 'keras-io/keras-io-master/examples/vision/barlow_twins.py', 'keras-io/keras-io-master/examples/vision/nerf.py', 'keras-io/keras-io-master/examples/vision/image_captioning.py', 'keras-io/keras-io-master/examples/vision/depth_estimation.py', 'keras-io/keras-io-master/examples/vision/swin_transformers.py', 'keras-io/keras-io-master/examples/vision/image_classification_from_scratch.py', 'keras-io/keras-io-master/examples/vision/video_transformers.py', 'keras-io/keras-io-master/examples/vision/pointnet.py', 'keras-io/keras-io-master/examples/vision/deeplabv3_plus.py', 'keras-io/keras-io-master/examples/vision/handwriting_recognition.py', 'keras-io/keras-io-master/examples/vision/semantic_image_clustering.py', 'keras-io/keras-io-master/examples/vision/mixup.py', 'keras-io/keras-io-master/examples/vision/attention_mil_classification.py', 'keras-io/keras-io-master/examples/vision/masked_image_modeling.py', 'keras-io/keras-io-master/examples/vision/keypoint_detection.py', 'keras-io/keras-io-master/examples/vision/metric_learning.py', 'keras-io/keras-io-master/examples/vision/cct.py', 'keras-io/keras-io-master/examples/vision/mlp_image_classification.py', 'keras-io/keras-io-master/examples/vision/3D_image_classification.py', 'keras-io/keras-io-master/examples/vision/mobilevit.py', 'keras-io/keras-io-master/examples/vision/randaugment.py', 'keras-io/keras-io-master/examples/vision/semisupervised_simclr.py', 'keras-io/keras-io-master/examples/vision/involution.py', 'keras-io/keras-io-master/examples/vision/conv_lstm.py', 'keras-io/keras-io-master/examples/vision/patch_convnet.py', 'keras-io/keras-io-master/examples/vision/reptile.py', 'keras-io/keras-io-master/examples/vision/bit.py', 'keras-io/keras-io-master/examples/vision/simsiam.py', 'keras-io/keras-io-master/examples/vision/fixres.py', 'keras-io/keras-io-master/examples/timeseries/timeseries_classification_from_scratch.py', 'keras-io/keras-io-master/examples/timeseries/timeseries_weather_forecasting.py', 'keras-io/keras-io-master/examples/timeseries/timeseries_classification_transformer.py', 'keras-io/keras-io-master/examples/timeseries/timeseries_anomaly_detection.py', 'keras-io/keras-io-master/examples/timeseries/timeseries_traffic_forecasting.py', 'keras-io/keras-io-master/examples/nlp/lstm_seq2seq.py', 'keras-io/keras-io-master/examples/nlp/semantic_similarity_with_bert.py', 'keras-io/keras-io-master/examples/nlp/text_extraction_with_bert.py', 'keras-io/keras-io-master/examples/nlp/text_classification_with_switch_transformer.py', 'keras-io/keras-io-master/examples/nlp/neural_machine_translation_with_transformer.py', 'keras-io/keras-io-master/examples/nlp/addition_rnn.py', 'keras-io/keras-io-master/examples/nlp/bidirectional_lstm_imdb.py', 'keras-io/keras-io-master/examples/nlp/active_learning_review_classification.py', 'keras-io/keras-io-master/examples/nlp/nl_image_search.py', 'keras-io/keras-io-master/examples/nlp/text_generation_fnet.py', 'keras-io/keras-io-master/examples/nlp/text_classification_from_scratch.py', 'keras-io/keras-io-master/examples/nlp/question_answering.py', 'keras-io/keras-io-master/examples/nlp/multimodal_entailment.py', 'keras-io/keras-io-master/examples/nlp/pretrained_word_embeddings.py', 'keras-io/keras-io-master/examples/nlp/text_classification_with_transformer.py', 'keras-io/keras-io-master/examples/nlp/multi_label_classification.py', 'keras-io/keras-io-master/examples/nlp/masked_language_modeling.py', 'keras-io/keras-io-master/examples/nlp/ner_transformers.py', 'keras-io/keras-io-master/examples/audio/melgan_spectrogram_inversion.py', 'keras-io/keras-io-master/examples/audio/transformer_asr.py', 'keras-io/keras-io-master/examples/audio/ctc_asr.py', 'keras-io/keras-io-master/examples/audio/speaker_recognition_using_cnn.py', 'keras-io/keras-io-master/examples/generative/deep_dream.py', 'keras-io/keras-io-master/examples/generative/gan_ada.py', 'keras-io/keras-io-master/examples/generative/adain.py', 'keras-io/keras-io-master/examples/generative/pixelcnn.py', 'keras-io/keras-io-master/examples/generative/neural_style_transfer.py', 'keras-io/keras-io-master/examples/generative/lstm_character_level_text_generation.py', 'keras-io/keras-io-master/examples/generative/wgan-graphs.py', 'keras-io/keras-io-master/examples/generative/dcgan_overriding_train_step.py', 'keras-io/keras-io-master/examples/generative/gaugan.py', 'keras-io/keras-io-master/examples/generative/stylegan.py', 'keras-io/keras-io-master/examples/generative/vae.py', 'keras-io/keras-io-master/examples/generative/real_nvp.py', 'keras-io/keras-io-master/examples/generative/vq_vae.py', 'keras-io/keras-io-master/examples/generative/cyclegan.py', 'keras-io/keras-io-master/examples/generative/text_generation_with_miniature_gpt.py', 'keras-io/keras-io-master/examples/generative/wgan_gp.py', 'keras-io/keras-io-master/examples/generative/conditional_gan.py']
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




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>Title</th>
      <th>Date created</th>
      <th>Last modified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/classification_with_tfdf>Classification with TensorFlow Decision Forests</a></td>
      <td>2022-01-25</td>
      <td>2022-01-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/patch_convnet>Augmenting convnets with aggregated attention</a></td>
      <td>2022-01-22</td>
      <td>2022-01-22</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/tabtransformer>Structured data learning with TabTransformer</a></td>
      <td>2022-01-18</td>
      <td>2022-01-18</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/question_answering>Question Answering with Hugging Face Transformers</a></td>
      <td>2022-01-13</td>
      <td>2022-01-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/vivit>Video Vision Transformer</a></td>
      <td>2022-01-12</td>
      <td>2022-01-12</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/vit_small_ds>Train a Vision Transformer on small datasets</a></td>
      <td>2022-01-07</td>
      <td>2022-01-10</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_traffic_forecasting>Traffic forecasting using graph neural networks and LSTM</a></td>
      <td>2021-12-28</td>
      <td>2021-12-28</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/gaugan>GauGAN for conditional image generation</a></td>
      <td>2021-12-26</td>
      <td>2022-01-03</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/masked_image_modeling>Masked image modeling with Autoencoders</a></td>
      <td>2021-12-20</td>
      <td>2021-12-21</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/token_learner>Learning to tokenize in Vision Transformers</a></td>
      <td>2021-12-10</td>
      <td>2021-12-15</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/adain>Neural Style Transfer with AdaIN</a></td>
      <td>2021-11-08</td>
      <td>2021-11-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/barlow_twins>Barlow Twins for Contrastive SSL</a></td>
      <td>2021-11-04</td>
      <td>2021-12-20</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/subclassing_conv_layers>Customizing the convolution operation of a Conv2D layer</a></td>
      <td>2021-11-03</td>
      <td>2021-11-03</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/active_learning_review_classification>Review Classification using Active Learning</a></td>
      <td>2021-10-29</td>
      <td>2021-10-29</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/gan_ada>Data-efficient GANs with Adaptive Discriminator Augmentation</a></td>
      <td>2021-10-28</td>
      <td>2021-10-28</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mobilevit>MobileViT: A mobile-friendly Transformer-based model for image classification</a></td>
      <td>2021-10-20</td>
      <td>2021-10-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/eanet>Image classification with EANet (External Attention Transformer)</a></td>
      <td>2021-10-19</td>
      <td>2021-10-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/convmixer>Image classification with ConvMixer</a></td>
      <td>2021-10-12</td>
      <td>2021-10-12</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/fixres>FixRes: Fixing train-test resolution discrepancy</a></td>
      <td>2021-10-08</td>
      <td>2021-10-10</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/sklearn_metric_callbacks>Evaluating and exporting scikit-learn metrics in a Keras callback</a></td>
      <td>2021-10-07</td>
      <td>2021-10-07</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_generation_fnet>Text Generation using FNet</a></td>
      <td>2021-10-05</td>
      <td>2021-10-05</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/metric_learning_tf_similarity>Metric learning for image similarity search using TensorFlow Similarity</a></td>
      <td>2021-09-30</td>
      <td>2021-09-30</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/ctc_asr>Automatic Speech Recognition using CTC</a></td>
      <td>2021-09-26</td>
      <td>2021-09-26</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/bit>Image Classification using BigTransfer (BiT)</a></td>
      <td>2021-09-24</td>
      <td>2021-09-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/zero_dce>Zero-DCE for low-light image enhancement</a></td>
      <td>2021-09-18</td>
      <td>2021-09-19</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/gat_node_classification>Graph attention network (GAT) for node classification</a></td>
      <td>2021-09-13</td>
      <td>2021-12-26</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/nnclr>Self-supervised contrastive learning with NNCLR</a></td>
      <td>2021-09-13</td>
      <td>2021-09-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mirnet>Low-light image enhancement using MIRNet</a></td>
      <td>2021-09-11</td>
      <td>2021-09-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/near_dup_search>Near-duplicate image search</a></td>
      <td>2021-09-10</td>
      <td>2021-09-10</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/swin_transformers>Image classification with Swin Transformers</a></td>
      <td>2021-09-08</td>
      <td>2021-09-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/deeplabv3_plus>Multiclass semantic segmentation using DeepLabV3+</a></td>
      <td>2021-08-31</td>
      <td>2021-09-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/depth_estimation>Monocular depth estimation</a></td>
      <td>2021-08-30</td>
      <td>2021-08-30</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/tensorflow_nu_models>Writing Keras Models With TensorFlow NumPy</a></td>
      <td>2021-08-28</td>
      <td>2021-08-28</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/handwriting_recognition>Handwriting recognition</a></td>
      <td>2021-08-16</td>
      <td>2021-08-16</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/attention_mil_classification>Classification using Attention-based Deep Multiple Instance Learning (MIL).</a></td>
      <td>2021-08-16</td>
      <td>2021-11-25</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/mpnn-molecular-graphs>Message-passing neural network (MPNN) for molecular property prediction</a></td>
      <td>2021-08-16</td>
      <td>2021-12-27</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/nerf>3D volumetric rendering with NeRF</a></td>
      <td>2021-08-09</td>
      <td>2021-08-09</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/multimodal_entailment>Multimodal entailment</a></td>
      <td>2021-08-08</td>
      <td>2021-08-15</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/better_knowledge_distillation>Knowledge distillation recipes</a></td>
      <td>2021-08-01</td>
      <td>2021-08-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/involution>Involutional neural networks</a></td>
      <td>2021-07-25</td>
      <td>2021-07-25</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/vq_vae>Vector-Quantized Variational Autoencoders</a></td>
      <td>2021-07-21</td>
      <td>2021-07-21</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/conditional_gan>Conditional GAN</a></td>
      <td>2021-07-13</td>
      <td>2021-07-15</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/stylegan>Face image generation with StyleGAN</a></td>
      <td>2021-07-01</td>
      <td>2021-07-01</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/wgan-graphs>WGAN-GP with R-GCN for the generation of small molecular graphs</a></td>
      <td>2021-06-30</td>
      <td>2021-06-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/cct>Compact Convolutional Transformers</a></td>
      <td>2021-06-30</td>
      <td>2021-06-30</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_classification_transformer>Timeseries classification with a Transformer model</a></td>
      <td>2021-06-25</td>
      <td>2021-08-05</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/ppo_cartpole>Proximal Policy Optimization</a></td>
      <td>2021-06-24</td>
      <td>2021-06-24</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/ner_transformers>Named Entity Recognition using Transformers</a></td>
      <td>2021-06-23</td>
      <td>2021-06-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/adamatch>Semi-supervision and domain adaptation with AdaMatch</a></td>
      <td>2021-06-19</td>
      <td>2021-06-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/gradient_centralization>Gradient Centralization for Better Training Performance</a></td>
      <td>2021-06-18</td>
      <td>2021-06-18</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/video_transformers>Video Classification with Transformers</a></td>
      <td>2021-06-08</td>
      <td>2021-06-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/cutmix>CutMix data augmentation for image classification</a></td>
      <td>2021-06-08</td>
      <td>2021-06-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/conv_lstm>Next-Frame Video Prediction with Convolutional LSTMs</a></td>
      <td>2021-06-02</td>
      <td>2021-06-05</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mlp_image_classification>Image classification with modern MLP models</a></td>
      <td>2021-05-30</td>
      <td>2021-05-30</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/gnn_citations>Node Classification with Graph Neural Networks</a></td>
      <td>2021-05-30</td>
      <td>2021-05-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_captioning>Image Captioning</a></td>
      <td>2021-05-29</td>
      <td>2021-10-31</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/video_classification>Video Classification with a CNN-RNN Architecture</a></td>
      <td>2021-05-28</td>
      <td>2021-06-05</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/neural_machine_translation_with_transformer>English-to-Spanish translation with a sequence-to-sequence Transformer</a></td>
      <td>2021-05-26</td>
      <td>2021-05-26</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/sample_size_estimate>Estimating required sample size for model training</a></td>
      <td>2021-05-20</td>
      <td>2021-06-06</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/node2vec_movielens>Graph representation learning with node2vec</a></td>
      <td>2021-05-15</td>
      <td>2021-05-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/siamese_contrastive>Image similarity estimation using a Siamese Network with a contrastive loss</a></td>
      <td>2021-05-06</td>
      <td>2021-05-06</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/keypoint_detection>Keypoint Detection with Transfer Learning</a></td>
      <td>2021-05-02</td>
      <td>2021-05-02</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/learnable_resizer>Learning to Resize in Computer Vision</a></td>
      <td>2021-04-30</td>
      <td>2021-05-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/perceiver_image_classification>Image classification with Perceiver</a></td>
      <td>2021-04-30</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/semisupervised_simclr>Semi-supervised image classification using contrastive pretraining with SimCLR</a></td>
      <td>2021-04-24</td>
      <td>2021-04-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/consistency_training>Consistency training with supervision</a></td>
      <td>2021-04-13</td>
      <td>2021-04-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/siamese_network>Image similarity estimation using a Siamese Network with a triplet loss</a></td>
      <td>2021-03-25</td>
      <td>2021-03-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/simsiam>Self-supervised contrastive learning with SimSiam</a></td>
      <td>2021-03-19</td>
      <td>2021-03-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/randaugment>RandAugment for Image Classification for Improved Robustness</a></td>
      <td>2021-03-13</td>
      <td>2021-03-17</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mixup>MixUp augmentation for image classification</a></td>
      <td>2021-03-06</td>
      <td>2021-03-06</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/autoencoder>Convolutional autoencoder for image denoising</a></td>
      <td>2021-03-01</td>
      <td>2021-03-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/semantic_image_clustering>Semantic Image Clustering</a></td>
      <td>2021-02-28</td>
      <td>2021-02-28</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/creating_tfrecords>Creating TFRecords</a></td>
      <td>2021-02-27</td>
      <td>2021-02-27</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/memory_efficient_embeddings>Memory-efficient embeddings for recommendation systems</a></td>
      <td>2021-02-15</td>
      <td>2021-02-15</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/classification_with_grn_and_vsn>Classification with Gated Residual and Variable Selection Networks</a></td>
      <td>2021-02-10</td>
      <td>2021-02-10</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/melgan_spectrogram_inversion>MelGAN-based spectrogram inversion using feature matching</a></td>
      <td>2021-02-09</td>
      <td>2021-09-15</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/nl_image_search>Natural language image search with a Dual Encoder</a></td>
      <td>2021-01-30</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_with_vision_transformer>Image classification with Vision Transformer</a></td>
      <td>2021-01-18</td>
      <td>2021-01-18</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/deep_neural_decision_forests>Classification with Neural Decision Forests</a></td>
      <td>2021-01-15</td>
      <td>2021-01-15</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/bayesian_neural_networks>Probabilistic Bayesian Neural Networks</a></td>
      <td>2021-01-15</td>
      <td>2021-01-15</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/transformer_asr>Automatic Speech Recognition with Transformer</a></td>
      <td>2021-01-13</td>
      <td>2021-01-13</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/wide_deep_cross_networks>Structured data learning with Wide, Deep, and Cross networks</a></td>
      <td>2020-12-31</td>
      <td>2021-05-05</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/movielens_recommendations_transformers>A Transformer-based recommendation system</a></td>
      <td>2020-12-30</td>
      <td>2020-12-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/supervised-contrastive-learning>Supervised Contrastive Learning</a></td>
      <td>2020-11-30</td>
      <td>2020-11-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/pointnet_segmentation>Point cloud segmentation with PointNet</a></td>
      <td>2020-10-23</td>
      <td>2020-10-24</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/multi_label_classification>Large-scale multi-label text classification</a></td>
      <td>2020-09-25</td>
      <td>2020-12-23</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/3D_image_classification>3D image classification from CT scans</a></td>
      <td>2020-09-23</td>
      <td>2020-09-23</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/masked_language_modeling>End-to-end Masked Language Modeling with BERT</a></td>
      <td>2020-09-18</td>
      <td>2020-09-18</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/knowledge_distillation>Knowledge Distillation</a></td>
      <td>2020-09-01</td>
      <td>2020-09-01</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/semantic_similarity_with_bert>Semantic Similarity with BERT</a></td>
      <td>2020-08-15</td>
      <td>2020-08-29</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/cyclegan>CycleGAN</a></td>
      <td>2020-08-12</td>
      <td>2020-08-12</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/real_nvp>Density estimation using Real NVP</a></td>
      <td>2020-08-10</td>
      <td>2020-08-10</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/tfrecord>How to train a Keras model on TFRecord files</a></td>
      <td>2020-07-29</td>
      <td>2020-08-07</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/xray_classification_with_tpus>Pneumonia Classification on TPU</a></td>
      <td>2020-07-28</td>
      <td>2020-08-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/super_resolution_sub_pixel>Image Super-Resolution using an Efficient Sub-Pixel CNN</a></td>
      <td>2020-07-28</td>
      <td>2020-08-27</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_classification_from_scratch>Timeseries classification from scratch</a></td>
      <td>2020-07-21</td>
      <td>2021-07-16</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning>Image classification via fine-tuning with EfficientNet</a></td>
      <td>2020-06-30</td>
      <td>2020-07-16</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_weather_forecasting>Timeseries forecasting for weather prediction</a></td>
      <td>2020-06-23</td>
      <td>2020-07-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/captcha_ocr>OCR model for reading Captchas</a></td>
      <td>2020-06-14</td>
      <td>2020-06-26</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/speaker_recognition_using_cnn>Speaker Recognition</a></td>
      <td>2020-06-14</td>
      <td>2020-03-07</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/structured_data_classification_from_scratch>Structured data classification from scratch</a></td>
      <td>2020-06-09</td>
      <td>2020-06-09</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/metric_learning>Metric learning for image similarity search</a></td>
      <td>2020-06-05</td>
      <td>2020-06-09</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/ddpg_pendulum>Deep Deterministic Policy Gradient (DDPG)</a></td>
      <td>2020-06-04</td>
      <td>2020-09-21</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/integrated_gradients>Model interpretability with Integrated Gradients</a></td>
      <td>2020-06-02</td>
      <td>2020-06-02</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_anomaly_detection>Timeseries anomaly detection using an Autoencoder</a></td>
      <td>2020-05-31</td>
      <td>2020-05-31</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/text_generation_with_miniature_gpt>Text generation with a miniature GPT</a></td>
      <td>2020-05-29</td>
      <td>2020-05-29</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/visualizing_what_convnets_learn>Visualizing what convnets learn</a></td>
      <td>2020-05-29</td>
      <td>2020-05-29</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/pointnet>Point cloud classification with PointNet</a></td>
      <td>2020-05-25</td>
      <td>2020-05-26</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/collaborative_filtering_movielens>Collaborative Filtering for Movie Recommendations</a></td>
      <td>2020-05-24</td>
      <td>2020-05-24</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/deep_q_network_breakout>Deep Q-Learning for Atari Breakout</a></td>
      <td>2020-05-23</td>
      <td>2020-06-17</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_extraction_with_bert>Text Extraction with BERT</a></td>
      <td>2020-05-23</td>
      <td>2020-05-23</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/reptile>Few-Shot learning with Reptile</a></td>
      <td>2020-05-21</td>
      <td>2020-05-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/retinanet>Object Detection with RetinaNet</a></td>
      <td>2020-05-17</td>
      <td>2020-07-14</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/pixelcnn>PixelCNN</a></td>
      <td>2020-05-17</td>
      <td>2020-05-23</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/debugging_tips>Keras debugging tips</a></td>
      <td>2020-05-16</td>
      <td>2020-05-16</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/actor_critic_cartpole>Actor Critic Method</a></td>
      <td>2020-05-13</td>
      <td>2020-05-13</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_with_switch_transformer>Text classification with Switch Transformer</a></td>
      <td>2020-05-10</td>
      <td>2021-02-15</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_with_transformer>Text classification with Transformer</a></td>
      <td>2020-05-10</td>
      <td>2020-05-10</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/wgan_gp>WGAN-GP overriding `Model.train_step`</a></td>
      <td>2020-05-09</td>
      <td>2020-05-09</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/pretrained_word_embeddings>Using pre-trained word embeddings</a></td>
      <td>2020-05-05</td>
      <td>2020-05-05</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/vae>Variational AutoEncoder</a></td>
      <td>2020-05-03</td>
      <td>2020-05-03</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/bidirectional_lstm_imdb>Bidirectional LSTM on IMDB</a></td>
      <td>2020-05-03</td>
      <td>2020-05-03</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_from_scratch>Image classification from scratch</a></td>
      <td>2020-04-27</td>
      <td>2020-04-28</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/grad_cam>Grad-CAM class activation visualization</a></td>
      <td>2020-04-26</td>
      <td>2021-03-07</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/quasi_svm>A Quasi-SVM in Keras</a></td>
      <td>2020-04-17</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_from_scratch>Text classification from scratch</a></td>
      <td>2019-11-06</td>
      <td>2020-05-17</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/imbalanced_classification>Imbalanced classification: credit card fraud detection</a></td>
      <td>2019-05-28</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/endpoint_layer_pattern>Endpoint layer pattern</a></td>
      <td>2019-05-10</td>
      <td>2019-05-10</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/dcgan_overriding_train_step>DCGAN to generate face images</a></td>
      <td>2019-04-29</td>
      <td>2021-01-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/oxford_pets_image_segmentation>Image segmentation with a U-Net-like architecture</a></td>
      <td>2019-03-20</td>
      <td>2020-04-20</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/lstm_seq2seq>Character-level recurrent sequence-to-sequence model</a></td>
      <td>2017-09-29</td>
      <td>2020-04-26</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/deep_dream>Deep Dream</a></td>
      <td>2016-01-13</td>
      <td>2020-05-02</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/neural_style_transfer>Neural style transfer</a></td>
      <td>2016-01-11</td>
      <td>2020-05-02</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/antirectifier>Simple custom layer example: Antirectifier</a></td>
      <td>2016-01-06</td>
      <td>2020-04-20</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/addition_rnn>Sequence to sequence learning for performing number addition</a></td>
      <td>2015-08-17</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mnist_convnet>Simple MNIST convnet</a></td>
      <td>2015-06-19</td>
      <td>2020-04-21</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/lstm_character_level_text_generation>Character-level text generation with LSTM</a></td>
      <td>2015-06-15</td>
      <td>2020-04-30</td>
    </tr>
  </tbody>
</table>




```python
# sorted by 'Last modified'

sorted_report_df = report_df.sort_values(by=['Last modified'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>Title</th>
      <th>Date created</th>
      <th>Last modified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/classification_with_tfdf>Classification with TensorFlow Decision Forests</a></td>
      <td>2022-01-25</td>
      <td>2022-01-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/patch_convnet>Augmenting convnets with aggregated attention</a></td>
      <td>2022-01-22</td>
      <td>2022-01-22</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/tabtransformer>Structured data learning with TabTransformer</a></td>
      <td>2022-01-18</td>
      <td>2022-01-18</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/question_answering>Question Answering with Hugging Face Transformers</a></td>
      <td>2022-01-13</td>
      <td>2022-01-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/vivit>Video Vision Transformer</a></td>
      <td>2022-01-12</td>
      <td>2022-01-12</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/vit_small_ds>Train a Vision Transformer on small datasets</a></td>
      <td>2022-01-07</td>
      <td>2022-01-10</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/gaugan>GauGAN for conditional image generation</a></td>
      <td>2021-12-26</td>
      <td>2022-01-03</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_traffic_forecasting>Traffic forecasting using graph neural networks and LSTM</a></td>
      <td>2021-12-28</td>
      <td>2021-12-28</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/mpnn-molecular-graphs>Message-passing neural network (MPNN) for molecular property prediction</a></td>
      <td>2021-08-16</td>
      <td>2021-12-27</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/gat_node_classification>Graph attention network (GAT) for node classification</a></td>
      <td>2021-09-13</td>
      <td>2021-12-26</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/masked_image_modeling>Masked image modeling with Autoencoders</a></td>
      <td>2021-12-20</td>
      <td>2021-12-21</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/barlow_twins>Barlow Twins for Contrastive SSL</a></td>
      <td>2021-11-04</td>
      <td>2021-12-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/token_learner>Learning to tokenize in Vision Transformers</a></td>
      <td>2021-12-10</td>
      <td>2021-12-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/attention_mil_classification>Classification using Attention-based Deep Multiple Instance Learning (MIL).</a></td>
      <td>2021-08-16</td>
      <td>2021-11-25</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/adain>Neural Style Transfer with AdaIN</a></td>
      <td>2021-11-08</td>
      <td>2021-11-08</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/subclassing_conv_layers>Customizing the convolution operation of a Conv2D layer</a></td>
      <td>2021-11-03</td>
      <td>2021-11-03</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_captioning>Image Captioning</a></td>
      <td>2021-05-29</td>
      <td>2021-10-31</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/active_learning_review_classification>Review Classification using Active Learning</a></td>
      <td>2021-10-29</td>
      <td>2021-10-29</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/gan_ada>Data-efficient GANs with Adaptive Discriminator Augmentation</a></td>
      <td>2021-10-28</td>
      <td>2021-10-28</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mobilevit>MobileViT: A mobile-friendly Transformer-based model for image classification</a></td>
      <td>2021-10-20</td>
      <td>2021-10-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/eanet>Image classification with EANet (External Attention Transformer)</a></td>
      <td>2021-10-19</td>
      <td>2021-10-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/convmixer>Image classification with ConvMixer</a></td>
      <td>2021-10-12</td>
      <td>2021-10-12</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/fixres>FixRes: Fixing train-test resolution discrepancy</a></td>
      <td>2021-10-08</td>
      <td>2021-10-10</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/sklearn_metric_callbacks>Evaluating and exporting scikit-learn metrics in a Keras callback</a></td>
      <td>2021-10-07</td>
      <td>2021-10-07</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_generation_fnet>Text Generation using FNet</a></td>
      <td>2021-10-05</td>
      <td>2021-10-05</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/metric_learning_tf_similarity>Metric learning for image similarity search using TensorFlow Similarity</a></td>
      <td>2021-09-30</td>
      <td>2021-09-30</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/ctc_asr>Automatic Speech Recognition using CTC</a></td>
      <td>2021-09-26</td>
      <td>2021-09-26</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/bit>Image Classification using BigTransfer (BiT)</a></td>
      <td>2021-09-24</td>
      <td>2021-09-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/zero_dce>Zero-DCE for low-light image enhancement</a></td>
      <td>2021-09-18</td>
      <td>2021-09-19</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/melgan_spectrogram_inversion>MelGAN-based spectrogram inversion using feature matching</a></td>
      <td>2021-02-09</td>
      <td>2021-09-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mirnet>Low-light image enhancement using MIRNet</a></td>
      <td>2021-09-11</td>
      <td>2021-09-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/nnclr>Self-supervised contrastive learning with NNCLR</a></td>
      <td>2021-09-13</td>
      <td>2021-09-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/near_dup_search>Near-duplicate image search</a></td>
      <td>2021-09-10</td>
      <td>2021-09-10</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/swin_transformers>Image classification with Swin Transformers</a></td>
      <td>2021-09-08</td>
      <td>2021-09-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/deeplabv3_plus>Multiclass semantic segmentation using DeepLabV3+</a></td>
      <td>2021-08-31</td>
      <td>2021-09-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/depth_estimation>Monocular depth estimation</a></td>
      <td>2021-08-30</td>
      <td>2021-08-30</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/tensorflow_nu_models>Writing Keras Models With TensorFlow NumPy</a></td>
      <td>2021-08-28</td>
      <td>2021-08-28</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/handwriting_recognition>Handwriting recognition</a></td>
      <td>2021-08-16</td>
      <td>2021-08-16</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/multimodal_entailment>Multimodal entailment</a></td>
      <td>2021-08-08</td>
      <td>2021-08-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/nerf>3D volumetric rendering with NeRF</a></td>
      <td>2021-08-09</td>
      <td>2021-08-09</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_classification_transformer>Timeseries classification with a Transformer model</a></td>
      <td>2021-06-25</td>
      <td>2021-08-05</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/better_knowledge_distillation>Knowledge distillation recipes</a></td>
      <td>2021-08-01</td>
      <td>2021-08-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/involution>Involutional neural networks</a></td>
      <td>2021-07-25</td>
      <td>2021-07-25</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/vq_vae>Vector-Quantized Variational Autoencoders</a></td>
      <td>2021-07-21</td>
      <td>2021-07-21</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_classification_from_scratch>Timeseries classification from scratch</a></td>
      <td>2020-07-21</td>
      <td>2021-07-16</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/conditional_gan>Conditional GAN</a></td>
      <td>2021-07-13</td>
      <td>2021-07-15</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/stylegan>Face image generation with StyleGAN</a></td>
      <td>2021-07-01</td>
      <td>2021-07-01</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/wgan-graphs>WGAN-GP with R-GCN for the generation of small molecular graphs</a></td>
      <td>2021-06-30</td>
      <td>2021-06-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/cct>Compact Convolutional Transformers</a></td>
      <td>2021-06-30</td>
      <td>2021-06-30</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/ppo_cartpole>Proximal Policy Optimization</a></td>
      <td>2021-06-24</td>
      <td>2021-06-24</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/ner_transformers>Named Entity Recognition using Transformers</a></td>
      <td>2021-06-23</td>
      <td>2021-06-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/adamatch>Semi-supervision and domain adaptation with AdaMatch</a></td>
      <td>2021-06-19</td>
      <td>2021-06-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/gradient_centralization>Gradient Centralization for Better Training Performance</a></td>
      <td>2021-06-18</td>
      <td>2021-06-18</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/cutmix>CutMix data augmentation for image classification</a></td>
      <td>2021-06-08</td>
      <td>2021-06-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/video_transformers>Video Classification with Transformers</a></td>
      <td>2021-06-08</td>
      <td>2021-06-08</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/sample_size_estimate>Estimating required sample size for model training</a></td>
      <td>2021-05-20</td>
      <td>2021-06-06</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/video_classification>Video Classification with a CNN-RNN Architecture</a></td>
      <td>2021-05-28</td>
      <td>2021-06-05</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/conv_lstm>Next-Frame Video Prediction with Convolutional LSTMs</a></td>
      <td>2021-06-02</td>
      <td>2021-06-05</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/gnn_citations>Node Classification with Graph Neural Networks</a></td>
      <td>2021-05-30</td>
      <td>2021-05-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mlp_image_classification>Image classification with modern MLP models</a></td>
      <td>2021-05-30</td>
      <td>2021-05-30</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/neural_machine_translation_with_transformer>English-to-Spanish translation with a sequence-to-sequence Transformer</a></td>
      <td>2021-05-26</td>
      <td>2021-05-26</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/node2vec_movielens>Graph representation learning with node2vec</a></td>
      <td>2021-05-15</td>
      <td>2021-05-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/learnable_resizer>Learning to Resize in Computer Vision</a></td>
      <td>2021-04-30</td>
      <td>2021-05-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/siamese_contrastive>Image similarity estimation using a Siamese Network with a contrastive loss</a></td>
      <td>2021-05-06</td>
      <td>2021-05-06</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/wide_deep_cross_networks>Structured data learning with Wide, Deep, and Cross networks</a></td>
      <td>2020-12-31</td>
      <td>2021-05-05</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/keypoint_detection>Keypoint Detection with Transfer Learning</a></td>
      <td>2021-05-02</td>
      <td>2021-05-02</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/semisupervised_simclr>Semi-supervised image classification using contrastive pretraining with SimCLR</a></td>
      <td>2021-04-24</td>
      <td>2021-04-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/consistency_training>Consistency training with supervision</a></td>
      <td>2021-04-13</td>
      <td>2021-04-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/siamese_network>Image similarity estimation using a Siamese Network with a triplet loss</a></td>
      <td>2021-03-25</td>
      <td>2021-03-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/simsiam>Self-supervised contrastive learning with SimSiam</a></td>
      <td>2021-03-19</td>
      <td>2021-03-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/randaugment>RandAugment for Image Classification for Improved Robustness</a></td>
      <td>2021-03-13</td>
      <td>2021-03-17</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/grad_cam>Grad-CAM class activation visualization</a></td>
      <td>2020-04-26</td>
      <td>2021-03-07</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mixup>MixUp augmentation for image classification</a></td>
      <td>2021-03-06</td>
      <td>2021-03-06</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/autoencoder>Convolutional autoencoder for image denoising</a></td>
      <td>2021-03-01</td>
      <td>2021-03-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/semantic_image_clustering>Semantic Image Clustering</a></td>
      <td>2021-02-28</td>
      <td>2021-02-28</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/creating_tfrecords>Creating TFRecords</a></td>
      <td>2021-02-27</td>
      <td>2021-02-27</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/memory_efficient_embeddings>Memory-efficient embeddings for recommendation systems</a></td>
      <td>2021-02-15</td>
      <td>2021-02-15</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_with_switch_transformer>Text classification with Switch Transformer</a></td>
      <td>2020-05-10</td>
      <td>2021-02-15</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/classification_with_grn_and_vsn>Classification with Gated Residual and Variable Selection Networks</a></td>
      <td>2021-02-10</td>
      <td>2021-02-10</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/nl_image_search>Natural language image search with a Dual Encoder</a></td>
      <td>2021-01-30</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/perceiver_image_classification>Image classification with Perceiver</a></td>
      <td>2021-04-30</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_with_vision_transformer>Image classification with Vision Transformer</a></td>
      <td>2021-01-18</td>
      <td>2021-01-18</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/bayesian_neural_networks>Probabilistic Bayesian Neural Networks</a></td>
      <td>2021-01-15</td>
      <td>2021-01-15</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/deep_neural_decision_forests>Classification with Neural Decision Forests</a></td>
      <td>2021-01-15</td>
      <td>2021-01-15</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/transformer_asr>Automatic Speech Recognition with Transformer</a></td>
      <td>2021-01-13</td>
      <td>2021-01-13</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/dcgan_overriding_train_step>DCGAN to generate face images</a></td>
      <td>2019-04-29</td>
      <td>2021-01-01</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/movielens_recommendations_transformers>A Transformer-based recommendation system</a></td>
      <td>2020-12-30</td>
      <td>2020-12-30</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/multi_label_classification>Large-scale multi-label text classification</a></td>
      <td>2020-09-25</td>
      <td>2020-12-23</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/supervised-contrastive-learning>Supervised Contrastive Learning</a></td>
      <td>2020-11-30</td>
      <td>2020-11-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/pointnet_segmentation>Point cloud segmentation with PointNet</a></td>
      <td>2020-10-23</td>
      <td>2020-10-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/3D_image_classification>3D image classification from CT scans</a></td>
      <td>2020-09-23</td>
      <td>2020-09-23</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/ddpg_pendulum>Deep Deterministic Policy Gradient (DDPG)</a></td>
      <td>2020-06-04</td>
      <td>2020-09-21</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/masked_language_modeling>End-to-end Masked Language Modeling with BERT</a></td>
      <td>2020-09-18</td>
      <td>2020-09-18</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/knowledge_distillation>Knowledge Distillation</a></td>
      <td>2020-09-01</td>
      <td>2020-09-01</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/semantic_similarity_with_bert>Semantic Similarity with BERT</a></td>
      <td>2020-08-15</td>
      <td>2020-08-29</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/super_resolution_sub_pixel>Image Super-Resolution using an Efficient Sub-Pixel CNN</a></td>
      <td>2020-07-28</td>
      <td>2020-08-27</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/xray_classification_with_tpus>Pneumonia Classification on TPU</a></td>
      <td>2020-07-28</td>
      <td>2020-08-24</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/cyclegan>CycleGAN</a></td>
      <td>2020-08-12</td>
      <td>2020-08-12</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/real_nvp>Density estimation using Real NVP</a></td>
      <td>2020-08-10</td>
      <td>2020-08-10</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/tfrecord>How to train a Keras model on TFRecord files</a></td>
      <td>2020-07-29</td>
      <td>2020-08-07</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_weather_forecasting>Timeseries forecasting for weather prediction</a></td>
      <td>2020-06-23</td>
      <td>2020-07-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning>Image classification via fine-tuning with EfficientNet</a></td>
      <td>2020-06-30</td>
      <td>2020-07-16</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/retinanet>Object Detection with RetinaNet</a></td>
      <td>2020-05-17</td>
      <td>2020-07-14</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/captcha_ocr>OCR model for reading Captchas</a></td>
      <td>2020-06-14</td>
      <td>2020-06-26</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/deep_q_network_breakout>Deep Q-Learning for Atari Breakout</a></td>
      <td>2020-05-23</td>
      <td>2020-06-17</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/metric_learning>Metric learning for image similarity search</a></td>
      <td>2020-06-05</td>
      <td>2020-06-09</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/structured_data_classification_from_scratch>Structured data classification from scratch</a></td>
      <td>2020-06-09</td>
      <td>2020-06-09</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/integrated_gradients>Model interpretability with Integrated Gradients</a></td>
      <td>2020-06-02</td>
      <td>2020-06-02</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_anomaly_detection>Timeseries anomaly detection using an Autoencoder</a></td>
      <td>2020-05-31</td>
      <td>2020-05-31</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/reptile>Few-Shot learning with Reptile</a></td>
      <td>2020-05-21</td>
      <td>2020-05-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/visualizing_what_convnets_learn>Visualizing what convnets learn</a></td>
      <td>2020-05-29</td>
      <td>2020-05-29</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/text_generation_with_miniature_gpt>Text generation with a miniature GPT</a></td>
      <td>2020-05-29</td>
      <td>2020-05-29</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/pointnet>Point cloud classification with PointNet</a></td>
      <td>2020-05-25</td>
      <td>2020-05-26</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/collaborative_filtering_movielens>Collaborative Filtering for Movie Recommendations</a></td>
      <td>2020-05-24</td>
      <td>2020-05-24</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/pixelcnn>PixelCNN</a></td>
      <td>2020-05-17</td>
      <td>2020-05-23</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_extraction_with_bert>Text Extraction with BERT</a></td>
      <td>2020-05-23</td>
      <td>2020-05-23</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_from_scratch>Text classification from scratch</a></td>
      <td>2019-11-06</td>
      <td>2020-05-17</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/debugging_tips>Keras debugging tips</a></td>
      <td>2020-05-16</td>
      <td>2020-05-16</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/actor_critic_cartpole>Actor Critic Method</a></td>
      <td>2020-05-13</td>
      <td>2020-05-13</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_with_transformer>Text classification with Transformer</a></td>
      <td>2020-05-10</td>
      <td>2020-05-10</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/wgan_gp>WGAN-GP overriding `Model.train_step`</a></td>
      <td>2020-05-09</td>
      <td>2020-05-09</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/pretrained_word_embeddings>Using pre-trained word embeddings</a></td>
      <td>2020-05-05</td>
      <td>2020-05-05</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/bidirectional_lstm_imdb>Bidirectional LSTM on IMDB</a></td>
      <td>2020-05-03</td>
      <td>2020-05-03</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/vae>Variational AutoEncoder</a></td>
      <td>2020-05-03</td>
      <td>2020-05-03</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/neural_style_transfer>Neural style transfer</a></td>
      <td>2016-01-11</td>
      <td>2020-05-02</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/deep_dream>Deep Dream</a></td>
      <td>2016-01-13</td>
      <td>2020-05-02</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/lstm_character_level_text_generation>Character-level text generation with LSTM</a></td>
      <td>2015-06-15</td>
      <td>2020-04-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_from_scratch>Image classification from scratch</a></td>
      <td>2020-04-27</td>
      <td>2020-04-28</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/lstm_seq2seq>Character-level recurrent sequence-to-sequence model</a></td>
      <td>2017-09-29</td>
      <td>2020-04-26</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mnist_convnet>Simple MNIST convnet</a></td>
      <td>2015-06-19</td>
      <td>2020-04-21</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/antirectifier>Simple custom layer example: Antirectifier</a></td>
      <td>2016-01-06</td>
      <td>2020-04-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/oxford_pets_image_segmentation>Image segmentation with a U-Net-like architecture</a></td>
      <td>2019-03-20</td>
      <td>2020-04-20</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/quasi_svm>A Quasi-SVM in Keras</a></td>
      <td>2020-04-17</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/imbalanced_classification>Imbalanced classification: credit card fraud detection</a></td>
      <td>2019-05-28</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/addition_rnn>Sequence to sequence learning for performing number addition</a></td>
      <td>2015-08-17</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/speaker_recognition_using_cnn>Speaker Recognition</a></td>
      <td>2020-06-14</td>
      <td>2020-03-07</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/endpoint_layer_pattern>Endpoint layer pattern</a></td>
      <td>2019-05-10</td>
      <td>2019-05-10</td>
    </tr>
  </tbody>
</table>




```python
# sorted by 'Category' and 'Date created'

sorted_report_df = report_df.sort_values(by=['Category', 'Date created'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>Title</th>
      <th>Date created</th>
      <th>Last modified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/patch_convnet>Augmenting convnets with aggregated attention</a></td>
      <td>2022-01-22</td>
      <td>2022-01-22</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/vivit>Video Vision Transformer</a></td>
      <td>2022-01-12</td>
      <td>2022-01-12</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/vit_small_ds>Train a Vision Transformer on small datasets</a></td>
      <td>2022-01-07</td>
      <td>2022-01-10</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/masked_image_modeling>Masked image modeling with Autoencoders</a></td>
      <td>2021-12-20</td>
      <td>2021-12-21</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/token_learner>Learning to tokenize in Vision Transformers</a></td>
      <td>2021-12-10</td>
      <td>2021-12-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/barlow_twins>Barlow Twins for Contrastive SSL</a></td>
      <td>2021-11-04</td>
      <td>2021-12-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mobilevit>MobileViT: A mobile-friendly Transformer-based model for image classification</a></td>
      <td>2021-10-20</td>
      <td>2021-10-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/eanet>Image classification with EANet (External Attention Transformer)</a></td>
      <td>2021-10-19</td>
      <td>2021-10-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/convmixer>Image classification with ConvMixer</a></td>
      <td>2021-10-12</td>
      <td>2021-10-12</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/fixres>FixRes: Fixing train-test resolution discrepancy</a></td>
      <td>2021-10-08</td>
      <td>2021-10-10</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/metric_learning_tf_similarity>Metric learning for image similarity search using TensorFlow Similarity</a></td>
      <td>2021-09-30</td>
      <td>2021-09-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/bit>Image Classification using BigTransfer (BiT)</a></td>
      <td>2021-09-24</td>
      <td>2021-09-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/zero_dce>Zero-DCE for low-light image enhancement</a></td>
      <td>2021-09-18</td>
      <td>2021-09-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/nnclr>Self-supervised contrastive learning with NNCLR</a></td>
      <td>2021-09-13</td>
      <td>2021-09-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mirnet>Low-light image enhancement using MIRNet</a></td>
      <td>2021-09-11</td>
      <td>2021-09-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/near_dup_search>Near-duplicate image search</a></td>
      <td>2021-09-10</td>
      <td>2021-09-10</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/swin_transformers>Image classification with Swin Transformers</a></td>
      <td>2021-09-08</td>
      <td>2021-09-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/deeplabv3_plus>Multiclass semantic segmentation using DeepLabV3+</a></td>
      <td>2021-08-31</td>
      <td>2021-09-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/depth_estimation>Monocular depth estimation</a></td>
      <td>2021-08-30</td>
      <td>2021-08-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/handwriting_recognition>Handwriting recognition</a></td>
      <td>2021-08-16</td>
      <td>2021-08-16</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/attention_mil_classification>Classification using Attention-based Deep Multiple Instance Learning (MIL).</a></td>
      <td>2021-08-16</td>
      <td>2021-11-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/nerf>3D volumetric rendering with NeRF</a></td>
      <td>2021-08-09</td>
      <td>2021-08-09</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/involution>Involutional neural networks</a></td>
      <td>2021-07-25</td>
      <td>2021-07-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/cct>Compact Convolutional Transformers</a></td>
      <td>2021-06-30</td>
      <td>2021-06-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/adamatch>Semi-supervision and domain adaptation with AdaMatch</a></td>
      <td>2021-06-19</td>
      <td>2021-06-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/gradient_centralization>Gradient Centralization for Better Training Performance</a></td>
      <td>2021-06-18</td>
      <td>2021-06-18</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/cutmix>CutMix data augmentation for image classification</a></td>
      <td>2021-06-08</td>
      <td>2021-06-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/video_transformers>Video Classification with Transformers</a></td>
      <td>2021-06-08</td>
      <td>2021-06-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/conv_lstm>Next-Frame Video Prediction with Convolutional LSTMs</a></td>
      <td>2021-06-02</td>
      <td>2021-06-05</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mlp_image_classification>Image classification with modern MLP models</a></td>
      <td>2021-05-30</td>
      <td>2021-05-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_captioning>Image Captioning</a></td>
      <td>2021-05-29</td>
      <td>2021-10-31</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/video_classification>Video Classification with a CNN-RNN Architecture</a></td>
      <td>2021-05-28</td>
      <td>2021-06-05</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/siamese_contrastive>Image similarity estimation using a Siamese Network with a contrastive loss</a></td>
      <td>2021-05-06</td>
      <td>2021-05-06</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/keypoint_detection>Keypoint Detection with Transfer Learning</a></td>
      <td>2021-05-02</td>
      <td>2021-05-02</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/perceiver_image_classification>Image classification with Perceiver</a></td>
      <td>2021-04-30</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/learnable_resizer>Learning to Resize in Computer Vision</a></td>
      <td>2021-04-30</td>
      <td>2021-05-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/semisupervised_simclr>Semi-supervised image classification using contrastive pretraining with SimCLR</a></td>
      <td>2021-04-24</td>
      <td>2021-04-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/consistency_training>Consistency training with supervision</a></td>
      <td>2021-04-13</td>
      <td>2021-04-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/siamese_network>Image similarity estimation using a Siamese Network with a triplet loss</a></td>
      <td>2021-03-25</td>
      <td>2021-03-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/simsiam>Self-supervised contrastive learning with SimSiam</a></td>
      <td>2021-03-19</td>
      <td>2021-03-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/randaugment>RandAugment for Image Classification for Improved Robustness</a></td>
      <td>2021-03-13</td>
      <td>2021-03-17</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mixup>MixUp augmentation for image classification</a></td>
      <td>2021-03-06</td>
      <td>2021-03-06</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/autoencoder>Convolutional autoencoder for image denoising</a></td>
      <td>2021-03-01</td>
      <td>2021-03-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/semantic_image_clustering>Semantic Image Clustering</a></td>
      <td>2021-02-28</td>
      <td>2021-02-28</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_with_vision_transformer>Image classification with Vision Transformer</a></td>
      <td>2021-01-18</td>
      <td>2021-01-18</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/supervised-contrastive-learning>Supervised Contrastive Learning</a></td>
      <td>2020-11-30</td>
      <td>2020-11-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/pointnet_segmentation>Point cloud segmentation with PointNet</a></td>
      <td>2020-10-23</td>
      <td>2020-10-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/3D_image_classification>3D image classification from CT scans</a></td>
      <td>2020-09-23</td>
      <td>2020-09-23</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/knowledge_distillation>Knowledge Distillation</a></td>
      <td>2020-09-01</td>
      <td>2020-09-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/xray_classification_with_tpus>Pneumonia Classification on TPU</a></td>
      <td>2020-07-28</td>
      <td>2020-08-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/super_resolution_sub_pixel>Image Super-Resolution using an Efficient Sub-Pixel CNN</a></td>
      <td>2020-07-28</td>
      <td>2020-08-27</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning>Image classification via fine-tuning with EfficientNet</a></td>
      <td>2020-06-30</td>
      <td>2020-07-16</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/captcha_ocr>OCR model for reading Captchas</a></td>
      <td>2020-06-14</td>
      <td>2020-06-26</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/metric_learning>Metric learning for image similarity search</a></td>
      <td>2020-06-05</td>
      <td>2020-06-09</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/integrated_gradients>Model interpretability with Integrated Gradients</a></td>
      <td>2020-06-02</td>
      <td>2020-06-02</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/visualizing_what_convnets_learn>Visualizing what convnets learn</a></td>
      <td>2020-05-29</td>
      <td>2020-05-29</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/pointnet>Point cloud classification with PointNet</a></td>
      <td>2020-05-25</td>
      <td>2020-05-26</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/reptile>Few-Shot learning with Reptile</a></td>
      <td>2020-05-21</td>
      <td>2020-05-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/retinanet>Object Detection with RetinaNet</a></td>
      <td>2020-05-17</td>
      <td>2020-07-14</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_from_scratch>Image classification from scratch</a></td>
      <td>2020-04-27</td>
      <td>2020-04-28</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/grad_cam>Grad-CAM class activation visualization</a></td>
      <td>2020-04-26</td>
      <td>2021-03-07</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/oxford_pets_image_segmentation>Image segmentation with a U-Net-like architecture</a></td>
      <td>2019-03-20</td>
      <td>2020-04-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mnist_convnet>Simple MNIST convnet</a></td>
      <td>2015-06-19</td>
      <td>2020-04-21</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_traffic_forecasting>Traffic forecasting using graph neural networks and LSTM</a></td>
      <td>2021-12-28</td>
      <td>2021-12-28</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_classification_transformer>Timeseries classification with a Transformer model</a></td>
      <td>2021-06-25</td>
      <td>2021-08-05</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_classification_from_scratch>Timeseries classification from scratch</a></td>
      <td>2020-07-21</td>
      <td>2021-07-16</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_weather_forecasting>Timeseries forecasting for weather prediction</a></td>
      <td>2020-06-23</td>
      <td>2020-07-20</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_anomaly_detection>Timeseries anomaly detection using an Autoencoder</a></td>
      <td>2020-05-31</td>
      <td>2020-05-31</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/classification_with_tfdf>Classification with TensorFlow Decision Forests</a></td>
      <td>2022-01-25</td>
      <td>2022-01-25</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/tabtransformer>Structured data learning with TabTransformer</a></td>
      <td>2022-01-18</td>
      <td>2022-01-18</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/classification_with_grn_and_vsn>Classification with Gated Residual and Variable Selection Networks</a></td>
      <td>2021-02-10</td>
      <td>2021-02-10</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/deep_neural_decision_forests>Classification with Neural Decision Forests</a></td>
      <td>2021-01-15</td>
      <td>2021-01-15</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/wide_deep_cross_networks>Structured data learning with Wide, Deep, and Cross networks</a></td>
      <td>2020-12-31</td>
      <td>2021-05-05</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/movielens_recommendations_transformers>A Transformer-based recommendation system</a></td>
      <td>2020-12-30</td>
      <td>2020-12-30</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/structured_data_classification_from_scratch>Structured data classification from scratch</a></td>
      <td>2020-06-09</td>
      <td>2020-06-09</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/collaborative_filtering_movielens>Collaborative Filtering for Movie Recommendations</a></td>
      <td>2020-05-24</td>
      <td>2020-05-24</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/imbalanced_classification>Imbalanced classification: credit card fraud detection</a></td>
      <td>2019-05-28</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/ppo_cartpole>Proximal Policy Optimization</a></td>
      <td>2021-06-24</td>
      <td>2021-06-24</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/ddpg_pendulum>Deep Deterministic Policy Gradient (DDPG)</a></td>
      <td>2020-06-04</td>
      <td>2020-09-21</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/deep_q_network_breakout>Deep Q-Learning for Atari Breakout</a></td>
      <td>2020-05-23</td>
      <td>2020-06-17</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/actor_critic_cartpole>Actor Critic Method</a></td>
      <td>2020-05-13</td>
      <td>2020-05-13</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/question_answering>Question Answering with Hugging Face Transformers</a></td>
      <td>2022-01-13</td>
      <td>2022-01-13</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/active_learning_review_classification>Review Classification using Active Learning</a></td>
      <td>2021-10-29</td>
      <td>2021-10-29</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_generation_fnet>Text Generation using FNet</a></td>
      <td>2021-10-05</td>
      <td>2021-10-05</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/multimodal_entailment>Multimodal entailment</a></td>
      <td>2021-08-08</td>
      <td>2021-08-15</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/ner_transformers>Named Entity Recognition using Transformers</a></td>
      <td>2021-06-23</td>
      <td>2021-06-24</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/neural_machine_translation_with_transformer>English-to-Spanish translation with a sequence-to-sequence Transformer</a></td>
      <td>2021-05-26</td>
      <td>2021-05-26</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/nl_image_search>Natural language image search with a Dual Encoder</a></td>
      <td>2021-01-30</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/multi_label_classification>Large-scale multi-label text classification</a></td>
      <td>2020-09-25</td>
      <td>2020-12-23</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/masked_language_modeling>End-to-end Masked Language Modeling with BERT</a></td>
      <td>2020-09-18</td>
      <td>2020-09-18</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/semantic_similarity_with_bert>Semantic Similarity with BERT</a></td>
      <td>2020-08-15</td>
      <td>2020-08-29</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_extraction_with_bert>Text Extraction with BERT</a></td>
      <td>2020-05-23</td>
      <td>2020-05-23</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_with_switch_transformer>Text classification with Switch Transformer</a></td>
      <td>2020-05-10</td>
      <td>2021-02-15</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_with_transformer>Text classification with Transformer</a></td>
      <td>2020-05-10</td>
      <td>2020-05-10</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/pretrained_word_embeddings>Using pre-trained word embeddings</a></td>
      <td>2020-05-05</td>
      <td>2020-05-05</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/bidirectional_lstm_imdb>Bidirectional LSTM on IMDB</a></td>
      <td>2020-05-03</td>
      <td>2020-05-03</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_from_scratch>Text classification from scratch</a></td>
      <td>2019-11-06</td>
      <td>2020-05-17</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/lstm_seq2seq>Character-level recurrent sequence-to-sequence model</a></td>
      <td>2017-09-29</td>
      <td>2020-04-26</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/addition_rnn>Sequence to sequence learning for performing number addition</a></td>
      <td>2015-08-17</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/subclassing_conv_layers>Customizing the convolution operation of a Conv2D layer</a></td>
      <td>2021-11-03</td>
      <td>2021-11-03</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/sklearn_metric_callbacks>Evaluating and exporting scikit-learn metrics in a Keras callback</a></td>
      <td>2021-10-07</td>
      <td>2021-10-07</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/tensorflow_nu_models>Writing Keras Models With TensorFlow NumPy</a></td>
      <td>2021-08-28</td>
      <td>2021-08-28</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/better_knowledge_distillation>Knowledge distillation recipes</a></td>
      <td>2021-08-01</td>
      <td>2021-08-01</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/sample_size_estimate>Estimating required sample size for model training</a></td>
      <td>2021-05-20</td>
      <td>2021-06-06</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/creating_tfrecords>Creating TFRecords</a></td>
      <td>2021-02-27</td>
      <td>2021-02-27</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/memory_efficient_embeddings>Memory-efficient embeddings for recommendation systems</a></td>
      <td>2021-02-15</td>
      <td>2021-02-15</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/bayesian_neural_networks>Probabilistic Bayesian Neural Networks</a></td>
      <td>2021-01-15</td>
      <td>2021-01-15</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/tfrecord>How to train a Keras model on TFRecord files</a></td>
      <td>2020-07-29</td>
      <td>2020-08-07</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/debugging_tips>Keras debugging tips</a></td>
      <td>2020-05-16</td>
      <td>2020-05-16</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/quasi_svm>A Quasi-SVM in Keras</a></td>
      <td>2020-04-17</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/endpoint_layer_pattern>Endpoint layer pattern</a></td>
      <td>2019-05-10</td>
      <td>2019-05-10</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/antirectifier>Simple custom layer example: Antirectifier</a></td>
      <td>2016-01-06</td>
      <td>2020-04-20</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/gat_node_classification>Graph attention network (GAT) for node classification</a></td>
      <td>2021-09-13</td>
      <td>2021-12-26</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/mpnn-molecular-graphs>Message-passing neural network (MPNN) for molecular property prediction</a></td>
      <td>2021-08-16</td>
      <td>2021-12-27</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/gnn_citations>Node Classification with Graph Neural Networks</a></td>
      <td>2021-05-30</td>
      <td>2021-05-30</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/node2vec_movielens>Graph representation learning with node2vec</a></td>
      <td>2021-05-15</td>
      <td>2021-05-15</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/gaugan>GauGAN for conditional image generation</a></td>
      <td>2021-12-26</td>
      <td>2022-01-03</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/adain>Neural Style Transfer with AdaIN</a></td>
      <td>2021-11-08</td>
      <td>2021-11-08</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/gan_ada>Data-efficient GANs with Adaptive Discriminator Augmentation</a></td>
      <td>2021-10-28</td>
      <td>2021-10-28</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/vq_vae>Vector-Quantized Variational Autoencoders</a></td>
      <td>2021-07-21</td>
      <td>2021-07-21</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/conditional_gan>Conditional GAN</a></td>
      <td>2021-07-13</td>
      <td>2021-07-15</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/stylegan>Face image generation with StyleGAN</a></td>
      <td>2021-07-01</td>
      <td>2021-07-01</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/wgan-graphs>WGAN-GP with R-GCN for the generation of small molecular graphs</a></td>
      <td>2021-06-30</td>
      <td>2021-06-30</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/cyclegan>CycleGAN</a></td>
      <td>2020-08-12</td>
      <td>2020-08-12</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/real_nvp>Density estimation using Real NVP</a></td>
      <td>2020-08-10</td>
      <td>2020-08-10</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/text_generation_with_miniature_gpt>Text generation with a miniature GPT</a></td>
      <td>2020-05-29</td>
      <td>2020-05-29</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/pixelcnn>PixelCNN</a></td>
      <td>2020-05-17</td>
      <td>2020-05-23</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/wgan_gp>WGAN-GP overriding `Model.train_step`</a></td>
      <td>2020-05-09</td>
      <td>2020-05-09</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/vae>Variational AutoEncoder</a></td>
      <td>2020-05-03</td>
      <td>2020-05-03</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/dcgan_overriding_train_step>DCGAN to generate face images</a></td>
      <td>2019-04-29</td>
      <td>2021-01-01</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/deep_dream>Deep Dream</a></td>
      <td>2016-01-13</td>
      <td>2020-05-02</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/neural_style_transfer>Neural style transfer</a></td>
      <td>2016-01-11</td>
      <td>2020-05-02</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/lstm_character_level_text_generation>Character-level text generation with LSTM</a></td>
      <td>2015-06-15</td>
      <td>2020-04-30</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/ctc_asr>Automatic Speech Recognition using CTC</a></td>
      <td>2021-09-26</td>
      <td>2021-09-26</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/melgan_spectrogram_inversion>MelGAN-based spectrogram inversion using feature matching</a></td>
      <td>2021-02-09</td>
      <td>2021-09-15</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/transformer_asr>Automatic Speech Recognition with Transformer</a></td>
      <td>2021-01-13</td>
      <td>2021-01-13</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/speaker_recognition_using_cnn>Speaker Recognition</a></td>
      <td>2020-06-14</td>
      <td>2020-03-07</td>
    </tr>
  </tbody>
</table>




```python
# sorted by 'Category' and 'Last modified'

sorted_report_df = report_df.sort_values(by=['Category', 'Last modified'], ascending=False)
rendering_html = sorted_report_df.to_html(render_links=True, escape=False, index=False)
IPython.display.HTML(rendering_html)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>Title</th>
      <th>Date created</th>
      <th>Last modified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/patch_convnet>Augmenting convnets with aggregated attention</a></td>
      <td>2022-01-22</td>
      <td>2022-01-22</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/vivit>Video Vision Transformer</a></td>
      <td>2022-01-12</td>
      <td>2022-01-12</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/vit_small_ds>Train a Vision Transformer on small datasets</a></td>
      <td>2022-01-07</td>
      <td>2022-01-10</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/masked_image_modeling>Masked image modeling with Autoencoders</a></td>
      <td>2021-12-20</td>
      <td>2021-12-21</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/barlow_twins>Barlow Twins for Contrastive SSL</a></td>
      <td>2021-11-04</td>
      <td>2021-12-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/token_learner>Learning to tokenize in Vision Transformers</a></td>
      <td>2021-12-10</td>
      <td>2021-12-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/attention_mil_classification>Classification using Attention-based Deep Multiple Instance Learning (MIL).</a></td>
      <td>2021-08-16</td>
      <td>2021-11-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_captioning>Image Captioning</a></td>
      <td>2021-05-29</td>
      <td>2021-10-31</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mobilevit>MobileViT: A mobile-friendly Transformer-based model for image classification</a></td>
      <td>2021-10-20</td>
      <td>2021-10-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/eanet>Image classification with EANet (External Attention Transformer)</a></td>
      <td>2021-10-19</td>
      <td>2021-10-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/convmixer>Image classification with ConvMixer</a></td>
      <td>2021-10-12</td>
      <td>2021-10-12</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/fixres>FixRes: Fixing train-test resolution discrepancy</a></td>
      <td>2021-10-08</td>
      <td>2021-10-10</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/metric_learning_tf_similarity>Metric learning for image similarity search using TensorFlow Similarity</a></td>
      <td>2021-09-30</td>
      <td>2021-09-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/bit>Image Classification using BigTransfer (BiT)</a></td>
      <td>2021-09-24</td>
      <td>2021-09-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/zero_dce>Zero-DCE for low-light image enhancement</a></td>
      <td>2021-09-18</td>
      <td>2021-09-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mirnet>Low-light image enhancement using MIRNet</a></td>
      <td>2021-09-11</td>
      <td>2021-09-15</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/nnclr>Self-supervised contrastive learning with NNCLR</a></td>
      <td>2021-09-13</td>
      <td>2021-09-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/near_dup_search>Near-duplicate image search</a></td>
      <td>2021-09-10</td>
      <td>2021-09-10</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/swin_transformers>Image classification with Swin Transformers</a></td>
      <td>2021-09-08</td>
      <td>2021-09-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/deeplabv3_plus>Multiclass semantic segmentation using DeepLabV3+</a></td>
      <td>2021-08-31</td>
      <td>2021-09-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/depth_estimation>Monocular depth estimation</a></td>
      <td>2021-08-30</td>
      <td>2021-08-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/handwriting_recognition>Handwriting recognition</a></td>
      <td>2021-08-16</td>
      <td>2021-08-16</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/nerf>3D volumetric rendering with NeRF</a></td>
      <td>2021-08-09</td>
      <td>2021-08-09</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/involution>Involutional neural networks</a></td>
      <td>2021-07-25</td>
      <td>2021-07-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/cct>Compact Convolutional Transformers</a></td>
      <td>2021-06-30</td>
      <td>2021-06-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/adamatch>Semi-supervision and domain adaptation with AdaMatch</a></td>
      <td>2021-06-19</td>
      <td>2021-06-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/gradient_centralization>Gradient Centralization for Better Training Performance</a></td>
      <td>2021-06-18</td>
      <td>2021-06-18</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/cutmix>CutMix data augmentation for image classification</a></td>
      <td>2021-06-08</td>
      <td>2021-06-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/video_transformers>Video Classification with Transformers</a></td>
      <td>2021-06-08</td>
      <td>2021-06-08</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/video_classification>Video Classification with a CNN-RNN Architecture</a></td>
      <td>2021-05-28</td>
      <td>2021-06-05</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/conv_lstm>Next-Frame Video Prediction with Convolutional LSTMs</a></td>
      <td>2021-06-02</td>
      <td>2021-06-05</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mlp_image_classification>Image classification with modern MLP models</a></td>
      <td>2021-05-30</td>
      <td>2021-05-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/learnable_resizer>Learning to Resize in Computer Vision</a></td>
      <td>2021-04-30</td>
      <td>2021-05-13</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/siamese_contrastive>Image similarity estimation using a Siamese Network with a contrastive loss</a></td>
      <td>2021-05-06</td>
      <td>2021-05-06</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/keypoint_detection>Keypoint Detection with Transfer Learning</a></td>
      <td>2021-05-02</td>
      <td>2021-05-02</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/semisupervised_simclr>Semi-supervised image classification using contrastive pretraining with SimCLR</a></td>
      <td>2021-04-24</td>
      <td>2021-04-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/consistency_training>Consistency training with supervision</a></td>
      <td>2021-04-13</td>
      <td>2021-04-19</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/siamese_network>Image similarity estimation using a Siamese Network with a triplet loss</a></td>
      <td>2021-03-25</td>
      <td>2021-03-25</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/simsiam>Self-supervised contrastive learning with SimSiam</a></td>
      <td>2021-03-19</td>
      <td>2021-03-20</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/randaugment>RandAugment for Image Classification for Improved Robustness</a></td>
      <td>2021-03-13</td>
      <td>2021-03-17</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/grad_cam>Grad-CAM class activation visualization</a></td>
      <td>2020-04-26</td>
      <td>2021-03-07</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mixup>MixUp augmentation for image classification</a></td>
      <td>2021-03-06</td>
      <td>2021-03-06</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/autoencoder>Convolutional autoencoder for image denoising</a></td>
      <td>2021-03-01</td>
      <td>2021-03-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/semantic_image_clustering>Semantic Image Clustering</a></td>
      <td>2021-02-28</td>
      <td>2021-02-28</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/perceiver_image_classification>Image classification with Perceiver</a></td>
      <td>2021-04-30</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_with_vision_transformer>Image classification with Vision Transformer</a></td>
      <td>2021-01-18</td>
      <td>2021-01-18</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/supervised-contrastive-learning>Supervised Contrastive Learning</a></td>
      <td>2020-11-30</td>
      <td>2020-11-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/pointnet_segmentation>Point cloud segmentation with PointNet</a></td>
      <td>2020-10-23</td>
      <td>2020-10-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/3D_image_classification>3D image classification from CT scans</a></td>
      <td>2020-09-23</td>
      <td>2020-09-23</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/knowledge_distillation>Knowledge Distillation</a></td>
      <td>2020-09-01</td>
      <td>2020-09-01</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/super_resolution_sub_pixel>Image Super-Resolution using an Efficient Sub-Pixel CNN</a></td>
      <td>2020-07-28</td>
      <td>2020-08-27</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/xray_classification_with_tpus>Pneumonia Classification on TPU</a></td>
      <td>2020-07-28</td>
      <td>2020-08-24</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning>Image classification via fine-tuning with EfficientNet</a></td>
      <td>2020-06-30</td>
      <td>2020-07-16</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/retinanet>Object Detection with RetinaNet</a></td>
      <td>2020-05-17</td>
      <td>2020-07-14</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/captcha_ocr>OCR model for reading Captchas</a></td>
      <td>2020-06-14</td>
      <td>2020-06-26</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/metric_learning>Metric learning for image similarity search</a></td>
      <td>2020-06-05</td>
      <td>2020-06-09</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/integrated_gradients>Model interpretability with Integrated Gradients</a></td>
      <td>2020-06-02</td>
      <td>2020-06-02</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/reptile>Few-Shot learning with Reptile</a></td>
      <td>2020-05-21</td>
      <td>2020-05-30</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/visualizing_what_convnets_learn>Visualizing what convnets learn</a></td>
      <td>2020-05-29</td>
      <td>2020-05-29</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/pointnet>Point cloud classification with PointNet</a></td>
      <td>2020-05-25</td>
      <td>2020-05-26</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/image_classification_from_scratch>Image classification from scratch</a></td>
      <td>2020-04-27</td>
      <td>2020-04-28</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/mnist_convnet>Simple MNIST convnet</a></td>
      <td>2015-06-19</td>
      <td>2020-04-21</td>
    </tr>
    <tr>
      <td>vision</td>
      <td><a href=https://keras.io/examples/vision/oxford_pets_image_segmentation>Image segmentation with a U-Net-like architecture</a></td>
      <td>2019-03-20</td>
      <td>2020-04-20</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_traffic_forecasting>Traffic forecasting using graph neural networks and LSTM</a></td>
      <td>2021-12-28</td>
      <td>2021-12-28</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_classification_transformer>Timeseries classification with a Transformer model</a></td>
      <td>2021-06-25</td>
      <td>2021-08-05</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_classification_from_scratch>Timeseries classification from scratch</a></td>
      <td>2020-07-21</td>
      <td>2021-07-16</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_weather_forecasting>Timeseries forecasting for weather prediction</a></td>
      <td>2020-06-23</td>
      <td>2020-07-20</td>
    </tr>
    <tr>
      <td>timeseries</td>
      <td><a href=https://keras.io/examples/timeseries/timeseries_anomaly_detection>Timeseries anomaly detection using an Autoencoder</a></td>
      <td>2020-05-31</td>
      <td>2020-05-31</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/classification_with_tfdf>Classification with TensorFlow Decision Forests</a></td>
      <td>2022-01-25</td>
      <td>2022-01-25</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/tabtransformer>Structured data learning with TabTransformer</a></td>
      <td>2022-01-18</td>
      <td>2022-01-18</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/wide_deep_cross_networks>Structured data learning with Wide, Deep, and Cross networks</a></td>
      <td>2020-12-31</td>
      <td>2021-05-05</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/classification_with_grn_and_vsn>Classification with Gated Residual and Variable Selection Networks</a></td>
      <td>2021-02-10</td>
      <td>2021-02-10</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/deep_neural_decision_forests>Classification with Neural Decision Forests</a></td>
      <td>2021-01-15</td>
      <td>2021-01-15</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/movielens_recommendations_transformers>A Transformer-based recommendation system</a></td>
      <td>2020-12-30</td>
      <td>2020-12-30</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/structured_data_classification_from_scratch>Structured data classification from scratch</a></td>
      <td>2020-06-09</td>
      <td>2020-06-09</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/collaborative_filtering_movielens>Collaborative Filtering for Movie Recommendations</a></td>
      <td>2020-05-24</td>
      <td>2020-05-24</td>
    </tr>
    <tr>
      <td>structured_data</td>
      <td><a href=https://keras.io/examples/structured_data/imbalanced_classification>Imbalanced classification: credit card fraud detection</a></td>
      <td>2019-05-28</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/ppo_cartpole>Proximal Policy Optimization</a></td>
      <td>2021-06-24</td>
      <td>2021-06-24</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/ddpg_pendulum>Deep Deterministic Policy Gradient (DDPG)</a></td>
      <td>2020-06-04</td>
      <td>2020-09-21</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/deep_q_network_breakout>Deep Q-Learning for Atari Breakout</a></td>
      <td>2020-05-23</td>
      <td>2020-06-17</td>
    </tr>
    <tr>
      <td>rl</td>
      <td><a href=https://keras.io/examples/rl/actor_critic_cartpole>Actor Critic Method</a></td>
      <td>2020-05-13</td>
      <td>2020-05-13</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/question_answering>Question Answering with Hugging Face Transformers</a></td>
      <td>2022-01-13</td>
      <td>2022-01-13</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/active_learning_review_classification>Review Classification using Active Learning</a></td>
      <td>2021-10-29</td>
      <td>2021-10-29</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_generation_fnet>Text Generation using FNet</a></td>
      <td>2021-10-05</td>
      <td>2021-10-05</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/multimodal_entailment>Multimodal entailment</a></td>
      <td>2021-08-08</td>
      <td>2021-08-15</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/ner_transformers>Named Entity Recognition using Transformers</a></td>
      <td>2021-06-23</td>
      <td>2021-06-24</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/neural_machine_translation_with_transformer>English-to-Spanish translation with a sequence-to-sequence Transformer</a></td>
      <td>2021-05-26</td>
      <td>2021-05-26</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_with_switch_transformer>Text classification with Switch Transformer</a></td>
      <td>2020-05-10</td>
      <td>2021-02-15</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/nl_image_search>Natural language image search with a Dual Encoder</a></td>
      <td>2021-01-30</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/multi_label_classification>Large-scale multi-label text classification</a></td>
      <td>2020-09-25</td>
      <td>2020-12-23</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/masked_language_modeling>End-to-end Masked Language Modeling with BERT</a></td>
      <td>2020-09-18</td>
      <td>2020-09-18</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/semantic_similarity_with_bert>Semantic Similarity with BERT</a></td>
      <td>2020-08-15</td>
      <td>2020-08-29</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_extraction_with_bert>Text Extraction with BERT</a></td>
      <td>2020-05-23</td>
      <td>2020-05-23</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_from_scratch>Text classification from scratch</a></td>
      <td>2019-11-06</td>
      <td>2020-05-17</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/text_classification_with_transformer>Text classification with Transformer</a></td>
      <td>2020-05-10</td>
      <td>2020-05-10</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/pretrained_word_embeddings>Using pre-trained word embeddings</a></td>
      <td>2020-05-05</td>
      <td>2020-05-05</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/bidirectional_lstm_imdb>Bidirectional LSTM on IMDB</a></td>
      <td>2020-05-03</td>
      <td>2020-05-03</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/lstm_seq2seq>Character-level recurrent sequence-to-sequence model</a></td>
      <td>2017-09-29</td>
      <td>2020-04-26</td>
    </tr>
    <tr>
      <td>nlp</td>
      <td><a href=https://keras.io/examples/nlp/addition_rnn>Sequence to sequence learning for performing number addition</a></td>
      <td>2015-08-17</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/subclassing_conv_layers>Customizing the convolution operation of a Conv2D layer</a></td>
      <td>2021-11-03</td>
      <td>2021-11-03</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/sklearn_metric_callbacks>Evaluating and exporting scikit-learn metrics in a Keras callback</a></td>
      <td>2021-10-07</td>
      <td>2021-10-07</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/tensorflow_nu_models>Writing Keras Models With TensorFlow NumPy</a></td>
      <td>2021-08-28</td>
      <td>2021-08-28</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/better_knowledge_distillation>Knowledge distillation recipes</a></td>
      <td>2021-08-01</td>
      <td>2021-08-01</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/sample_size_estimate>Estimating required sample size for model training</a></td>
      <td>2021-05-20</td>
      <td>2021-06-06</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/creating_tfrecords>Creating TFRecords</a></td>
      <td>2021-02-27</td>
      <td>2021-02-27</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/memory_efficient_embeddings>Memory-efficient embeddings for recommendation systems</a></td>
      <td>2021-02-15</td>
      <td>2021-02-15</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/bayesian_neural_networks>Probabilistic Bayesian Neural Networks</a></td>
      <td>2021-01-15</td>
      <td>2021-01-15</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/tfrecord>How to train a Keras model on TFRecord files</a></td>
      <td>2020-07-29</td>
      <td>2020-08-07</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/debugging_tips>Keras debugging tips</a></td>
      <td>2020-05-16</td>
      <td>2020-05-16</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/antirectifier>Simple custom layer example: Antirectifier</a></td>
      <td>2016-01-06</td>
      <td>2020-04-20</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/quasi_svm>A Quasi-SVM in Keras</a></td>
      <td>2020-04-17</td>
      <td>2020-04-17</td>
    </tr>
    <tr>
      <td>keras_recipes</td>
      <td><a href=https://keras.io/examples/keras_recipes/endpoint_layer_pattern>Endpoint layer pattern</a></td>
      <td>2019-05-10</td>
      <td>2019-05-10</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/mpnn-molecular-graphs>Message-passing neural network (MPNN) for molecular property prediction</a></td>
      <td>2021-08-16</td>
      <td>2021-12-27</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/gat_node_classification>Graph attention network (GAT) for node classification</a></td>
      <td>2021-09-13</td>
      <td>2021-12-26</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/gnn_citations>Node Classification with Graph Neural Networks</a></td>
      <td>2021-05-30</td>
      <td>2021-05-30</td>
    </tr>
    <tr>
      <td>graph</td>
      <td><a href=https://keras.io/examples/graph/node2vec_movielens>Graph representation learning with node2vec</a></td>
      <td>2021-05-15</td>
      <td>2021-05-15</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/gaugan>GauGAN for conditional image generation</a></td>
      <td>2021-12-26</td>
      <td>2022-01-03</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/adain>Neural Style Transfer with AdaIN</a></td>
      <td>2021-11-08</td>
      <td>2021-11-08</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/gan_ada>Data-efficient GANs with Adaptive Discriminator Augmentation</a></td>
      <td>2021-10-28</td>
      <td>2021-10-28</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/vq_vae>Vector-Quantized Variational Autoencoders</a></td>
      <td>2021-07-21</td>
      <td>2021-07-21</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/conditional_gan>Conditional GAN</a></td>
      <td>2021-07-13</td>
      <td>2021-07-15</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/stylegan>Face image generation with StyleGAN</a></td>
      <td>2021-07-01</td>
      <td>2021-07-01</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/wgan-graphs>WGAN-GP with R-GCN for the generation of small molecular graphs</a></td>
      <td>2021-06-30</td>
      <td>2021-06-30</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/dcgan_overriding_train_step>DCGAN to generate face images</a></td>
      <td>2019-04-29</td>
      <td>2021-01-01</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/cyclegan>CycleGAN</a></td>
      <td>2020-08-12</td>
      <td>2020-08-12</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/real_nvp>Density estimation using Real NVP</a></td>
      <td>2020-08-10</td>
      <td>2020-08-10</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/text_generation_with_miniature_gpt>Text generation with a miniature GPT</a></td>
      <td>2020-05-29</td>
      <td>2020-05-29</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/pixelcnn>PixelCNN</a></td>
      <td>2020-05-17</td>
      <td>2020-05-23</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/wgan_gp>WGAN-GP overriding `Model.train_step`</a></td>
      <td>2020-05-09</td>
      <td>2020-05-09</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/vae>Variational AutoEncoder</a></td>
      <td>2020-05-03</td>
      <td>2020-05-03</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/deep_dream>Deep Dream</a></td>
      <td>2016-01-13</td>
      <td>2020-05-02</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/neural_style_transfer>Neural style transfer</a></td>
      <td>2016-01-11</td>
      <td>2020-05-02</td>
    </tr>
    <tr>
      <td>generative</td>
      <td><a href=https://keras.io/examples/generative/lstm_character_level_text_generation>Character-level text generation with LSTM</a></td>
      <td>2015-06-15</td>
      <td>2020-04-30</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/ctc_asr>Automatic Speech Recognition using CTC</a></td>
      <td>2021-09-26</td>
      <td>2021-09-26</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/melgan_spectrogram_inversion>MelGAN-based spectrogram inversion using feature matching</a></td>
      <td>2021-02-09</td>
      <td>2021-09-15</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/transformer_asr>Automatic Speech Recognition with Transformer</a></td>
      <td>2021-01-13</td>
      <td>2021-01-13</td>
    </tr>
    <tr>
      <td>audio</td>
      <td><a href=https://keras.io/examples/audio/speaker_recognition_using_cnn>Speaker Recognition</a></td>
      <td>2020-06-14</td>
      <td>2020-03-07</td>
    </tr>
  </tbody>
</table>


