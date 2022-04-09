# Exploring Gender Biases in Word2vec
ECE324 Project <br>
Winter 2022

### Authors
- [Nicole Streltsov](https://github.com/NicoleStrel)
- [Emily Traynor](https://github.com/emily0622)
- [Jasmine Zhang](https://github.com/JasmineZhangxyz)

### Contents

1. [Data Collection](#1-data-collection)
2. [Bias Measurements](#2-bias-measurements)
3. [Custom Word2Vec](#3-custom-word2vec)
4. [Bolukbasi et al. Debiasing](#4-bolukbasi-et-al-debiasing)


## 1. Data Collection

Wikipedia:
* **Downloading_data.ipynb**: Jupyter notebook which downloads 2020 Wikipidia articles from Tensorflow's cloud database. ([Database](https://www.tensorflow.org/datasets/catalog/wikipedia#wikipedia20201201en)) ([Code credit](https://github.com/noanabeshima/wikipedia-downloader))
* **wikipedia-en-1000.json**: A sample of a Wikipedia downloaded. We will be using much larger files which will not be stored in this Github in the future.

Gutenburg Books:
* **gutenberg_data.py**: Reads the urls from the gutenburg url files and reads the data inside. These functions are called in Word2Vec Model and Bias Measurements.ipynb.
* Gutenburg URL files ([Source](https://console.cloud.google.com/storage/browser/deepmind-gutenberg;tab=objects?prefix=&forceOnObjectsSortingFiltering=false&pli=1)):
  * **gutenberg-test-urls.csv**: test data urls
  * **gutenberg-train-urls.csv**: train data urls
  * **gutenberg-validation-urls.csv**: validation data urls

## 2. Bias Measurements

Initialization of genism's word2vec model ([Source](https://radimrehurek.com/gensim/models/word2vec.html?fbclid=IwAR2rdN_kXEqMMBNsH-ux_WjIujHiOBOSCKtAg5oBz2KV6aFQPysCDftZI8I#gensim.models.word2vec.Word2Vec)), training of the two datasets on the model, and the three bias measurements (direct, indirect, WEAT) can be found in **Word2Vec Model and Bias Measurements.ipynb**

## 3. Gensim Word Embedding Models

Exploring_Gender_Biases_in_Word2Vec is where the gensim word embedding training occurs for the Gutenberg dataset.
Wiki_Word2Vec_f100_debiasing_Model_and_Bias_Measurements-Copy1.ipynb is where the training occurs for the Wikipedia dataset.

## 4. Storing the Word Embedding Models

The model file stores both the Wikipedia and Gutenberg models. The format is that produced by using the built-in save function in the gensim library.
The embedding folder saves the word embedding dictionaries in a .txt format.

## 5. Custom Word2Vec

A start to our custom word2vec model (to be used in other debiasing techniques) can be found in **Custom Word2Vec.ipynb**

## 6. Bolukbasi et al. Debiasing

Code related to [Bolukbasi et al.](https://arxiv.org/pdf/1607.06520.pdf) debaising is found in the 2016debais folder. See the ReadMe in the folder for more details. ([Code credit](https://github.com/tolga-b/debiaswe))

## 7. Zhao et al. Debiasing

Code related to [Zhao et al.](https://arxiv.org/pdf/1809.01496.pdf) debiasing is found in the 2018debias folder, along with the debiased .txt word embeddings files.

## 8. Savani et al. Debiasing

Code related to [Savani et al.](https://proceedings.neurips.cc/paper/2020/file/1d8d70dddf147d2d92a634817f01b239-Paper.pdf) debiasing is found in the Custome_Word2Vec_2020_debiasing.ipynb file. This file includes the original custome_word2vec code as well as the debiasing technique as the debiasing method requires the word embedding weights.
