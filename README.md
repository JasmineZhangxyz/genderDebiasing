# Exploring Gender Biases in Word2vec
ECE324 Project <br>
Winter 2022

### Authors
- [Nicole Streltsov](https://github.com/NicoleStrel)
- [Emily Traynor](https://github.com/emily0622)
- [Jasmine Zhang](https://github.com/JasmineZhangxyz)

### Contents

1. [Data Collection](#1-data-collection)
2. [Gensim Word Embedding Model Training](#2-gensim-word-embedding-model-training)
3. [Custom Word2Vec Definition and Training](#3-custom-word2vec-definition-and-training)
4. [Storing the Word Embedding Models](#4-storing-the-word-embedding-models)
5. [Bolukbasi et al. Debiasing](#5-bolukbasi-et-al-debiasing)
6. [Zhao et al. Debiasing](#6-zhao-et-al-debiasing)
7. [Savani et al. Debiasing](#7-savani-et-al-debiasing)
8. [Bias Measurements](#8-bias-measurements)

## 1. Data Collection

Wikipedia:
* **wiki/Downloading_wiki.ipynb**: Jupyter notebook which downloads 2020 Wikipidia articles from Tensorflow's cloud database. ([Database](https://www.tensorflow.org/datasets/catalog/wikipedia#wikipedia20201201en)) ([Code credit](https://github.com/noanabeshima/wikipedia-downloader))

Gutenburg Books:
* **gutenberg/gutenberg_data.py**: Reads the urls from the gutenburg url files and reads the data inside. These functions are called in Word2Vec Model and Bias Measurements.ipynb.
* Gutenburg URL files ([Source](https://console.cloud.google.com/storage/browser/deepmind-gutenberg;tab=objects?prefix=&forceOnObjectsSortingFiltering=false&pli=1)):
  * **gutenberg/gutenberg-test-urls.csv**: test data urls
  * **gutenberg/gutenberg-train-urls.csv**: train data urls
  * **gutenberg/gutenberg-validation-urls.csv**: validation data urls

## 2. Gensim Word Embedding Model Training

- **Exploring_Gender_Biases_in_Word2Vec.ipynb** is where the gensim word2vec model is defined ([Source](https://radimrehurek.com/gensim/models/word2vec.html?fbclid=IwAR2rdN_kXEqMMBNsH-ux_WjIujHiOBOSCKtAg5oBz2KV6aFQPysCDftZI8I#gensim.models.word2vec.Word2Vec)) and embedding training occurs for the Gutenberg dataset.
- **Wiki_Word2Vec_f100_Training.ipynb** is where the training occurs for the Wikipedia dataset.

## 3. Custom Word2Vec Definition and Training

- The custom word2vec model instance, along with the random pertubution algorithm functions can be found in **custom_word2vec.py**. 
- The training for the custom model was done in the main notebook, **Exploring_Gender_Biases_in_Word2Vec.ipynb**

## 4. Storing the Word Embedding Models 

- **/models:** For the genism models, this folder stores the Wikipedia and Gutenberg models produced by the built-in save function in the gensim library. For the custom models, the files where saved using pickle, but were not added to this github due to the large file size. 
- **/embeddings/**:The embedding folder saves the word embedding dictionaries (word, embeddings on each line) in a .txt format.

## 5. Bolukbasi et al. Debiasing

Code related to [Bolukbasi et al.](https://arxiv.org/pdf/1607.06520.pdf) debaising is found in the 2016debais folder. See the ReadMe in the folder for more details. ([Code credit](https://github.com/tolga-b/debiaswe))

## 6. Zhao et al. Debiasing

Code related to [Zhao et al.](https://arxiv.org/pdf/1809.01496.pdf) debiasing is found in the 2018debias folder, along with the debiased .txt word embeddings files.

## 7. Savani et al. Debiasing

Code related to [Savani et al.](https://proceedings.neurips.cc/paper/2020/file/1d8d70dddf147d2d92a634817f01b239-Paper.pdf) debiasing is found in the main notebook, **Exploring_Gender_Biases_in_Word2Vec.ipynb**, right after the custom word2vec model is trained. 

## 8. Bias Measurements

The three bias measurements (direct, indirect, WEAT) and their results can be found in **Exploring_Gender_Biases_in_Word2Vec.ipynb**

