## Documentation

This repository contains code for the topic modeling pipeline for the CLARIAH-VL digital text analysis dashboard and pipeline (DTADP). 

### Installation

1. Create a new conda environment: ```conda create -n {name here} python=3.9.18```
2. Clone the repository: ```git clone https://github.com/LemmensJens/CLARIAH-topic.git```
3. Install the requirements: ```pip3 install -r requirements.txt```
4. Download NLTK's stopwords: ```python -m nltk.downloader stopwords```

### Pipeline overview

![Alt text](clariah_topic_pipeline.png)

### User interface
To run the pipeline in a Gradio User Interface, run python app.py to host the UI locally.

### From the command line
#### Set up config file
When using the tool from the command line, specify the following in ```create_config.py```:
- which algorithm to use: BERTopic, Top2Vec, LDA or NMDF
- input format (.csv file or .zip containing .txt)
- in case of .csv: the column name containing the text data and the delimiter
- the directory to the data

Then, specify parameters related to the chosen algorithm:
- Base model (for BERTopic and Top2Vec)
- Number of topics
- ...

And specify which preprocessing steps to apply to your data:
- tokenize text
- lemmatize text
- remove NLTK stopwords
- remove custom stopwords (specify file path to .txt containing one stopword per line)
- remove punctuation
- lowercase text
- use ngrams
- language (relevant for tokenization/lemmatization, and removal of NLTK stopwords)

#### Run the pipeline
Create the config file by running ```python create_config.py``` and start the pipeline with ```python topic_modeling.py```.

#### Output
```evaluation.csv``` contains diversity and coherence scores

```keywords_per_topic.csv``` shows top n most important keywords per topic

```topic_doc_matrix.csv``` matrix showing how much each document relates to each topic

```topic_term_matrix.csv``` matrix containing the weights of all tokens with respect to the topics

```visualizations``` folder containing html visualizations of the topic modeling results

### User Guidelines
#### Algorithm selection
- Top2Vec* (Angelov, 2020): State-of-the-art neural algorithm (default option). Text representations can be based on a pre-trained transformer, or new embeddings can be generated from the input corpus (by setting base model to "Doc2Vec" in the config file). The latter can be useful if the corpus has a substantial size and when it is expected that the type of language used in the corpus is not represented in the pre-trained language models that Top2Vec features: 'universal-sentence-encoder', 'universal-sentence-encoder-multilingual', 'distiluse-base-multilingual-cased', 'all-MiniLM-L6-v2', and 'paraphrase-multilingual-MiniLM-L12-v2'. 
- BERTopic*: State-of-the-art neural algorithm that can be used with any pre-trained transformer model on the HuggingFace hub: https://huggingface.co/models. This is useful when you want to use a model that is trained on the same (or similar) data as your corpus. If not specified, the default model that is used is "all-MiniLM-L6-v2".
- NMF/LDA: Generally perform worse than Top2Vec and BERTopic, but worth investigating when working with a large corpus (+10 000 items) that contains texts that are very long or when it is expected that the type of language in the corpus is not well-represented in the pre-trained models that the neural algorithms are based on, such as historical texts.

*The selection of the base model should, a.o. depend on the language of your data. If no model is available for that language, or if your corpus contains texts written in multiple languages, it is recommended to use a multilingual model.

#### Preprocessing
When using the classical machine learning algorithms (NMF/LDA), it is recommended to apply all preprocessing steps provided in the pipeline (tokenization, lemmatization, lowercasing, and removing stopwords and punctuation). For the neural models, it is not required, since they rely on more sophisticated methods, but experimenting with different preprocessing steps could still result in improvements.

#### Model parameter tuning and evaluation of the results
Which model and hyperparameters are optimal depends on the data that is used. Therefore, optimization experiments are necessary to find the best configuration. To evaluate the results of the topic modeling algorithm, it is important to investigate both the quantitative results - the diversity and coherence scores - but also the qualitative results by looking at the individual topic predictions, visualizations, and the most important keywords per topic. 

#### Automatic evaluation
In order to help the user evaluate the topic model, diversity and coherence scores are computed. The diversity score, computed as the proportion of unique words, indicates how "diverse" the different topics are from one another. A diversity of score close to 0 indicates redundant topics, whereas a score of 1 indicates high diversity between topics. Coherence, on the other hand, indicates how frequently words within a topic co-occur. The coherence score can range from -14 to +14 (higher is better).
