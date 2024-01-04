## Documentation

This repository contains code for the topic modeling pipeline for the CLARIAH-VL digital text analysis dashboard and pipeline (DTADP). 

### Installation

1. Create a new conda environment: ```conda create -n {name here} python=3.9.18```
2. Clone the repository: ```git clone https://github.com/LemmensJens/CLARIAH-topic.git```
3. Install the requirements: ```pip3 install -r requirements.txt```
4. Download NLTK's stopwords: ```python -m nltk.downloader stopwords```

### Quick start
1. Set up the configuration file
2. Run the pipeline by calling ```python -u topic_modeling.py```

### Demo
1. Run ```python get_demo_data.py``` to retrieve the 20 NewsGroups dataset.
2. Initialize the config file with the default settings by calling ```python create_config.py```
3. Run the pipeline: ```python -u topic_modeling.py```

### Pipeline usage and overview

#### Set up config file
In ```create_config.py```, specify the following:
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
- remove custom stopwords
- remove punctuation
- lowercase text
- use ngrams
- language (relevant for tokenization/lemmatization, and removal of NLTK stopwords)

#### Run the pipeline
Create the config file by running ```python create_config.py``` and start the pipeline with ```python -u topic_modeling.py```.

#### Output
```evaluation.csv``` contains diversity and coherence scores

```keywords_per_topic.csv``` shows top n most important keywords per topic

```topic_doc_matrix.csv``` matrix showing how much each document relates to each topic

```topic_term_matrix.csv``` matrix containing the weights of all tokens with respect to the topics

```visualizations``` folder containing html visualizations of the topic modeling results

### User Guidelines
#### Algorithm selection
The user has the option to choose between two types of topic modeling architectures: neural (Top2Vec/BERTopic) and classical (LDA/NMF). Generally, it is recommended to use the neural algorithms, as they tend to produce better results. Since Top2Vec is the fastest of the two neural architectures, it is presented as the default option. The guidelines below should give you an intuition of when to use which model:
- Top2Vec: State-of-the-art, high-speed algorithm (default option). Text representations can be based on a pre-trained transformer, or new embeddings can be generated from the input corpus (by setting base model to "Doc2Vec" in the config file). The latter can be useful when it is expected that the type of language used in the corpus is not represented in the pre-trained language models that Top2Vec provides. Note that the corpus should have a substantial size in that case.
- BERTopic: State-of-the-art algorithm that can be used with any model on the HuggingFace transformers hub: https://huggingface.co/models. This is useful when you want to use a publicly available model that is trained on the same (or similar) data to your corpus.
- NMF/LDA: Generally perform worse than Top2Vec and BERTopic, but worth investigating when working with a large corpus (+10 000 items) that contains texts that are very long or when it is expected that the type of language in the corpus is not well-represented in the pre-trained models that the neural algorithms are based on, such as historical texts.

#### Preprocessing
When using the classical machine learning algorithms (NMF/LDA), it is recommended to apply all preprocessing steps provided in the pipeline (tokenization, lemmatization, lowercasing, and removing stopwords and punctuation). For the neural models, it is not required, since they rely on more sophisticated methods, but experimenting with different preprocessing steps could still result in improvements.

#### Model parameter tuning and evaluation of the results
Which model and hyperparameters are optimal depends on the data that is used. Therefore, optimization experiments are necessary to find the best configuration. To evaluate the results of the topic modeling algorithm, it is important to investigate both the quantitative results - the diversity and coherence scores - but also the qualitative results by looking at the individual topic predictions, visualizations, and the most important keywords per topic. 
