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
