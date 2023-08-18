### Introduction

This repository contains code for the topic modeling pipeline for the CLARIAH-VL DTADP. 

### Installation

Create and activate a conda python 3.9 environment: ```conda create -n {name here} python=3.9```

Clone the repository: ```git clone https://github.com/LemmensJens/CLARIAH-topic.git```

Afterwards, install the requirements with ```pip install -r requirements.txt```. 

Then, install Spacy's models with ```python -m spacy download nl_core_news_sm``` and ```python -m spacy download en_core_web_sm```. 

Finally, download NLTK's stopwords with ```python -m nltk.downloader stopwords```.

### Usage
Set up the configuration file (create_config.py) and run ```python topic_modeling.py```.
