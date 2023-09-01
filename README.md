### Introduction

This repository contains code for the topic modeling pipeline for the CLARIAH-VL DTADP. 

### Installation

Create and activate a conda python 3.9 environment: ```conda create -n {name here} python=3.9```

Clone the repository: ```git clone https://github.com/LemmensJens/CLARIAH-topic.git```

Afterwards, install the requirements with ```pip install -r requirements.txt```. 

Finally, download NLTK's stopwords with ```python -m nltk.downloader stopwords```.

### Usage
Set up the configuration file (create_config.py) and run ```python topic_modeling.py```.

### To do
- Output topic-term matrix for Top2Vec: API only returns a maximum of top 50 most important words (overall; not per topic)
- Two-dimensional visualizations for topic clusters / hierarchies and for individual documents (interactive visualizations or static? tbd).

