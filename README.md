This repository contains code for the topic modeling pipeline for the CLARIAH-VL DTADP. 
First, install Spacy's models with ```python -m spacy download nl_core_news_lg``` and ```python -m spacy download en_core_web_lg```,  and the other requirements with ```pip install -r requirements.txt```.
For a demonstration of the code, add a corpus to the config file and run ```python topic_modeling.py``` (specify data directory in config).
