
from configparser import ConfigParser

config_object = ConfigParser()

config_object["PROCESSING_CONFIG"] = {
    "model": '',
    "n_keywords": 10,
    "generate_wordclouds": True,
    "lemmatize":True,
    "remove_stopwords":True,
}

config_object["INPUT_CONFIG"] = {
    "input": 'repdata.csv',
    "input_format": 'csv',
    "text_column": 'text' #if input_format=csv
}

config_object["OUTPUT_CONFIG"] = {
    "output_dir": './output',
    "overwrite_output_dir": True
}

with open('config.ini', 'w') as conf:
    config_object.write(conf)
