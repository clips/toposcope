from configparser import ConfigParser

config_object = ConfigParser()

config_object["PROCESSING_CONFIG"] = {
    "model": '',
    "topic_reduction": 0,
    "n_keywords": 10,
    "lemmatize": 0,
    "remove_stopwords": 0,
    "upper_ngram_range": 1,
    "index_column": 'None',
}

config_object["INPUT_CONFIG"] = {
    "input": 'repdata.csv',
    "input_format": 'csv',
    "text_column": 'text' #if input_format=csv
}

config_object["OUTPUT_CONFIG"] = {
    "output_dir": './output',
    "overwrite_output_dir": 1}

with open('config.ini', 'w') as conf:
    config_object.write(conf)
