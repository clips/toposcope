from configparser import ConfigParser

config_object = ConfigParser()

config_object["INPUT_CONFIG"] = {
    "algorithm": 'BERTopic', #'LDA', 'NMF', 'Top2Vec', 'BERTopic'
    "input": '../data/frenk_train_no_labels.csv', #full path to input data
    "input_format": 'csv', #'csv', 'xlsx', 'txt'
    "text_column": 'text' #only relevant if input_format='csv' or 'xlsx'
}

config_object["BERTOPIC_CONFIG"] = {
    "model": '', #to do
    "topic_reduction": 0, #0 -> no reduction, 0< -> max. number of topics allowed
    "min_topic_size": 2, #>1
    "n_keywords": 10, #number of keywords to extract per topic
    "lemmatize": 0, #0 or 1
    "remove_stopwords": 1, #0 or 1
    "custom_stopword_list": '', #full path to .txt file with 1 stopword per line
    "seed_topic_list": '', #full path to .txt with seed topics: n keywords from 1 topic per line, separated by a comma
    "remove_punct": 1, #0 or 1
    "lowercase": 1, #0 or 1
    "upper_ngram_range": 1, #upper ngram range for keywords, creates ngrams for range (1, n)
    "index_column": 'None', #column containing document indices to use for topic annotations (only relevant when using csv/xlsx as input)
    "lang": 'english' #'dutch', 'english'
}

config_object["TOP2VEC_CONFIG"] = {
    "model": '', #to do
    "topic_reduction": 0, #0 -> no reduction, 0< -> max. number of topics allowed
    "min_topic_size": 2, #>1
    "n_keywords": 10, #number of keywords to extract per topic
    "lemmatize": 0, #0 or 1
    "remove_stopwords": 1, #0 or 1
    "custom_stopword_list": '', #full path to .txt file with 1 stopword per line
    "seed_topic_list": '', #full path to .txt with seed topics: n keywords from 1 topic per line, separated by a comma
    "remove_punct": 1, #0 or 1
    "lowercase": 1, #0 or 1
    "upper_ngram_range": 1, #upper ngram range for keywords, creates ngrams for range (1, n)
    "index_column": 'None', #column containing document indices to use for topic annotations (only relevant when using csv/xlsx as input)
    "lang": 'english' #'dutch', 'english'
}

config_object["LDA_CONFIG"] = {
    "n_components": 10, #0 -> no reduction, 0< -> max. number of topics allowed
    "n_keywords": 10, #number of keywords to extract per topic
    "lemmatize": 0, #0 or 1
    "remove_stopwords": 1, #0 or 1
    "custom_stopword_list": '', #full path to .txt file with 1 stopword per line
    "remove_punct": 1, #0 or 1
    "lowercase": 1, #0 or 1
    "upper_ngram_range": 1, #upper ngram range for keywords, creates ngrams for range (1, n)
    "index_column": 'None', #column containing document indices to use for topic annotations (only relevant when using csv/xlsx as input)
    "lang": 'english' #'dutch', 'english'
}

config_object["NMF_CONFIG"] = {
    "n_components": 10, #0 -> no reduction, 0< -> max. number of topics allowed
    "n_keywords": 10, #number of keywords to extract per topic
    "lemmatize": 0, #0 or 1
    "remove_stopwords": 1, #0 or 1
    "custom_stopword_list": '', #full path to .txt file with 1 stopword per line
    "remove_punct": 1, #0 or 1
    "lowercase": 1, #0 or 1
    "upper_ngram_range": 1, #upper ngram range for keywords, creates ngrams for range (1, n)
    "index_column": 'None', #column containing document indices to use for topic annotations (only relevant when using csv/xlsx as input)
    "lang": 'english' #'dutch', 'english'
}

config_object["OUTPUT_CONFIG"] = {
    "output_dir": './output',
    "overwrite_output_dir": 1
}

with open('config.ini', 'w') as conf:
    config_object.write(conf)
