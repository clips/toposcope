from configparser import ConfigParser

config_object = ConfigParser()

config_object["INPUT_CONFIG"] = {
    "algorithm": '', #'LDA', 'NMF', 'Top2Vec', 'BERTopic'
    "input": '', #full path to input data
    "input_format": '', #'csv' or 'zip'
    "text_column": '', #only relevant if input_format='csv'
    "delimiter": '' #only relevant if input_format='csv'
}

config_object["BERTOPIC_CONFIG"] = {
    "model": 'doc2vec', # embedding model to use (any HuggingFace model)
    "topic_reduction": 10, # 0: no reduction, else: max. number of topics allowed
    "min_topic_size": 5, # minimum number of documents per topic, must be 1<
    "n_keywords": 10, # number of keywords to extract per topic, must be 1<, will be set to default value of the library if not specified
    "lemmatize": 1, # 0 (no) or 1 (yes)
    "remove_stopwords": 1, # 0 (no) or 1 (yes), removes NLTK's default stopwords, see 'language'
    "remove_custom_stopwords": 0, # 0 (no) or 1 (yes)
    "custom_stopword_list": '', # full path to .txt file with 1 stopword per line, only relevant when "remove_custom_stopwords"=1
    "seed_topic_list": '', # full path to .txt file with seed topics: n keywords from 1 topic per line, separated by a comma
    "remove_punct": 1, # 0 (no) or 1 (yes)
    "lowercase": 1, # 0 (no) or 1 (yes)
    "upper_ngram_range": 2, # upper ngram range for keywords, creates ngrams for range (1, n), must be 0<
    "lang": 'english' # 'dutch', 'english', 'french', 'german'
}

config_object["TOP2VEC_CONFIG"] = {
    "model": 'doc2vec', # Any of the following: "doc2vec", "universal-sentence-encoder", "universal-sentence-encoder-multilingual", "distiluse-base-multilingual-cased"
    "topic_reduction": 0, # 0: no reduction, else: max. number of topics allowed
    "min_topic_size": 2, # minimum number of documents per topic, must be 1<
    "n_keywords": 10, # number of keywords to extract per topic, must be 1<, will be set to default value of the library if not specified
    "lemmatize": 0, # 0 (no) or 1 (yes)
    "remove_stopwords": 0, # 0 (no) or 1 (yes)
    "remove_custom_stopwords": 0, # 0 (no) or 1 (yes)
    "custom_stopword_list": '', # full path to .txt file with 1 stopword per line, only relevant if "remove_custom_stopwords"=1
    "seed_topic_list": '', # full path to .txt with seed topics: n keywords from 1 topic per line, separated by a comma
    "remove_punct": 1, # 0 (no) or 1 (yes)
    "lowercase": 1, # 0 (no) or 1 (yes)
    "upper_ngram_range": 2, #upper ngram range for keywords, creates ngrams for range (1, n)
    "lang": 'english' #'dutch', 'english'
}

config_object["LDA_CONFIG"] = {
    "n_components": 10, # number of topics to detect
    "n_keywords": 10, # number of keywords to extract per topic
    "lemmatize": 0, # 0 (no) or 1 (yes)
    "remove_stopwords": 1, # 0 (no) or 1 (yes)
    "remove_custom_stopwords": 0, # 0 (no) or 1 (yes)
    "custom_stopword_list": '', # full path to .txt file with 1 stopword per line, only relevant if "remove_stopwords"=1
    "remove_punct": 1, # 0 (no) or 1 (yes)
    "lowercase": 1, # 0 (no) or 1 (yes)
    "upper_ngram_range": 1, #upper ngram range for keywords, creates ngrams for range (1, n)
    "lang": 'english' #'dutch', 'english'
}

config_object["NMF_CONFIG"] = {
    "n_components": 10, # number of topics to detect
    "n_keywords": 10, #number of keywords to extract per topic
    "lemmatize": 0, # 0 (no) or 1 (yes)
    "remove_stopwords": 1, # 0 (no) or 1 (yes)
    "remove_custom_stopwords": 0, # 0 (no) or 1 (yes)
    "custom_stopword_list": '', #full path to .txt file with 1 stopword per line
    "remove_punct": 1, # 0 (no) or 1 (yes)
    "lowercase": 1, # 0 (no) or 1 (yes)
    "upper_ngram_range": 1, #upper ngram range for keywords, creates ngrams for range (1, n)
    "lang": 'english' # 'dutch', 'english'
}

config_object["OUTPUT_CONFIG"] = {
    "output_dir": './output',
    "overwrite_output_dir": 1
}

with open('config.ini', 'w') as conf:
    config_object.write(conf)
