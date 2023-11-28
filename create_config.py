from configparser import ConfigParser

config_object = ConfigParser()

config_object["INPUT_CONFIG"] = {
    "algorithm": 'Top2Vec', #'LDA', 'NMF', 'Top2Vec', 'BERTopic'
    "input": './demo/demo_data.csv', #full path to input data
    "input_format": 'csv', #'csv' or 'zip'
    "text_column": 'text', #only relevant if input_format='csv'
    "timestamp_column": '', #only relevent if input_format='csv', computes topics over time if provided
    "delimiter": ',' #only relevant if input_format='csv'
}

config_object["BERTOPIC_CONFIG"] = {
    "model": '', # embedding model to use (any HuggingFace model); empty will result in default model 
    "topic_reduction": 20, # 0: no reduction, else: max. number of topics allowed
    "min_topic_size": 2, # minimum number of documents per topic, must be 1<
    "preprocess": 1, # 0 (no) or 1 (yes)
    "tokenize": 1, # 0 (no) or 1 (yes), must be 1 if 'lemmatize' is 1 or will result in error
    "lemmatize": 0, # 0 (no) or 1 (yes)
    "remove_nltk_stopwords": 1, # 0 (no) or 1 (yes), removes NLTK's default stopwords, see 'lang' config parameter
    "remove_custom_stopwords": '', # if not emtpy: full path to .txt file with 1 stopword per line
    "remove_punct": 1, # 0 (no) or 1 (yes)
    "lowercase": 1, # 0 (no) or 1 (yes)
    "upper_ngram_range": 1, # upper ngram range for keywords, creates ngrams for range (1, n), must be 0<
    "lang": 'english' # 'dutch', 'english', 'french', 'german'; relevant for stopwords, tokenization, lemmatization, embedding model selection (if not specified above)
}

config_object["TOP2VEC_CONFIG"] = {
    "model": 'universal-sentence-encoder', # Any of the following: "doc2vec", "universal-sentence-encoder", "all-MiniLM-L6-v2", "distiluse-base-multilingual-cased", "paraphrase-multilingual-MiniLM-L12-v2" only use "universal-sentence-encoder" for English data
    "topic_reduction": 20, # 0: no reduction, else: max. number of topics allowed
    "min_topic_size": 2, # minimum number of documents per topic, must be 1<
    "preprocess": 1, # 0 (no) or 1 (yes)
    "tokenize": 1, # 0 (no) or 1 (yes), must be 1 if 'lemmatize' is 1 or will result in error
    "lemmatize": 0, # 0 (no) or 1 (yes)
    "remove_nltk_stopwords": 1, # 0 (no) or 1 (yes), removes NLTK's default stopwords, see 'lang' config parameter
    "remove_custom_stopwords": '', # if not empty: full path to .txt file with 1 stopword per line
    "remove_punct": 1, # 0 (no) or 1 (yes)
    "lowercase": 1, # 0 (no) or 1 (yes)
    "upper_ngram_range": 1, #upper ngram range for keywords, creates ngrams for range (1, n)
    "lang": 'english' # 'dutch', 'english', 'french', 'german'; relevant for stopwords, tokenization, lemmatization, embedding model selection
}

config_object["LDA_CONFIG"] = {
    "n_components": 20, # number of topics to detect
    "preprocess": 1, # 0 (no) or 1 (yes)
    "tokenize": 1, # 0 (no) or 1 (yes), must be 1 if 'lemmatize' is 1 or will result in error
    "lemmatize": 0, # 0 (no) or 1 (yes)
    "remove_nltk_stopwords": 1, # 0 (no) or 1 (yes), removes NLTK's default stopwords, see 'lang' config parameter
    "remove_custom_stopwords": '', # if not emtpy: full path to .txt file with 1 stopword per line
    "remove_punct": 1, # 0 (no) or 1 (yes)
    "lowercase": 1, # 0 (no) or 1 (yes)
    "upper_ngram_range": 1, #upper ngram range for keywords, creates ngrams for range (1, n)
    "lang": 'english' # 'dutch', 'english', 'french', 'german'; relevant for stopwords, tokenization, lemmatization
}

config_object["NMF_CONFIG"] = {
    "n_components": 20, # number of topics to detect
    "preprocess": 1, # 0 (no) or 1 (yes)
    "tokenize": 1, # 0 (no) or 1 (yes), must be 1 if 'lemmatize' is 1 or will result in error
    "lemmatize": 0, # 0 (no) or 1 (yes)
    "remove_nltk_stopwords": 1, # 0 (no) or 1 (yes), removes NLTK's default stopwords, see 'lang' config parameter
    "remove_custom_stopwords": '', # if not emtpy: full path to .txt file with 1 stopword per line
    "remove_punct": 1, # 0 (no) or 1 (yes)
    "lowercase": 1, # 0 (no) or 1 (yes)
    "upper_ngram_range": 1, #upper ngram range for keywords, creates ngrams for range (1, n)
    "lang": 'english' # 'dutch', 'english', 'french', 'german'; relevant for stopwords, tokenization, lemmatization
}

config_object["OUTPUT_CONFIG"] = {
    "output_dir": './demo/output',
    "overwrite_output_dir": 1
}

with open('config.ini', 'w') as conf:
    config_object.write(conf)
