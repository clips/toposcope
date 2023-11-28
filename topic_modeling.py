import os, shutil
import spacy
import pandas as pd
import random as rd
from configparser import ConfigParser
from utils import *
from tqdm import tqdm
tqdm.pandas()

rd.seed(42)

#______________________________________________________________________________________________

def main():

#LOAD_CONFIG___________________________________________________________________________________
    config_object = ConfigParser()
    config_object.read('config.ini')
    input_config = config_object["INPUT_CONFIG"] 
    output_config = config_object["OUTPUT_CONFIG"]
    
    if input_config['algorithm'] == 'LDA':
        processing_config = config_object["LDA_CONFIG"]
    elif input_config['algorithm'] == 'NMF':
        processing_config = config_object["NMF_CONFIG"]
    elif input_config['algorithm'] == 'BERTopic':
        processing_config = config_object["BERTOPIC_CONFIG"]
    elif input_config['algorithm'] =='Top2Vec':
        processing_config = config_object["TOP2VEC_CONFIG"]
    else:
        raise KeyError("Please specify one of the following algorithms: 'BERTopic', 'Top2Vec', 'NMF', 'LDA'.")

#LOAD_DATA_____________________________________________________________________________________
    print("Loading data...")

    in_dir = input_config['input']
    input_format = input_config['input_format']
    delimiter = input_config['delimiter']

    df = load_data(in_dir, input_format, delimiter)

    text_column = input_config['text_column']
    if not text_column.strip():
        text_column = 'text'
    df[text_column] = df[text_column].apply(lambda x: str(x))

    # compute topics over time?
    if input_config['timestamp_column'].strip() and input_format == 'csv':
        timestamps = df[input_config['timestamp_column'].strip()].tolist()
    else:
        timestamps = None
            
#PREPROCESSING_________________________________________________________________________________
    print('Preprocessing data...')
    do_preprocess = int(processing_config['preprocess'])

    if do_preprocess:
        tokenize = int(processing_config['tokenize'])
        lemmatize = int(processing_config['lemmatize'])
        remove_nltk_stopwords = int(processing_config['remove_nltk_stopwords'])
        remove_custom_stopwords = processing_config['remove_custom_stopwords'].strip()
        remove_punct = int(processing_config['remove_punct'])
        lowercase = int(processing_config['lowercase'])
        lang = processing_config['lang'].lower()

        #load relevant SpaCy model
        if lang =='dutch':
            nlp = spacy.load("nl_core_news_sm")
        elif lang == 'english':
            nlp = spacy.load("en_core_web_sm")
        elif lang == 'french':
            nlp = spacy.load('fr_core_news_sm')
        elif lang == 'german':
            nlp = spacy.load('de_core_news_sm')
        else:
            raise ValueError(f"'{lang}' is not a valid language, please use one of the following languages: 'Dutch', 'English', 'French', 'German'.")

        print("    Tokenize:", bool(tokenize))
        print("    Lemmatize:", bool(lemmatize))
        print("    Remove NLTK stopwords:", bool(remove_nltk_stopwords))
        print("    Remove custom stopwords:", bool(remove_custom_stopwords))
        print("    Lowercase:", bool(lowercase))
        print("    Remove punctuation:", bool(remove_punct))
        
        df[text_column] = df[text_column].progress_apply(lambda x: preprocess(
            x, 
            nlp, 
            lang, 
            tokenize,
            lemmatize, 
            remove_nltk_stopwords, 
            remove_custom_stopwords, 
            remove_punct, 
            lowercase)
        )
    
#PREPARE_OUTPUT_DIR____________________________________________________________________________
    dir_out = output_config['output_dir']

    # if overwrite_output_dir is True, delete the directory
    # else, check if output dir exists already and return error if it does
    # create the output directory
    if int(output_config['overwrite_output_dir']):
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
    else:
        assert os.path.exists(dir_out) == False

    os.mkdir(dir_out)
    os.mkdir(os.path.join(dir_out, 'visualizations'))

#FIT_MODEL_____________________________________________________________________________________
    if input_config['algorithm'] == 'BERTopic':
        topic_doc_matrix, keyword_df, topic_term_matrix = BERT_topic(df, text_column, dir_out, lang, timestamps)
    
    elif input_config['algorithm'] == 'LDA':
        topic_doc_matrix, keyword_df, topic_term_matrix = LDA_model(df, text_column, dir_out, timestamps)
    
    elif input_config['algorithm'] == 'NMF':
        topic_doc_matrix, keyword_df, topic_term_matrix = NMF_model(df, text_column, dir_out, timestamps)
    
    elif input_config['algorithm'] == 'Top2Vec':
        topic_doc_matrix, keyword_df, topic_term_matrix = top_2_vec(df, text_column, dir_out, timestamps)

    keywords = keyword_df.keywords.tolist()

#EVALUATION____________________________________________________________________________________
    print('Evaluating model...')
    texts = [doc.split() for doc in df[text_column]]
    print('    - Coherence')
    coherence_score = coherence(keywords, texts)
    print('    - Diversity')
    diversity = compute_diversity(keywords)

#SAVE_OUTPUT__________________________________________________________________________________
    print('Generating output...')
    #EVALUATION
    eval_df = pd.DataFrame(data={
        'diversity': [diversity],
        'coherence': [coherence_score]
    })
    eval_df.to_csv(os.path.join(dir_out, 'evaluation.csv'), index=False)
    
    #KEYWORDS PER TOPIC
    keyword_df.to_csv(os.path.join(dir_out, 'keywords_per_topic.csv'), index=False)

    #TOPIC-TERM MATRIX
    topic_term_matrix.sort_index(axis=1, inplace=True)
    topic_term_matrix.to_csv(os.path.join(dir_out, 'topic_term_matrix.csv'))

    #TOPIC-DOC MATRIX
    idx_column = topic_doc_matrix['idx']
    topic_doc_matrix = topic_doc_matrix.drop(columns=['idx'])
    topic_doc_matrix.sort_index(axis=1, inplace=True)
    topic_doc_matrix = pd.concat([idx_column, topic_doc_matrix], axis=1)
    topic_doc_matrix.to_csv(os.path.join(dir_out, 'topic_doc_matrix.csv'), index=False)

    #ANNOTATIONS
    idx_column = topic_doc_matrix['idx']
    topic_doc_matrix = topic_doc_matrix.drop(columns=['idx'])  
    label_column = topic_doc_matrix.apply(lambda row: row.idxmax(), axis=1)
    topic_doc_matrix = pd.concat([idx_column, label_column], axis=1)
    topic_doc_matrix.to_csv(os.path.join(dir_out, 'annotations.csv'), index=False)

    print('Done!')
#______________________________________________________________________________________________
if __name__ == "__main__":
    main()
