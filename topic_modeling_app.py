import os, shutil, uuid
import spacy
import pandas as pd
import random as rd
import gradio as gr
from configparser import ConfigParser
from utils import *
from tqdm import tqdm
tqdm.pandas()

rd.seed(42)

#______________________________________________________________________________________________

def main(file, lang, algorithm, preprocessing_steps, model, min_topic_size, timestamp_col, n_topics, upper_ngram_range, progress=gr.Progress(track_tqdm=True)):
    
    unique_output_id = str(uuid.uuid4())

    min_topic_size = int(min_topic_size)
    n_topics = int(n_topics)
    upper_ngram_range = int(n_topics)

#LOAD_DATA_____________________________________________________________________________________
    print("Loading data...")
    input_format = file[-3:].lower()
    df = load_data(file, input_format, ',')
    if input_format == 'csv' and timestamp_col.strip():
        timestamps = df[timestamp_col]
    else:
        timestamps = None
    df['text'] = df['text'].apply(lambda x: str(x))
            
#PREPROCESSING_________________________________________________________________________________
    print('Preprocessing data...')

    if preprocessing_steps:

        tokenize = True if 'tokenize' in preprocessing_steps or 'lemmatize' in preprocessing_steps else False
        lemmatize = True if 'lemmatize' in preprocessing_steps else False
        remove_nltk_stopwords = True if 'remove NLTK stopwords' in preprocessing_steps else False
        remove_custom_stopwords = True if 'remove custom stopwords' in preprocessing_steps else False
        lowercase = True if 'lowercase' in preprocessing_steps else False
        remove_punct = True if 'remove punctuation' in preprocessing_steps else False

        #load relevant SpaCy model
        if tokenize or lemmatize:
            if lang =='Dutch':
                nlp = spacy.load("nl_core_news_lg")
            elif lang == 'English':
                nlp = spacy.load("en_core_web_lg")
            elif lang == 'French':
                nlp = spacy.load('fr_core_news_lg')
            elif lang == 'German':
                nlp = spacy.load('de_core_news_lg')
            else:
                raise ValueError(f"'{lang}' is not a valid language, please use one of the following languages: 'Dutch', 'English', 'French', 'German'.")
        else:
            nlp = None

        print("    Tokenize:", tokenize)
        print("    Lemmatize:", lemmatize)
        print("    Remove NLTK stopwords:", remove_nltk_stopwords)
        print("    Remove custom stopwords:", remove_custom_stopwords)
        print("    Lowercase:", lowercase)
        print("    Remove punctuation:", remove_punct)
        
        df['text'] = df['text'].progress_apply(lambda x: preprocess(
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
    # first check if directory where all outputs are stored exists
    main_dir_out = 'outputs'
    if not os.path.exists(main_dir_out):
        os.mkdir(main_dir_out)

    # then create unique output dir for process
    unique_dir_out = os.path.join(main_dir_out, unique_output_id)
    if os.path.exists(unique_dir_out): # this should not happen in theory
        shutil.rmtree(unique_dir_out)
    os.mkdir(unique_dir_out)
    os.mkdir(os.path.join(unique_dir_out, 'visualizations'))

#FIT_MODEL_____________________________________________________________________________________
    if algorithm == 'BERTopic':
        topic_doc_matrix, keyword_df, topic_term_matrix, doc_plot = BERT_topic(df, model, 'text', unique_dir_out, lang, upper_ngram_range, min_topic_size, n_topics, input_format, timestamps)
    
    elif algorithm == 'LDA':
        topic_doc_matrix, keyword_df, topic_term_matrix, doc_plot = LDA_model(df, 'text', unique_dir_out, upper_ngram_range, n_topics, input_format, timestamps)
    
    elif algorithm == 'NMF':
        topic_doc_matrix, keyword_df, topic_term_matrix, doc_plot = NMF_model(df, 'text', unique_dir_out, upper_ngram_range, n_topics, input_format, timestamps)
    
    elif algorithm == 'Top2Vec':
        topic_doc_matrix, keyword_df, topic_term_matrix, doc_plot = top_2_vec(df, 'text', model, unique_dir_out, n_topics, input_format, timestamps)

    keywords = keyword_df.keywords.tolist()

#EVALUATION____________________________________________________________________________________
    print('Evaluating model...')
    texts = [doc.split() for doc in df['text']]
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
    eval_df.to_csv(os.path.join(unique_dir_out, 'evaluation.csv'), index=False)
    
    #KEYWORDS PER TOPIC
    keyword_df.to_csv(os.path.join(unique_dir_out, 'keywords_per_topic.csv'), index=False)

    #TOPIC-TERM MATRIX
    topic_term_matrix.sort_index(axis=1, inplace=True)
    topic_term_matrix.to_csv(os.path.join(unique_dir_out, 'topic_term_matrix.csv'))

    #TOPIC-DOC MATRIX
    idx_column = topic_doc_matrix['idx']
    topic_doc_matrix = topic_doc_matrix.drop(columns=['idx'])
    topic_doc_matrix.sort_index(axis=1, inplace=True)
    topic_doc_matrix = pd.concat([idx_column, topic_doc_matrix], axis=1)
    topic_doc_matrix.to_csv(os.path.join(unique_dir_out, 'topic_doc_matrix.csv'), index=False)

    #ANNOTATIONS
    idx_column = topic_doc_matrix['idx']
    topic_doc_matrix = topic_doc_matrix.drop(columns=['idx'])  
    label_column = topic_doc_matrix.apply(lambda row: row.idxmax(), axis=1)
    label_column.name = 'main topic'
    topic_doc_matrix = pd.concat([idx_column, label_column], axis=1)
    topic_doc_matrix.to_csv(os.path.join(unique_dir_out, 'annotations.csv'), index=False)

    print('Done!')

    return (
        shutil.make_archive(base_name=os.path.join(unique_dir_out), format='zip', base_dir=unique_dir_out),
        doc_plot,
    )
#______________________________________________________________________________________________
if __name__ == "__main__":
    main()
