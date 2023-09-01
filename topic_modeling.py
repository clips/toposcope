import os
import pandas as pd
from configparser import ConfigParser
from utils import *

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
    in_dir = input_config['input']
    input_format = input_config['input_format']
    delimiter = input_config['delimiter']
    df = load_data(in_dir, input_format, delimiter)
    text_column = input_config['text_column']
    if not text_column.strip():
        text_column = 'text'
            
#PREPROCESSING_________________________________________________________________________________
    lemmatize = int(processing_config['lemmatize'])
    remove_stopwords = int(processing_config['remove_stopwords'])
    remove_custom_stopwords = int(processing_config['remove_custom_stopwords'])
    remove_punct = int(processing_config['remove_punct'])
    lowercase = int(processing_config['lowercase'])

    df[text_column] = df[text_column].apply(lambda x: preprocess(x, lemmatize, remove_stopwords, remove_custom_stopwords, remove_punct, lowercase))
    
#PREPARE_OUTPUT_DIR____________________________________________________________________________
    dir_out = output_config['output_dir']
    if not int(output_config['overwrite_output_dir']):
        assert os.path.exists(dir_out) == False

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
        os.mkdir(dir_out+'/topic_term_weights/')

#FIT_MODEL_____________________________________________________________________________________
    if input_config['algorithm'] == 'BERTopic':
        topic_doc_matrix, keyword_df, topic_term_matrix = BERT_topic(df, text_column, dir_out)
    
    elif input_config['algorithm'] == 'LDA':
        topic_doc_matrix, keyword_df, topic_term_matrix = LDA_model(df[text_column], dir_out)
    
    elif input_config['algorithm'] == 'NMF':
        topic_doc_matrix, keyword_df, topic_term_matrix = NMF_model(df[text_column], dir_out)
    
    elif input_config['algorithm'] == 'Top2Vec':
        topic_doc_matrix, keyword_df = top_2_vec(df, text_column, dir_out)

    keywords = keyword_df.keywords.tolist()

#EVALUATION____________________________________________________________________________________
    texts = [doc.split() for doc in df[text_column]]
    coherence_score = coherence(keywords, texts)
    diversity = proportion_unique_words(keywords)

#SAVE_OUTPUT__________________________________________________________________________________

    #EVALUATION
    eval_df = pd.DataFrame(data={
        'diversity': [diversity],
        'coherence': [coherence_score]
    })
    eval_df.to_csv(os.path.join(dir_out, 'evaluation.csv'), index=False)
    
    #KEYWORDS PER TOPIC
    keyword_df.to_csv(os.path.join(dir_out, 'keywords_per_topic.csv'), index=False)

    #TOPIC-TERM MATRIX
    topic_term_matrix.to_csv(os.path.join(dir_out, 'topic_term_matrix.csv'))

    #ANNOTATIONS
    topic_doc_matrix.to_csv(os.path.join(dir_out, 'topic_doc_matrix.csv'), index=False)

    #VISUALIZATION (to do)
    #   - wordclouds per topic
    generate_bar_charts(topic_term_matrix, dir_out)

#______________________________________________________________________________________________
if __name__ == "__main__":
    main()
