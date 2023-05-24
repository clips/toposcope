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
    df = load_data(in_dir, input_format)
    text_column = input_config['text_column']
            
#PREPROCESSING_________________________________________________________________________________
    lemmatize = int(processing_config['lemmatize'])
    remove_stopwords = int(processing_config['remove_stopwords'])
    remove_punct = int(processing_config['remove_punct'])
    lowercase = int(processing_config['lowercase'])

    df[text_column] = df[text_column].apply(lambda x: preprocess(x, lemmatize, remove_stopwords, remove_punct, lowercase))
    
#PREPARE_OUTPUT_DIR____________________________________________________________________________
    if not int(output_config['overwrite_output_dir']):
        assert os.path.exists(output_config['output_dir']) == False

    if not os.path.exists(output_config['output_dir']):
        os.mkdir(output_config['output_dir'])

#FIT_MODEL_____________________________________________________________________________________
    if input_config['algorithm'] == 'BERTopic':
        topic_doc_matrix, keyword_df = BERT_topic(df, text_column)
    
    elif input_config['algorithm'] == 'LDA':
        topic_doc_matrix, keyword_df = LDA_model(df[text_column])
    
    elif input_config['algorithm'] == 'NMF':
        topic_doc_matrix, keyword_df = NMF_model(df[text_column])
    
    elif input_config['algorithm'] == 'Top2Vec':
        topic_doc_matrix, keyword_df = top_2_vec(df, text_column)

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
    eval_df.to_csv(os.path.join(output_config['output_dir'], 'evaluation.csv'), index=False)
    
    #KEYWORDS PER TOPIC
    keyword_df.to_csv(os.path.join(output_config['output_dir'], 'keywords_per_topic.csv'), index=False)

    #ANNOTATIONS
    topic_doc_matrix.to_csv(os.path.join(output_config['output_dir'], 'topic_doc_matrix.csv'), index=False)

    #VISUALIZATION (to do)
    #   - wordclouds per topic
    #   - two-dimensional representation of documents/topic clusters

#______________________________________________________________________________________________
if __name__ == "__main__":
    main()
