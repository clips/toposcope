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
    processing_config = config_object["PROCESSING_CONFIG"]

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
    if processing_config['algorithm'] == 'BERTopic':
        idx, topics, probs, keywords, keyword_df = BERT_topic(df, text_column)
    
    # elif processing_config['algorithm'] == 'LDA':
    #     idx, topics, probs, keywords, keyword_df = LDA_model(df, text_column)
    
    elif processing_config['algorithm'] == 'Top2Vec':
        idx, topics, probs, keywords, keyword_df = top_2_vec(df, text_column)

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
    data = {'id': idx, 'topic': topics, 'score': probs}
    topic_df = pd.DataFrame(data=data)
    topic_df = topic_df.astype({"id": 'int64', "topic": 'int64', "score": float}, errors='raise')
    topic_df.sort_values(by='id', inplace=True)
    topic_df.to_csv(os.path.join(output_config['output_dir'], 'topic_scores.csv'), index=False)

    #VISUALIZATION (to do)
    #   - wordclouds per topic
    #   - two-dimensional representation of documents/topic clusters

#______________________________________________________________________________________________
if __name__ == "__main__":
    main()
