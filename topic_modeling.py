import os
from xml.dom.pulldom import parseString
import numpy as np
import pandas as pd
import spacy, top2vec
from top2vec import Top2Vec
from configparser import ConfigParser
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk.util import ngrams

#______________________________________________________________________________________________

def main():
    config_object = ConfigParser()
    config_object.read('config.ini')
    input_config = config_object["INPUT_CONFIG"] 
    output_config = config_object["OUTPUT_CONFIG"]
    processing_config = config_object["PROCESSING_CONFIG"]

#LOAD_DATA_____________________________________________________________________________________
    if input_config['input_format'] == 'csv': #csv file
        df = pd.read_csv(input_config['input'])

    elif input_config['input_format'] == 'xlsx': #excel file
        df = pd.read_excel(input_config['input'])

    else: #folder with txt
        filenames = os.listdir(input_config['input'])
        texts = []
        for fn in filenames:
            with open(os.path.join(input_config['input'], fn)) as f:
                text = ' '.join(f.readlines())
            texts.append(text)
        df = pd.DataFrame(data={
            'filename': filenames,
            'text':texts
            })
    df = df[:200]

#PREPROCESSING_________________________________________________________________________________
    print('preprocessing data...')
    nlp = spacy.load("nl_core_news_sm")
    content_words = {'VERB', 'ADV', 'NOUN', 'PROPN', 'ADJ'}

    def preprocess(text):
        doc = nlp(text)
        if int(processing_config['lemmatize']) and int(processing_config['remove_stopwords']):
            text = ' '.join([t.lemma_ for t in doc if t.pos_ in content_words])
        elif int(processing_config['lemmatize']) and not int(processing_config['remove_stopwords']):
            text = ' '.join([t.lemma_ for t in doc])
        elif not int(processing_config['lemmatize']) and int(processing_config['remove_stopwords']):
            text = ' '.join([t.text for t in doc if t.pos_ in content_words])
        else:
            text = ' '.join([t.text for t in doc])
        return text

    df['text'] = df['text'].apply(lambda x: preprocess(x))
    
    def tokenizer(text, upper_n=int(processing_config['upper_ngram_range'])):
        result = []
        n = 1
        while n <= upper_n:
            for gram in ngrams(text.split(' '), n):
                result.append(' '.join(gram).strip())
            n += 1
        return result
    
#PREPARE_OUTPUT_DIR____________________________________________________________________________
    if not int(output_config['overwrite_output_dir']):
        assert os.path.exists(output_config['output_dir']) == False
    if not os.path.exists(output_config['output_dir']):
        os.mkdir(output_config['output_dir'])

#TRAIN_MODEL___________________________________________________________________________________
    default_models = {
        'doc2vec', 
        'universal-sentence-encoder', 
        'universal-sentence-encoder-large', 
        'universal-sentence-encoder-multilingual', 
        'universal-sentence-encoder-multilingual-large',
        'distiluse-base-multilingual-cased',
        'all-MiniLM-L6-v2',
        'paraphrase-multilingual-MiniLM-L12-v2'
        }

    base_model = processing_config['model'].strip()
    if not base_model:
        base_model = 'doc2vec'

    model = Top2Vec(
        df[input_config['text_column']].tolist(), 
        embedding_model=base_model,
        tokenizer=tokenizer 
        )

    #to do: enable any HuggingFace model

#HIERARCHICAL_TOPIC_REDUCTION__________________________________________________________________
    reduced = False
    max_n_topics = int(processing_config['topic_reduction']) 
    if max_n_topics and max_n_topics < model.get_num_topics():
        model.hierarchical_topic_reduction(num_topics=max_n_topics)
        reduced = True

#GET_KEYWORDS__________________________________________________________________________________
    keywords = []
    n_keywords = int(processing_config['n_keywords'])

    if not reduced:
        n_topics = model.get_num_topics()
        topic_idx = list(range(n_topics))

        for i in topic_idx:
            topic_keywords = model.topic_words[i].tolist()
            if len(topic_keywords) >= n_keywords:
                topic_keywords = topic_keywords[:n_keywords]
            topic_keywords = ', '.join(topic_keywords)
            keywords.append(topic_keywords)

    else:
        n_topics = model.get_num_topics_reduced()
        topic_idx = list(range(n_topics))

        for i in topic_idx:
            topic_keywords = model.topic_words_reduced[i].tolist()
            if len(topic_keywords) >= n_keywords:
                topic_keywords = topic_keywords[:n_keywords]
            topic_keywords = ', '.join(topic_keywords)
            keywords.append(topic_keywords)

    keyword_df = pd.DataFrame(data={'topic_id': topic_idx, 'keywords': keywords})
    keyword_df.to_csv(os.path.join(output_config['output_dir'], 'keywords_per_topic.csv'), index=False)

#RETURN_DOCUMENT_SCORES_PER_TOPIC______________________________________________________________    
    if not reduced:
        topics = model.doc_top.tolist()
        scores = model.doc_dist.tolist()
    
    else:
        topics = model.doc_top_reduced.tolist()
        scores = model.doc_dist_reduced.tolist()
    
    if processing_config['index_column'] != 'None':
        idx = df[processing_config['index_column']]
    elif 'filename' in df.columns:
        idx = df['filename']
    else:
        idx = list(range(0, len(df)))

    data = {'id': idx, 'topic': topics, 'score': scores}
    topic_df = pd.DataFrame(data=data)
    topic_df = topic_df.astype({"id": 'int64', "topic": 'int64', "score": float}, errors='raise')
    topic_df.sort_values(by='id', inplace=True) 
    topic_df.to_csv(os.path.join(output_config['output_dir'], 'topic_scores.csv'), index=False)
    print('done!')
#______________________________________________________________________________________________
if __name__ == "__main__":
    main()
