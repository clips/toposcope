import os
from xml.dom.pulldom import parseString
import numpy as np
import pandas as pd
import spacy
from top2vec import Top2Vec
from configparser import ConfigParser
from tqdm import tqdm

#______________________________________________________________________________________________

def main():
    config_object = ConfigParser()
    config_object.read('config.ini')
    input_config = config_object["INPUT_CONFIG"] 
    output_config = config_object["OUTPUT_CONFIG"]
    processing_config = config_object["PROCESSING_CONFIG"]

#LOAD_DATA_____________________________________________________________________________________
    print('loading data...')
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
    df = df[:1000]

# #PREPROCESSING_________________________________________________________________________________
#     print('preprocessing data...')
#     nlp = spacy.load("nl_core_news_sm")

#     content_words = {'VERB', 'ADV', 'NOUN', 'PROPN', 'ADJ'}
#     def preprocess(text):
#         if not processing_config['lemmatize'] and not processing_config['remove_stopwords']:
#             pass
#         else:
#             doc = nlp(text)
#             if processing_config['lemmatize'] and processing_config['remove_stopwords']:
#                 text = ' '.join([t.lemma_ for t in doc if t.pos_ in content_words])
#             elif processing_config['lemmatize'] and not processing_config['remove_stopwords']:
#                 text = ' '.join([t.lemma_ for t in doc])
#             else:
#                 text = ' '.join([t for t in doc if t.pos_ in content_words])
#         return text

#     df['text'] = df['text'].apply(lambda x: preprocess(x))
    
#PREPARE_OUTPUT_DIR____________________________________________________________________________
    if not output_config['overwrite_output_dir']:
        assert os.path.exists(output_config['output_dir']) == False
    if not os.path.exists(output_config['output_dir']):
        os.mkdir(output_config['output_dir'])

#TRAIN_MODEL___________________________________________________________________________________
    print('training model...')
    model = Top2Vec(df[input_config['text_column']].tolist())
    topic_sizes, topic_nums = model.get_topic_sizes()

#GET_KEYWORDS__________________________________________________________________________________
    print('get keywords...')
    n_topics = model.get_num_topics()
    topic_idx = list(range(n_topics))

    keywords = []
    n_keywords = int(processing_config['n_keywords'])
    for i in topic_idx:
        topic_keywords = model.topic_words[i].tolist()
        if len(topic_keywords) >= n_keywords:
            topic_keywords = topic_keywords[:n_keywords]
        topic_keywords = ', '.join(topic_keywords)
        keywords.append(topic_keywords)

    keyword_df = pd.DataFrame(data={'topic_id': topic_idx, 'keywords': keywords})
    keyword_df.to_csv(os.path.join(output_config['output_dir'], 'keywords_per_topic.csv'), index=False)

#RETURN_DOCUMENT_SCORES_PER_TOPIC______________________________________________________________
    print('retrieve topic scores per doc...')
    topic_df = pd.DataFrame()
    topic_scores = dict()
    for size, num in tqdm(zip(topic_sizes, topic_nums)):
        _, document_scores, document_ids = model.search_documents_by_topic(topic_num=num, num_docs=size)
        for i, score in zip(document_ids, document_scores):
            if i not in topic_scores.keys():
                doc_dict = {num: score}
                topic_scores[i] = doc_dict
            else:
                doc_dict = topic_scores[i]
                doc_dict[num] = score
                topic_scores[i] = doc_dict
    #print(topic_scores)
    for doc_i, doc_dict in topic_scores.items():
        row = {'id': doc_i}
        for topic, score in doc_dict.items():
            row['topic'] = topic
            row['score'] = score
        topic_df = topic_df.append(row, ignore_index=True)
    topic_df = topic_df.astype({"id": 'int64', "topic": 'int64', "score": float}, errors='raise') 
    topic_df.to_csv(os.path.join(output_config['output_dir'], 'topic_scores.csv'), index=False)
    print('done!')
#______________________________________________________________________________________________
if __name__ == "__main__":
    main()
