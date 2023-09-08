#system
import os, zipfile
from configparser import ConfigParser

#preprocessing
import spacy, regex, string
import pandas as pd
import numpy as np
import hdbscan

from nltk.util import ngrams
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

#LDA
from sklearn.decomposition import LatentDirichletAllocation, NMF

#Top2Vec
from top2vec import Top2Vec

#BERTopic
from bertopic import BERTopic
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer

#Visualizations
import plotly.graph_objects as go
import umap
import matplotlib.pyplot as plt

#load SpaCy and config_______________________________________________________________________
config_object = ConfigParser()
config_object.read('config.ini')
input_config = config_object["INPUT_CONFIG"] 
output_config = config_object["OUTPUT_CONFIG"]

if input_config['algorithm'] =='LDA':
    processing_config = config_object["LDA_CONFIG"]
elif input_config['algorithm'] =='NMF':
    processing_config = config_object["NMF_CONFIG"]
elif input_config['algorithm'] =='BERTopic':
    processing_config = config_object["BERTOPIC_CONFIG"]
elif input_config['algorithm'] =='Top2Vec':
    processing_config = config_object["TOP2VEC_CONFIG"]
else:
    raise KeyError("Please specify one of the following algorithms: 'BERTopic', 'Top2Vec', 'NMF', 'LDA'.")

#_____________________________________________________________________________________________
def BERT_topic(df, text_column, dir_out):

    """Training pipeline for BERTopic"""

    #get base model
    if processing_config['model']:
        embedding_model = pipeline("feature-extraction", processing_config['model'])
    else:
        embedding_model = None

    #check if seed topics were provided
    if processing_config['seed_topic_list']:
        with open(processing_config['seed_topic_list']) as f:
            lines = f.readlines()
            seed_topic_list = [[w.strip() for w in l.split(',')] for l in lines]
    else:
        seed_topic_list = None

    #instantiate topic model object
    if embedding_model:
        topic_model = BERTopic(
            n_gram_range=(1, int(processing_config['upper_ngram_range'])),
            language=processing_config['lang'],
            top_n_words=int(processing_config['n_keywords']),
            min_topic_size=int(processing_config['min_topic_size']),
            seed_topic_list=seed_topic_list,
            embedding_model=embedding_model, 
            nr_topics=int(processing_config['topic_reduction']),
            calculate_probabilities=True,
        )
    
    else:
        topic_model = BERTopic(
            n_gram_range=(1, int(processing_config['upper_ngram_range'])),
            language=processing_config['lang'],
            top_n_words=int(processing_config['n_keywords']),
            min_topic_size=int(processing_config['min_topic_size']),
            seed_topic_list=seed_topic_list,
            nr_topics=int(processing_config['topic_reduction']),
            calculate_probabilities=True,
        )

    topics, probs = topic_model.fit_transform(df[text_column].to_numpy())

    topic_idx = topic_model.get_topic_info()['Topic']
    
    keywords = []

    for i in topic_idx: 
        topic_keywords = [x[0] for x in topic_model.get_topic(i)]
        topic_keywords = ', '.join(topic_keywords)
        keywords.append(topic_keywords)

    keyword_df = pd.DataFrame(data={
        'idx': topic_idx,
        'keywords': keywords,
    })
    
    if input_config['input_format'] == 'zip':
        idx = df['filename'].tolist()
    else:
        idx = list(df.index)

    topic_doc_matrix = pd.DataFrame(probs)
    topic_doc_matrix.insert(loc=0, column='idx', value=idx)

    #topic-term matrix
    vocab = topic_model.vectorizer_model.get_feature_names()
    topic_term_weights = topic_model.c_tf_idf_.toarray()
    topic_term_matrix = pd.DataFrame(topic_term_weights)
    topic_term_matrix.index = topic_idx
    topic_term_matrix.columns = vocab
    
    return topic_doc_matrix, keyword_df, topic_term_matrix

def coherence(topics, texts):

    dictionary = Dictionary(texts)
    corpus = [[dictionary.doc2bow(text)] for text in texts]

    cm = CoherenceModel(topics=topics, 
                        texts=texts, 
                        corpus=corpus, 
                        dictionary=dictionary, 
                        coherence='c_v')
    
    coherence_score = round(cm.get_coherence(), 3)

    return coherence_score

def LDA_model(texts, dir_out):

    #vectorize text
    vectorizer = CountVectorizer(ngram_range=(1, int(processing_config['upper_ngram_range'])))
    X = vectorizer.fit_transform(texts)

    #initialize and fit model
    lda = LatentDirichletAllocation(
    	n_components=int(processing_config['n_components']),
        learning_method='online',
		random_state=42,
		max_iter=100,
		n_jobs=2
	)
    lda.fit(X)

    #calculate probabilities per doc/topic
    scores = lda.transform(X)
    components_df = pd.DataFrame(lda.components_, columns=vectorizer.get_feature_names())

    #get keywords per topic
    keywords = []
    topic_idx = range(components_df.shape[0])

    for topic in topic_idx:
        tmp = components_df.iloc[topic]
        keywords.append(tmp.nlargest(int(processing_config['n_keywords'])).index.tolist())

    #get text indices
    if input_config['input_format'] == 'zip':
        idx = df['filename'].tolist()
    else:
        idx = list(df.index)

    #create doc_topic df
    data = OrderedDict()
    data["idx"] = idx
    for t in topic_idx:
        data[str(t)] = [scores[i][t] for i in range(len(scores))] 
    topic_doc_matrix = pd.DataFrame(data=data)

    #create keyword_df with keywords per topic
    keyword_df = pd.DataFrame(data={
        'idx': topic_idx,
        'keywords': keywords,
    }) 

    # Generate two-dimensional UMAP visualization of document topics
    # Reduce dimensionality using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(scores)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(topic_idx):
        plt.scatter(umap_embeddings[:, 0][scores.argmax(axis=1) == i],
                    umap_embeddings[:, 1][scores.argmax(axis=1) == i],
                    label=label)
        
    plt.title('UMAP Visualization of Documents per Topic with LDA')
    plt.legend()
    plt.savefig(os.path.join(dir_out, 'UMAP_visualization.png'))
    plt.show()

    return topic_doc_matrix, keyword_df, components_df

def generate_bar_charts(df, dir_out):

    # Iterate over each topic in the dataframe
    for topic in df.index:
        topic_data = df.loc[topic].sort_values(ascending=False)[:20]
        
        # Create a horizontal bar chart using Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=topic_data.values,
            y=topic_data.index,
            orientation='h',
        ))
        
        # Set the title and axis labels
        fig.update_layout(
            title=f'Topic {topic}',
            xaxis_title='Weight',
            yaxis_title='Term',
            yaxis=dict(autorange="reversed"),
        )
        
        # Display the bar chart
        fig.write_image(dir_out + '/topic_term_weights/' + f'topic_{topic}_term_weights.png')

def plot_document_topics_umap(model, label_names, output_dir):
    """
    Generate a 2D plot of documents representing their topics using UMAP.

    Parameters:
    model (sklearn.decomposition.LatentDirichletAllocation): Trained LDA model.
    label_names (list): List of label names corresponding to the topics.
    output_dir (str): Directory where the plot should be saved.

    Returns:
    None
    """
    # Get topic proportions for each document
    document_topics = model.transform(YOUR_DOCUMENTS)  # Replace with your document data

    # Reduce dimensionality using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(document_topics)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(label_names):
        plt.scatter(umap_embeddings[:, 0][document_topics.argmax(axis=1) == i],
                    umap_embeddings[:, 1][document_topics.argmax(axis=1) == i],
                    label=label)
    plt.title('Document Topics UMAP')
    plt.legend()
    plt.savefig(output_dir)
    plt.show()

def load_data(in_dir, input_format, delimiter):

    """Load data. Valid formats are .csv file or directory of .txt files"""

    if input_format == 'csv': # csv file
        df = pd.read_csv(in_dir, delimiter=delimiter)

    elif input_format == 'zip': # zip folder with txt
        df = pd.DataFrame(columns=['filename', 'text'])

        with zipfile.ZipFile(in_dir, 'r') as zip_file:
            for file_info in zip_file.infolist():
                if file_info.filename.endswith('.txt'):
                    filename = os.path.basename(file_info.filename)
                    with zip_file.open(file_info) as txt_file:
                        text = txt_file.read().decode('utf-8')  # Assuming UTF-8 encoding
                    df = df.append({'filename': filename, 'text': text}, ignore_index=True)
        df = df.sort_values('filename')
    
    else:
        raise ValueError('Please specify a valid input format: "zip" or "csv".')
    
    return df

def NMF_model(texts, dir_out):

    vectorizer = CountVectorizer(ngram_range=(1, int(processing_config['upper_ngram_range'])))
    X = vectorizer.fit_transform(texts)

    nmf = NMF(
        n_components=int(processing_config['n_components']), 
        init='random', 
        random_state=42
    )

    nmf.fit(X)

    scores = nmf.transform(X)
    components_df = pd.DataFrame(nmf.components_, columns=vectorizer.get_feature_names())

    #get keywords per topic
    keywords = []
    topic_idx = range(components_df.shape[0])
    for topic in topic_idx:
        tmp = components_df.iloc[topic]
        keywords.append(tmp.nlargest(int(processing_config['n_keywords'])).index.tolist())

    #get text indices
    if input_config['input_format'] == 'zip':
        idx = df['filename'].tolist()
    else:
        idx = list(df.index)
    
    #create doc_topic df
    data = OrderedDict()
    data["idx"] = idx
    for t in topic_idx:
        data[str(t)] = [scores[i][t] for i in range(len(scores))] 
    topic_doc_matrix = pd.DataFrame(data=data)

    #create df with keywords per topic
    keyword_df = pd.DataFrame(data={
        'idx': topic_idx,
        'keywords': keywords,
    })

    # Generate two-dimensional UMAP visualization of document topics
    # Reduce dimensionality using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(scores)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(topic_idx):
        plt.scatter(umap_embeddings[:, 0][scores.argmax(axis=1) == i],
                    umap_embeddings[:, 1][scores.argmax(axis=1) == i],
                    label=label)
        
    plt.title('UMAP Visualization of Documents per Topic with NMF')
    plt.legend()
    plt.savefig(os.path.join(dir_out, 'UMAP_visualization.png'))
    plt.show()

    return topic_doc_matrix, keyword_df, components_df

def preprocess(text, lemmatize, remove_stopwords, remove_custom_stopwords, remove_punct, lowercase):

    """Create Spacy doc from input. 
    Lemmatize and/or remove stopwords (incl. punctuation) if requested."""

    lang = processing_config['lang']

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
        raise ValueError(f"'{lang}' is not a valid language, please use one of the following languages: 'dutch', 'english', 'french', 'german'.")

    #create doc object
    doc = nlp(text)

    #lemmatize
    if lemmatize:
        text = ' '.join([t.lemma_ for t in doc])
    else:
        text = ' '.join([t.text for t in doc])
    
    #lowercase
    if lowercase:
        text = text.lower()
    
    #remove stopwords
    if remove_stopwords:
        stop_words = stopwords.words(lang)
        text = ' '.join([t for t in text.split() if t not in stop_words])

    #custom stop word list
    custom_stopword_dir = processing_config['custom_stopword_list']
    if custom_stopword_dir.strip() and remove_custom_stopwords:
        with open(custom_stopword_dir) as x:
            lines = x.readlines()
            custom_stopwords = [l.strip() for l in lines]
            text = ' '.join([t for t in text.split() if t not in custom_stopwords])

    # remove punctuation
    if remove_punct: 
        punct = string.punctuation
        punct += '‘’“”′″‴'
        for p in punct:
            text = text.replace(p, '')
        text = ' '.join(text.split())

    return text

def proportion_unique_words(topics, topk=10):
    """
    compute the proportion of unique words
    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity will be computed
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = len(unique_words) / (topk * len(topics))
        return round(puw, 3)


def tokenizer(text, upper_n=int(processing_config['upper_ngram_range'])):
        
        """Tokenizer function to use later in case ngrams are requested."""

        result = []
        n = 1
        while n <= int(upper_n):
            for gram in ngrams(text.split(' '), n):
                result.append(' '.join(gram).strip())
            n += 1
        return result

def top_2_vec(df, text_column, dir_out):

    "Top2Vec training pipeline"

    embedding_model = processing_config['model']

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

    #get base model
    if processing_config['model']:
        embedding_model = processing_config['model']
        if embedding_model not in default_models:
            raise KeyError(
                """embedding_model must be one of: 'doc2vec', 'universal-sentence-encoder', 'universal-sentence-encoder-large', 'universal-sentence-encoder-multilingual', 'universal-sentence-encoder-multilingual-large', 'distiluse-base-multilingual-cased', 'all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2'"""
                )

    model = Top2Vec(
        df[text_column].tolist(), 
        embedding_model=embedding_model,
        split_documents=False,
        min_count=50, #words occurring less frequently than 'min_count' are ignored
        tokenizer=tokenizer 
    )

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
        n_topics = len([t for t in model.topic_words_reduced])
        topic_idx = list(range(n_topics))

        for i in topic_idx:
            topic_keywords = model.topic_words_reduced[i].tolist()
            if len(topic_keywords) >= n_keywords:
                topic_keywords = topic_keywords[:n_keywords]
            topic_keywords = ', '.join(topic_keywords)
            keywords.append(topic_keywords)

    keyword_df = pd.DataFrame(data={'topic_id': topic_idx, 'keywords': keywords})

    #RETURN_DOCUMENT_SCORES_PER_TOPIC______________________________________________________________    
    topic_nums, topic_scores, _, __ = model.get_documents_topics(list(range(len(df))), reduced=reduced, num_topics=n_topics)
    
    if input_config['input_format'] == 'zip':
        idx = df['filename'].tolist()
    else:
        idx = list(df.index)
    
    #create doc_topic df
    topic_doc_matrix = pd.DataFrame()
    for i in range(len(topic_scores)):
        row = {topic: score for topic, score in zip(topic_nums[i], topic_scores[i])}
        topic_doc_matrix = topic_doc_matrix.append(row, ignore_index=True)
    topic_doc_matrix.insert(loc=0, column='idx', value=idx)

    # create topic_term_matrix
    topic_term_matrix = pd.DataFrame()
    topic_term_matrix.index = topic_idx
    for i, topic in enumerate(topic_idx):
        for word, score in zip(model.topic_words[i], model.topic_word_scores[i]):
            topic_term_matrix.at[topic, word] = score
    topic_term_matrix = topic_term_matrix.fillna(0)
    
    return topic_doc_matrix, keyword_df, topic_term_matrix
