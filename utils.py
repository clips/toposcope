#system
import os, zipfile
import random as rd
from configparser import ConfigParser

#preprocessing
import string
import pandas as pd
from nltk.util import ngrams
from nltk.corpus import stopwords
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
from umap import UMAP
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline

#Visualizations
import umap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from visualizations import *

rd.seed(42)

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
def BERT_topic(df, text_column, dir_out, lang):

    """
    Training pipeline for BERTopic
    Arguments:
        df: pd.DataFrame with corpus,
        text_column: text column name (str),
        dir_out: output dir,
        lang: language
    Returns:
        Topic document matrix
        Keywords per topic dataframe
        Topic keyword matrix
    """

    # Load embedding model
    print('    Computing embeddings with SentenceTransformers...')
    if not processing_config['model'].strip():
        if lang == 'english':
            base_model = 'all-MiniLM-L6-v2' # default model for English
        else:
            base_model = 'paraphrase-multilingual-MiniLM-L12-v2' # default model for all other languages
    else:
        base_model = processing_config['model'].strip() # custom model

    # precompute embeddings for visualizations
    sentence_model = SentenceTransformer(base_model)
    embeddings = sentence_model.encode(df[text_column].to_numpy(), show_progress_bar=True)

    #check if seed topics were provided
    if processing_config['seed_topic_list']:
        with open(processing_config['seed_topic_list']) as f:
            lines = f.readlines()
            seed_topic_list = [[w.strip() for w in l.split(',')] for l in lines]
    else:
        seed_topic_list = None

    # define umap model with default BERTopic values, 
    # but with random state in order to ensure reproducible results
    umap = UMAP(n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            low_memory=False,
            random_state=42)

    #instantiate topic model object
    topic_model = BERTopic(
        n_gram_range=(1, int(processing_config['upper_ngram_range'])),
        language=processing_config['lang'],
        top_n_words=10,
        min_topic_size=int(processing_config['min_topic_size']),
        seed_topic_list=seed_topic_list,
        embedding_model=sentence_model, 
        nr_topics=int(processing_config['topic_reduction']),
        calculate_probabilities=True,
        umap_model=umap,
    )

    print('    Fitting BERTopic model...')
    _, probs = topic_model.fit_transform(df[text_column].to_numpy())
    topic_idx = topic_model.get_topic_info()['Topic']

    # fit the model
    _, probs = topic_model.fit_transform(
        df[text_column].to_numpy(), 
        embeddings=embeddings
        )
    
    print('    Generating outputs...')
    topic_idx = topic_model.get_topic_info()['Topic']
    
    keywords = []

    for i in topic_idx: 
        topic_keywords = [x[0] for x in topic_model.get_topic(i)]
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

    # Generate visualizations
    generate_bertopic_visualizations(topic_model, dir_out, df[text_column].to_numpy(), embeddings)
    
    return topic_doc_matrix, keyword_df, topic_term_matrix

def coherence(topics, texts):

    """
    Compute coherence score for topic model.
    Arguments:
        topics: keywords per topic (list of lists)
        texts: tokenized texts (list of lists)
    Returns:
        coherence score (float)
    """

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    coherence_model = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_score = coherence_model.get_coherence()

    return coherence_score

def LDA_model(df, text_column_name, dir_out):

    """
    Training pipeline for LDA model.
    Arguments:
        df: pd.DataFrame with corpus 
        text_column_name: text column name in df
        dir_out: output directory
    Returns:
        Topic document matrix
        Keywords per topic dataframe
        Topic keyword matrix
    """

    #vectorize text
    texts = df[text_column_name]
    vectorizer = CountVectorizer(min_df=5, ngram_range=(1, int(processing_config['upper_ngram_range'])))
    X = vectorizer.fit_transform(texts)

    #initialize and fit model
    lda = LatentDirichletAllocation(
    	n_components=int(processing_config['n_components']),
        learning_method='online',
		random_state=42,
		max_iter=100,
		n_jobs=1
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
        keywords.append(tmp.nlargest(10).index.tolist())

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

    new_topic_doc_matrix = topic_doc_matrix.drop(columns=['idx'])  
    annotations = new_topic_doc_matrix.apply(lambda row: row.idxmax(), axis=1).tolist()

    #create keyword_df with keywords per topic
    keyword_df = pd.DataFrame(data={
        'idx': topic_idx,
        'keywords': keywords,
    }) 

    # Generate visualizations
    print("Generate visualizations...")
    keyword_barcharts = lda_visualize_barchart(lda, vectorizer, annotations)
    keyword_barcharts.write_html(os.path.join(dir_out, 'visualizations', 'keyword_barcharts.html'))

    # document topic plot
    document_topic_fig = nmf_visualize_documents(lda, vectorizer, X, annotations, texts)
    document_topic_fig.write_html(os.path.join(dir_out, 'visualizations', 'document_topic_plot.html'))

    return topic_doc_matrix, keyword_df, components_df

def plot_document_topics_umap(model, texts, label_names, output_dir):
    """
    Generate a 2D plot of documents representing their topics using UMAP.
    Arguments:
        model: trained LDA model
        label_names: list of label names corresponding to the topics
        output_dir: output directory
    Returns:
        None
    """
    # Get topic proportions for each document
    document_topics = model.transform(texts)  # Replace with your document data

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

    """Load data. 
    Arguments:
        in_dir: path to corpus
        input_format: 'csv' or 'zip'
        delimiter: delimiter for csv, if applicable
    Returns:
        pd.DataFrame() with corpus
    """

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

def NMF_model(df, text_column_name, dir_out):

    """
    Training pipeline for NMF model.
    Arguments:
        texts: pd.Series (column of corpus DF)
        dir_out: output directory
    Returns:
        Topic document matrix
        Keywords per topic dataframe
        Topic keyword matrix
    """

    texts = df[text_column_name].to_numpy()
    vectorizer = CountVectorizer(min_df=5, ngram_range=(1, int(processing_config['upper_ngram_range'])))
    X = vectorizer.fit_transform(texts)

    nmf = NMF(
        n_components=int(processing_config['n_components']), 
        init='random', 
        random_state=42
    )

    nmf.fit(X)

    scores = nmf.transform(X)
    components_df = pd.DataFrame(nmf.components_, columns=vectorizer.get_feature_names())

    print("Generating output...")
    #get keywords per topic
    keywords = []
    topic_idx = range(components_df.shape[0])
    for topic in topic_idx:
        tmp = components_df.iloc[topic]
        keywords.append(tmp.nlargest(10).index.tolist())

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
    new_topic_doc_matrix = topic_doc_matrix.drop(columns=['idx'])  
    annotations = new_topic_doc_matrix.apply(lambda row: row.idxmax(), axis=1).tolist()

    #create df with keywords per topic
    keyword_df = pd.DataFrame(data={
        'idx': topic_idx,
        'keywords': keywords,
    })

    # Generate visualizations
    print("Generating visualizations...")
    # keywords
    keywords_fig = nmf_visualize_barchart(nmf, vectorizer, annotations)
    keywords_fig.write_html(os.path.join(dir_out, 'visualizations', 'keyword_barcharts.html'))

    # document topic plot
    document_topic_fig = nmf_visualize_documents(nmf, vectorizer, X, annotations, texts)
    document_topic_fig.write_html(os.path.join(dir_out, 'visualizations', 'document_topic_plot.html'))

    return topic_doc_matrix, keyword_df, components_df

def preprocess(text, nlp, lang, tokenize, lemmatize, remove_nltk_stopwords, remove_custom_stopwords, remove_punct, lowercase):

    """Preprocess input text.
    Arguments:
        text: Str,
        nlp: spacy model,
        lang: language ('dutch', 'english', 'french', 'german'),
        tokenize: bool (True = tokenize)
        lemmatize: bool (True = lemmatize),
        remove_nltk_stopwords: bool (True = remove stopwords),
        remove_custom_stopwords: bool (True = remove custom stopwords),
        remove_punct: bool (True = remove punctuation),
        lowercase: bool (True = lowercase),
    Returns:
        Preprocessed Str
    """

    #tokenize/lemmatize
    disable = ['ner']
    if not lemmatize or not tokenize:
        disable.extend(['parser', 'tagger'])

    if tokenize and lemmatize:
        doc = nlp(text, disable=disable)
        text = ' '.join([t.lemma_ for t in doc])
    elif tokenize and not lemmatize:
        doc = nlp(text, disable=disable)
        text = ' '.join([t.text for t in doc])
    elif not tokenize and lemmatize:
        raise ValueError("'tokenize' cannot be False if 'lemmatize' is True. Please change your configuration file.")
    else: # no preprocessing with spacy
        pass

    #lowercase
    if lowercase:
        text = text.lower()

    #remove NLTK stopwords
    if remove_nltk_stopwords:
        stop_words = stopwords.words(lang)
        text = ' '.join([t for t in text.split() if t.lower() not in stop_words])

    #remove custom stop words
    #to do: allow phrases and regex
    if remove_custom_stopwords:
        with open(remove_custom_stopwords) as x:
            lines = x.readlines()
            custom_stopwords = set([l.strip() for l in lines])
            text = ' '.join([t for t in text.split() if t.lower() not in custom_stopwords])

    # remove punctuation
    if remove_punct: 
        punct = string.punctuation
        punct += '‘’“”′″‴'
        for p in punct:
            text = text.replace(p, '')
        text = ' '.join(text.split())

    return text

def calculate_proportion_of_unique_words(topic, topk):
    unique_words = set(topic[:topk])
    return len(unique_words) / topk

def compute_diversity(topics, topk=10):
    """
    Compute the proportion of unique words. Used as diversity score for evaluation. 
    Arguments:
        topics: a list of lists of words
        topk: top k words on which the topic diversity will be computed
    Returns:
        Proportion of unique words (float)
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        diversity_scores = []
    
        for topic in topics:
            puw = calculate_proportion_of_unique_words(topic, topk)
            diversity_scores.append(puw)
        
        # Calculate the average diversity score
        average_diversity_score = sum(diversity_scores) / len(diversity_scores)
        
        return average_diversity_score


def tokenizer(text, upper_n=int(processing_config['upper_ngram_range'])):
        
        """
        Tokenizer function to use later in case ngrams are requested.
        Arguments:
            text: Str
            upper_n: upper ngram range, lower is always set to 1
        Returns:
            tokenized text (list of strings)
        """

        result = []
        n = 1
        while n <= int(upper_n):
            for gram in ngrams(text.split(' '), n):
                result.append(' '.join(gram).strip())
            n += 1
        return result

def top_2_vec(df, text_column, dir_out):

    """
    Training pipeline for Top2Vec. Also creates visualizations.
    Arguments:
        df: pd.DataFrame with corpus
        text_column: text column name (str)
        dir_out: output dir
    Returns:
        Topic document matrix
        Keywords per topic dataframe
        Topic keyword matrix
    """

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
    if processing_config['model'].strip():
        embedding_model = processing_config['model']
        if embedding_model not in default_models:
            raise KeyError(
                """embedding_model must be one of: 'doc2vec', 'universal-sentence-encoder', 'universal-sentence-encoder-large', 'universal-sentence-encoder-multilingual', 'universal-sentence-encoder-multilingual-large', 'distiluse-base-multilingual-cased', 'all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2'"""
                )
    else:
        embedding_model = ''

    model = Top2Vec(
        df[text_column].tolist(), 
        embedding_model=embedding_model,
        split_documents=False,
        min_count=50, #words occurring less frequently than 'min_count' are ignored
        tokenizer=tokenizer,
    )

    #HIERARCHICAL_TOPIC_REDUCTION__________________________________________________________________
    reduced = False
    max_n_topics = int(processing_config['topic_reduction']) 
    if max_n_topics and max_n_topics < model.get_num_topics():
        model.hierarchical_topic_reduction(num_topics=max_n_topics)
        reduced = True

    #GET_KEYWORDS__________________________________________________________________________________
    keywords = []
    n_keywords = 10

    n_topics = model.get_num_topics(reduced=reduced)
    topic_idx = list(range(n_topics))

    for i in topic_idx:
        if reduced:
            topic_keywords = model.topic_words_reduced[i].tolist()
        else:
            topic_keywords = model.topic_words[i].tolist()

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

    new_topic_doc_matrix = topic_doc_matrix.drop(columns=['idx'])  
    annotations = new_topic_doc_matrix.apply(lambda row: row.idxmax(), axis=1).tolist()

    # Create topic term matrix
    if not reduced:
        topic_term_matrix = pd.DataFrame(0, index=topic_idx, columns=model.topic_words[0])
        for i, topic in enumerate(topic_idx):
            words = model.topic_words[i]
            scores = model.topic_word_scores[i]
            topic_term_matrix.loc[topic, words] = scores
    else:
        topic_term_matrix = pd.DataFrame(0, index=topic_idx, columns=model.topic_words_reduced[0])
        for i, topic in enumerate(topic_idx):
            words = model.topic_words_reduced[i]
            scores = model.topic_word_scores_reduced[i]
            topic_term_matrix.loc[topic, words] = scores       
    topic_term_matrix = topic_term_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

    print("Generating visualizations...")
    print('    Keyword barcharts...')
    # visualizations
    #keywords
    bar_charts = top2vec_visualize_barchart(model, reduced, top_n_topics=len(topic_idx), n_words=10, width=400)
    bar_charts.write_html(os.path.join(dir_out, 'visualizations', 'keyword_barcharts.html'))

    #2D document plot
    print('    Document plot...')
    docs = df[text_column].tolist()
    documents_fig = top2vec_visualize_documents(model, annotations, reduced, docs)
    documents_fig.write_html(os.path.join(dir_out, 'visualizations', 'document_topic_plot.html'))
    
    return topic_doc_matrix, keyword_df, topic_term_matrix
