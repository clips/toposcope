#system
import os, zipfile
import random as rd

#preprocessing
import string
import pandas as pd
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import OrderedDict
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

#LDA, NMF
from sklearn.decomposition import LatentDirichletAllocation, NMF

#Top2Vec
from top2vec import Top2Vec

#BERTopic
#from bertopic import BERTopic
from umap import UMAP
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline

#Visualizations
import umap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from visualizations import *

rd.seed(42)
#_____________________________________________________________________________________________
def BERT_topic(df, base_model, text_column, dir_out, lang, upper_ngram_range, min_topic_size, topic_reduction, input_format, timestamps=None):

    """
    Training pipeline for BERTopic
    Arguments:
        df: pd.DataFrame with corpus,
        text_column: text column name (str),
        dir_out: output dir,
        lang: language,
        timestamps: if provided, visualize topics over time
    Returns:
        Topic document matrix
        Keywords per topic dataframe
        Topic keyword matrix
    """

    # Load embedding model
    if not base_model:
        if lang == 'english':
            base_model = 'all-MiniLM-L6-v2' # default model for English
        else:
            base_model = 'paraphrase-multilingual-MiniLM-L12-v2' # default model for all other languages
    print(f'\nComputing embeddings with SentenceTransformers using {base_model} as base model...')
    # precompute embeddings for visualizations
    sentence_model = SentenceTransformer(base_model)
    embeddings = sentence_model.encode(df[text_column].to_numpy(), show_progress_bar=True)

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
        n_gram_range=(1, int(upper_ngram_range)),
        language=lang,
        top_n_words=10,
        min_topic_size=int(min_topic_size),
        embedding_model=sentence_model, 
        nr_topics=int(topic_reduction),
        calculate_probabilities=True,
        umap_model=umap,
    )

    print('\nFitting BERTopic model...')
    _, probs = topic_model.fit_transform(df[text_column].to_numpy())
    topic_idx = topic_model.get_topic_info()['Topic']

    # fit the model
    _, probs = topic_model.fit_transform(
        df[text_column].to_numpy(), 
        embeddings=embeddings
        )
    
    topic_idx = topic_model.get_topic_info()['Topic']
    
    keywords = []

    for i in topic_idx: 
        topic_keywords = [x[0] for x in topic_model.get_topic(i)]
        keywords.append(topic_keywords)

    keyword_df = pd.DataFrame(data={
        'idx': topic_idx,
        'keywords': keywords,
    })
    
    if input_format == 'zip':
        idx = df['filename'].tolist()
    else:
        idx = list(df.index)

    topic_doc_matrix = pd.DataFrame(probs)
    topic_doc_matrix.insert(loc=0, column='idx', value=idx)

    #topic-term matrix
    vocab = topic_model.vectorizer_model.get_feature_names_out()
    topic_term_weights = topic_model.c_tf_idf_.toarray()
    topic_term_matrix = pd.DataFrame(topic_term_weights)
    topic_term_matrix.index = topic_idx
    topic_term_matrix.columns = vocab

    # Generate visualizations
    print("Generating visualizations...")
    documents_fig = generate_bertopic_visualizations(topic_model, dir_out, df[text_column].to_numpy(), embeddings, topic_reduction, timestamps)
    
    return topic_doc_matrix, keyword_df, topic_term_matrix, documents_fig

def coherence(topics, texts):

    """
    Compute coherence score for topic model.
    Arguments:
        topics: keywords per topic (list of lists)
        texts: tokenized texts (list of lists)
    Returns:
        coherence score (float)
    """
    print(topics)
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    coherence_model = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_score = coherence_model.get_coherence()

    return coherence_score

def LDA_model(df, text_column_name, dir_out, upper_ngram_range, n_topics, input_format, timestamps=None):

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
    print('Vectorizing texts...')
    texts = df[text_column_name].to_numpy()
    vectorizer = CountVectorizer(lowercase=False, min_df=5, ngram_range=(1, int(upper_ngram_range)))
    X = vectorizer.fit_transform(texts)

    #initialize and fit model
    lda = LatentDirichletAllocation(
    	n_components=int(n_topics),
        learning_method='online',
		random_state=42,
		max_iter=100,
		n_jobs=1
	)
    print("\nFitting LDA model...")
    lda.fit(X)

    #calculate probabilities per doc/topic
    scores = lda.transform(X)
    components_df = pd.DataFrame(lda.components_, columns=vectorizer.get_feature_names_out())

    #get keywords per topic
    keywords = []
    topic_idx = range(components_df.shape[0])

    for topic in topic_idx:
        tmp = components_df.iloc[topic]
        keywords.append(tmp.nlargest(10).index.tolist())

    #get text indices
    if input_format == 'zip':
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
    print("Generating visualizations...")
    keyword_barcharts = lda_visualize_barchart(lda, vectorizer, annotations)
    keyword_barcharts.write_html(os.path.join(dir_out, 'visualizations', 'keyword_barcharts.html'))

    # document topic plot
    documents_fig, topic_labels = nmf_lda_visualize_documents(lda, vectorizer, df[text_column_name].to_numpy(), X, annotations)
    documents_fig.write_html(os.path.join(dir_out, 'visualizations', 'document_topic_plot.html'))

    #compute topics over time
    if isinstance(timestamps, pd.Series):
        documents = pd.DataFrame(data={
            'Document': texts,
            'Timestamps': timestamps,
            'Topic': annotations,
        })
        topics_over_time = get_topics_over_time(documents, topic_labels)
        topics_over_time = pd.DataFrame(topics_over_time, columns=["Topic", "Words", "Frequency", "Timestamp"])
        time_fig = visualize_topics_over_time(annotations, topic_labels, topics_over_time)
        time_fig.write_html(os.path.join(dir_out, 'visualizations', 'topics_over_time.html'))

    return topic_doc_matrix, keyword_df, components_df, documents_fig

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

def NMF_model(df, text_column_name, dir_out, upper_ngram_range, n_topics, input_format, timestamps=None):

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
    vectorizer = TfidfVectorizer(lowercase=False, min_df=5, ngram_range=(1, upper_ngram_range))
    X = vectorizer.fit_transform(texts)

    nmf = NMF(
        n_components=int(n_topics), 
        init='random', 
        random_state=42,
    )

    print('\nFitting NMF model...')
    nmf.fit(X)

    scores = nmf.transform(X)
    components_df = pd.DataFrame(nmf.components_, columns=vectorizer.get_feature_names_out())

    #get keywords per topic
    keywords = []
    topic_idx = range(components_df.shape[0])
    for topic in topic_idx:
        tmp = components_df.iloc[topic]
        keywords.append(tmp.nlargest(10).index.tolist())

    #get text indices
    if input_format == 'zip':
        idx = df['filename'].tolist()
    else:
        idx = df.index.tolist()
    
    #create doc_topic df
    data = OrderedDict()
    data["idx"] = idx
    for t in topic_idx:
        data[str(t)] = [scores[i][t] for i in range(len(scores))] 
    topic_doc_matrix = pd.DataFrame(data=data)

    #extract annotations
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
    documents_fig, topic_labels = nmf_lda_visualize_documents(nmf, vectorizer, df[text_column_name].to_numpy(), X, annotations)
    documents_fig.write_html(os.path.join(dir_out, 'visualizations', 'document_topic_plot.html'))

    # compute topics over time
    if isinstance(timestamps, pd.Series):
        documents = pd.DataFrame(data={
            'Document': texts,
            'Timestamps': timestamps,
            'Topic': annotations,
        })
        topics_over_time = get_topics_over_time(documents, topic_labels)
        topics_over_time = pd.DataFrame(topics_over_time, columns=["Topic", "Words", "Frequency", "Timestamp"])
        time_fig = visualize_topics_over_time(annotations, topic_labels, topics_over_time)
        time_fig.write_html(os.path.join(dir_out, 'visualizations', 'topics_over_time.html'))

    return topic_doc_matrix, keyword_df, components_df, documents_fig

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

def compute_diversity(topics, topk=10):
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
        return puw


def tokenizer(text, upper_ngram_range):
        
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
        while n <= int(upper_ngram_range):
            for gram in ngrams(text.split(' '), n):
                result.append(' '.join(gram).strip())
            n += 1
        return result

def top_2_vec(df, text_column, base_model, dir_out, topic_reduction, input_format, upper_ngram_range, timestamps=None):

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
        'distiluse-base-multilingual-cased',
        'all-MiniLM-L6-v2',
        'paraphrase-multilingual-MiniLM-L12-v2'
    }

    #get base model
    if base_model.strip():
        embedding_model = base_model.strip()
        if embedding_model not in default_models:
            raise KeyError(
                """embedding_model must be one of: 'doc2vec', 'universal-sentence-encoder', 'universal-sentence-encoder-multilingual' 'distiluse-base-multilingual-cased', 'all-MiniLM-L6-v2', 'paraphrase-multilingual-MiniLM-L12-v2'"""
                )
    else:
        embedding_model = ''

    umap_args = {
        'n_neighbors': 15,
        'n_components': 5,
        'random_state': 42,
        'metric': "cosine",
    }

    print(f'\nFitting Top2Vec model...')
    print(f'Using {embedding_model} as embedding model...\n')
    model = Top2Vec(
        df[text_column].tolist(), 
        embedding_model=embedding_model,
        split_documents=False,
        min_count=50, #words occurring less frequently than 'min_count' are ignored
        tokenizer=lambda x: tokenizer(x, upper_ngram_range=upper_ngram_range),
        umap_args=umap_args,
    )

    #HIERARCHICAL_TOPIC_REDUCTION__________________________________________________________________
    reduced = False
    max_n_topics = int(topic_reduction)
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
    #get text indices
    if input_format == 'zip':
        idx = df['filename'].tolist()
    else:
        idx = df.index.tolist()

    topic_nums, topic_scores, _, __ = model.get_documents_topics(idx, reduced=reduced, num_topics=n_topics)
    
    if input_format == 'zip':
        idx = df['filename'].tolist()
    else:
        idx = list(df.index)
    
    #create doc_topic df
    topic_doc_matrix = pd.DataFrame()
    for i in range(len(topic_scores)):
        row = {topic: score for topic, score in zip(topic_nums[i], topic_scores[i])}
        topic_doc_matrix = topic_doc_matrix.append(row, ignore_index=True)
    topic_doc_matrix.insert(loc=0, column='idx', value=idx)
    
    annotations, _, _, _ = model.get_documents_topics(idx, reduced=reduced, num_topics=1)
    annotations = annotations.tolist()

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
    # visualizations
    #keywords
    bar_charts = top2vec_visualize_barchart(model, reduced, top_n_topics=len(topic_idx), n_words=10, width=400)
    bar_charts.write_html(os.path.join(dir_out, 'visualizations', 'keyword_barcharts.html'))

    #2D document plot
    docs = df[text_column].tolist()
    documents_fig, topic_labels = top2vec_visualize_documents(model, annotations, reduced, docs)
    documents_fig.write_html(os.path.join(dir_out, 'visualizations', 'document_topic_plot.html'))

    #hierarchy
    # hierarchy_fig = top2vec_visualize_hierarchy(model, annotations, reduced)
    # hierarchy_fig.write_html(os.path.join(dir_out, 'visualizations', 'hierarchy.html'))

    #topics over time
    if isinstance(timestamps, pd.Series):
        documents = pd.DataFrame(data={
            'Document': docs,
            'Timestamps': timestamps,
            'Topic': annotations,
        })
        topics_over_time = get_topics_over_time(documents, topic_labels)
        topics_over_time = pd.DataFrame(topics_over_time, columns=["Topic", "Words", "Frequency", "Timestamp"])
        time_fig = visualize_topics_over_time(annotations, topic_labels, topics_over_time)
        time_fig.write_html(os.path.join(dir_out, 'visualizations', 'topics_over_time.html'))
    
    return topic_doc_matrix, keyword_df, topic_term_matrix, documents_fig
