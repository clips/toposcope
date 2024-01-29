#system
import itertools, os
import numpy as np

#preprocessing
import itertools
import pandas as pd
from collections import Counter
from scipy.spatial.distance import squareform
from sklearn.preprocessing import normalize

#Visualizations
from typing import List, Union, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.cluster import hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from scipy.sparse import csr_matrix

"""
This file contains adaptations of the source code of BERTopic visualizations:
https://github.com/MaartenGr/BERTopic/tree/62e97ddea6cdcf9e4da25f9eaed478b22a9f9e20/bertopic/plotting
The functions are adapted to be compatible with other topic models.
"""

#BERTOPIC________________________________________________________________________________________________________________________________
def generate_bertopic_visualizations(model, dir_out, docs, embeddings, topic_reduction, timestamps=None):
    """
    Generate visualizations for BERTopic
    Arguments:
        model: Fitted BERTopic model,
        dir_out: output directory (str),
        docs: pandas Series containing corpus
        timestamps: None by default
    Return:
        None
    """

    # topic hierarchy
    # if topic_reduction:
    #     hierarchical_topics = model.hierarchical_topics(docs)
    #     hierarchy_fig = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    # else:
    #     hierarchy_fig = model.visualize_hierarchy()
    # hierarchy_fig.write_html(os.path.join(dir_out, 'visualizations', 'hierarchy.html'))

    # most important keywords per topic
    keyword_fig = model.visualize_barchart(topics=model.get_topics(), width=400)
    keyword_fig.write_html(os.path.join(dir_out, 'visualizations', 'keyword_barcharts.html'))

    # documents and topics
    reduced_embeddings = UMAP(metric='cosine', random_state=42).fit_transform(embeddings)
    if topic_reduction:
        hierarchical_topics = model.hierarchical_topics(docs)
        document_fig = model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
    else:
        document_fig = model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
    document_fig.write_html(os.path.join(dir_out, 'visualizations', 'document_topic_plot.html'))

    # topics over time
    if timestamps:
        topics_over_time = model.topics_over_time(docs, timestamps, evolution_tuning=False, global_tuning=False)
        time_fig = model.visualize_topics_over_time(topics_over_time)
        time_fig.write_html(os.path.join(dir_out, 'visualizations', 'topics_over_time.html'))
    
    return document_fig

#TOP2VEC_________________________________________________________________________________________________________________________________
def get_topics_over_time(documents, topic_names):

    """
    Compute topics over time.
    Arguments
        documents: pd.DataFrame(
            Document: str
            Timestamps
            Topic
        ),
        topic names: dict: {topic_id: topic_id_name}
    Returns:
        Topics over time (list)
    """
    
    # For each unique timestamp, create topic representations
    topics_over_time = []
    timestamps = documents.Timestamps.tolist()
    topic_names = {str(k): v for k,v in topic_names.items()}

    for timestamp in timestamps:

        selection = documents.loc[documents.Timestamps == timestamp, :]
        documents_per_topic = selection.groupby(['Topic'], as_index=False).agg({'Document': ' '.join,
                                                                                "Timestamps": "count"})
        # Extract the words per topic
        topic_frequency = pd.Series(documents_per_topic.Timestamps.values, index=documents_per_topic.Topic).to_dict()
        topic_frequency = {int(k): v for k,v in topic_frequency.items()}

        # Fill dataframe with results
        topics_at_timestamp = [(int(topic), ", ".join(topic_names[str(topic)].split('_')[1:]), topic_frequency[int(topic)], timestamp) for topic in topic_frequency.keys()]
        topics_over_time.extend(topics_at_timestamp)
    return topics_over_time

def visualize_topics_over_time(annotations,
                               topic_labels,
                               topics_over_time: pd.DataFrame,
                               normalize_frequency: bool = False,
                               title: str = "<b>Topics over Time</b>",
                               width: int = 1250,
                               height: int = 450) -> go.Figure:
    """
    Visualize topics over time
    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_over_time: The topics you would like to be visualized with the
                          corresponding topic representation
        normalize_frequency: Whether to normalize each topic's frequency individually
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Returns:
        A plotly.graph_objects.Figure including all traces
    """
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]

    # Select topics based on top_n and topics args
    frequencies = Counter(annotations)
    freq_df = pd.DataFrame({'Topic': list(frequencies.keys()), 'Count': list(frequencies.values())})
    freq_df = freq_df.sort_values("Count", ascending=False)
    freq_df = freq_df.loc[freq_df.Topic != -1, :]

    # Prepare data
    topic_names = {int(key): value[:40] + "..." if len(value) > 40 else value for key, value in topic_labels.items()}
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.sort_values(["Topic", "Timestamp"])

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                                 mode='lines',
                                 marker_color=colors[index % 7],
                                 hoverinfo="text",
                                 name=topic_name,
                                 hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title="<b>Global Topic Representation",
        )
    )
    return fig

def top2vec_visualize_hierarchy(topic_model,
                        annotations,
                        reduced,
                        orientation: str = "left",
                        topics: List[int] = None,
                        title: str = "<b>Hierarchical Clustering</b>",
                        width: int = 1000,
                        height: int = 600,
                        hierarchical_topics: pd.DataFrame = None,
                        linkage_function: Callable[[csr_matrix], np.ndarray] = None,
                        distance_function: Callable[[csr_matrix], csr_matrix] = None,
                        color_threshold: int = 1) -> go.Figure:
    """ 
    Visualize a hierarchical structure of the topics
    Arguments:
        topic_model: A fitted Top2Vec instance.
        annotations: topic annotations,
        reduced: Bool (True if hierarchical topic reduction is used),
        orientation: The orientation of the figure.
                     Either 'left' or 'bottom'
        topics: A selection of topics to visualize
        title: Title of the plot.
        width: The width of the figure. Only works if orientation is set to 'left'
        height: The height of the figure. Only works if orientation is set to 'bottom'
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                             represented by their parents and their children.
                             NOTE: The hierarchical topic names are only visualized
                             if both `topics` and `top_n_topics` are not set.
        linkage_function: The linkage function to use. Default is:
                          `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
                          NOTE: Make sure to use the same `linkage_function` as used
                          in `topic_model.hierarchical_topics`.
        distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                           `lambda x: 1 - cosine_similarity(x)`.
                            You can pass any function that returns either a square matrix of 
                            shape (n_samples, n_samples) with zeros on the diagonal and 
                            non-negative values or condensed distance matrix of shape 
                            (n_samples * (n_samples - 1) / 2,) containing the upper 
                            triangular of the distance matrix.
                           NOTE: Make sure to use the same `distance_function` as used
                           in `topic_model.hierarchical_topics`.
        color_threshold: Value at which the separation of clusters will be made which
                         will result in different colors for different clusters.
                         A higher value will typically lead in less colored clusters.
    Returns:
        fig: A plotly figure
    """
    if distance_function is None:
        distance_function = lambda x: 1 - cosine_similarity(x)

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

    # Select topics based on top_n and topics args
    unique_topics = list(set(annotations))
    counts = [annotations.count(t) for t in unique_topics]

    freq_df = pd.DataFrame(data={'Topic': unique_topics, 'Count': counts})
    freq_df = freq_df.sort_values(by=['Count'], ascending=False)
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    topics = freq_df.Topic.tolist()

    # Select topic embeddings
    embeddings = topic_model.topic_vectors
    if reduced:
        embeddings = topic_model.topic_vectors_reduced
    
    # Annotations
    annotations = None

    # wrap distance function to validate input and return a condensed distance matrix
    distance_function_viz = lambda x: validate_distance_matrix(
        distance_function(x), embeddings.shape[0])
    
    # Create dendogram
    fig = ff.create_dendrogram(embeddings,
                               orientation=orientation,
                               distfun=distance_function_viz,
                               linkagefun=linkage_function,
                               hovertext=annotations,
                               color_threshold=color_threshold)

    # Create nicer labels
    axis = "yaxis" if orientation == "left" else "xaxis" 
    new_labels = ['_'.join([x]+topic_model.topic_words[int(x)][:3].tolist()) if not reduced else '_'.join([x]+topic_model.topic_words_reduced[int(x)][:3].tolist()) for x in fig.layout[axis]['ticktext']]

    # Stylize layout
    fig.update_layout(
        plot_bgcolor='#ECEFF1',
        template="plotly_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    # Stylize orientation
    if orientation == "left":
        fig.update_layout(height=200 + (15 * len(topics)),
                          width=width,
                          yaxis=dict(tickmode="array",
                                     ticktext=new_labels))

        # Fix empty space on the bottom of the graph
        y_max = max([trace['y'].max() + 5 for trace in fig['data']])
        y_min = min([trace['y'].min() - 5 for trace in fig['data']])
        fig.update_layout(yaxis=dict(range=[y_min, y_max]))

    else:
        fig.update_layout(width=200 + (15 * len(topics)),
                          height=height,
                          xaxis=dict(tickmode="array",
                                     ticktext=new_labels))

    if hierarchical_topics is not None:
        for index in [0, 3]:
            axis = "x" if orientation == "left" else "y"
            xs = [data["x"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            ys = [data["y"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            hovertext = [data["text"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]

            fig.add_trace(go.Scatter(x=xs, y=ys, marker_color='black',
                                     hovertext=hovertext, hoverinfo="text",
                                     mode='markers', showlegend=False))
    return fig

def top2vec_visualize_barchart(top2vec_model, hierarchy, topics: List[int] = None, top_n_topics: int = 8, 
                               n_words: int = 10, custom_labels: Union[bool, str] = False, 
                               title: str = "<b>Topic Word Scores</b>", width: int = 250, height: int = 250) -> go.Figure:
    """ 
    Visualize a barchart of selected topics from a fitted Top2Vec instance.
    Arguments:
        top2vec_model: A fitted Top2Vec instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most relevant topics.
        n_words: Number of words to show in a topic.
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `top2vec_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of each figure.
        height: The height of each figure.

    Returns:
        fig: A plotly figure
    """
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    # Select topics based on top_n and topics args
    topics = list(range(top2vec_model.get_num_topics(reduced=hierarchy)))

    # Initialize figure
    subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:
        if not hierarchy:
            words = [word + "  " for word in top2vec_model.topic_words[topic]][:n_words][::-1]
            scores = top2vec_model.topic_word_scores[topic][:n_words][::-1]
        else:
            words = [word + "  " for word in top2vec_model.topic_words_reduced[topic]][:n_words][::-1]
            scores = top2vec_model.topic_word_scores_reduced[topic][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{title}",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig

def validate_distance_matrix(X, n_samples):
    """ Validate the distance matrix and convert it to a condensed distance matrix
    if necessary.

    A valid distance matrix is either a square matrix of shape (n_samples, n_samples) 
    with zeros on the diagonal and non-negative values or condensed distance matrix 
    of shape (n_samples * (n_samples - 1) / 2,) containing the upper triangular of the 
    distance matrix.
    
    Arguments:
        X: Distance matrix to validate.
        n_samples: Number of samples in the dataset.

    Returns:
        X: Validated distance matrix.

    Raises:
        ValueError: If the distance matrix is not valid.
    """
    # Make sure it is the 1-D condensed distance matrix with zeros on the diagonal
    s = X.shape
    if len(s) == 1:
        # check it has correct size
        n = s[0]
        if n != (n_samples * (n_samples -1) / 2):
            raise ValueError("The condensed distance matrix must have "
                            "shape (n*(n-1)/2,).")
    elif len(s) == 2:
        # check it has correct size
        if (s[0] != n_samples) or (s[1] != n_samples):
            raise ValueError("The distance matrix must be of shape "
                            "(n, n) where n is the number of samples.")
        # force zero diagonal and convert to condensed
        np.fill_diagonal(X, 0)
        X = squareform(X)
    else:
        raise ValueError("The distance matrix must be either a 1-D condensed "
                        "distance matrix of shape (n*(n-1)/2,) or a "
                        "2-D square distance matrix of shape (n, n)."
                        "where n is the number of documents."
                        "Got a distance matrix of shape %s" % str(s))

    # Make sure its entries are non-negative
    if np.any(X < 0):
        raise ValueError("Distance matrix cannot contain negative values.")

    return X

def top2vec_visualize_documents(topic_model,
                        annotations,
                        reduced,
                        title: str = "<b>Documents and Topics</b>",
                        width: int = 1200,
                        height: int = 750):
    """ 
    Visualize documents and their topics in 2D
    Arguments:
        topic_model: A fitted Top2Vec instance.
        reduced: Bool (True if hierarchical reduction was used)
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.
    """

    df = pd.DataFrame()
    df["doc"] = topic_model.documents
    df["topic"] = annotations

    # Extract embeddings
    embeddings_to_reduce = topic_model.document_vectors
    umap_model = UMAP(n_neighbors=15, metric='cosine', n_components=2, random_state=42).fit(embeddings_to_reduce)
    embeddings_2d = umap_model.embedding_

    unique_topics = topics = set(annotations)

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names
    topic_words, _, topic_nums = topic_model.get_topics(reduced=reduced)
    names = [f"{topic_num}_" + "_".join(topic[:3]) for topic_num, topic in zip(topic_nums, topic_words)]
    topic_labels = {name.split('_')[0]: name for name in names}

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = [-1]
    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc,
            hoverinfo="text",
            mode='markers+text',
            name="other",
            showlegend=False,
            marker=dict(color='#CFD8DC', size=5, opacity=0.5)
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc,
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5)
                )
            )

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig, topic_labels

# NMF___________________________________________________________________________________________________________________________________
def nmf_visualize_barchart(topic_model,
                       vectorizer,
                       annotations,
                       n_words: int = 10,
                       title: str = "<b>Topic Word Scores</b>",
                       width: int = 400,
                       height: int = 250) -> go.Figure:
    """ Visualize a barchart of selected topics

    Arguments:
        topic_model: A fitted NMF instance.
        vectorizer: A fitted vectorizer instance.
        annotations: Topic annotations for the corpus.
        n_words: Number of words to show in a topic
        title: Title of the plot.
        width: The width of each figure.
        height: The height of each figure.

    Returns:
        fig: A plotly figure
    """
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    unique_topics = list(range(topic_model.n_components_))
    topic_counts = np.bincount(sorted([int(a) for a in annotations]))
    
    freq_df = pd.DataFrame({'Topic': unique_topics, 'Count': topic_counts})
    topics = sorted(freq_df.Topic.to_list())

    # Initialize figure
    subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:

        # Get the top N words and their corresponding scores for the topic
        topic_word_scores = list(enumerate(topic_model.components_[topic]))
        top_words = sorted(topic_word_scores, key=lambda x: x[1], reverse=True)[:n_words]

        # Extract the words and their scores
        words = [vectorizer.get_feature_names_out()[word_idx] for word_idx, _ in top_words][::-1]
        scores = [score for _, score in top_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{title}",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig

#LDA___________________________________________________________________________________________________________________________________
def lda_visualize_barchart(topic_model,
                       vectorizer,
                       annotations,
                       n_words: int = 10,
                       title: str = "<b>Topic Word Scores</b>",
                       width: int = 400,
                       height: int = 250) -> go.Figure:
    """ 
    Visualize a barchart of selected topics
    Arguments:
        topic_model: A fitted LDA instance.
        vectorizer: A fitted vectorizer instance.
        annotations: Topic annotations for the corpus.
        n_words: Number of words to show in a topic
        title: Title of the plot.
        width: The width of each figure.
        height: The height of each figure.
    Returns:
        fig: A plotly figure
    """
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    unique_topics = list(range(len(topic_model.components_)))
    topic_counts = np.bincount(sorted([int(a) for a in annotations]))
    
    freq_df = pd.DataFrame({'Topic': unique_topics, 'Count': topic_counts})
    topics = sorted(freq_df.Topic.to_list())

    # Initialize figure
    subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:

        # Get the top N words and their corresponding scores for the topic
        topic_word_scores = list(enumerate(topic_model.components_[topic]))
        top_words = sorted(topic_word_scores, key=lambda x: x[1], reverse=True)[:n_words]

        # Extract the words and their scores
        words = [vectorizer.get_feature_names_out()[word_idx] for word_idx, _ in top_words][::-1]
        scores = [score for _, score in top_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{title}",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def nmf_lda_visualize_documents(
        topic_model,
        vectorizer,
        documents,
        X,
        annotations,
        title: str = "<b>Documents and Topics</b>",
        width: int = 1200,
        height: int = 750):
    """ 
    Visualize documents and their topics in 2D
    Arguments:
        topic_model: A fitted NMF or LDA instance.
        vectorizer: A fitten vectorizer instance.
        documents: Documents used to fit topic_model.
        X: Vectorized documents.
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.
    """

    df = pd.DataFrame(data={
        'doc': documents,
        'topic': annotations
        })

    # Extract embeddings
    embeddings_to_reduce = X
    umap_model = UMAP(n_neighbors=15, metric='cosine', n_components=2, random_state=42).fit(embeddings_to_reduce)
    embeddings_2d = umap_model.embedding_

    unique_topics = topics = set(annotations)

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names
    names = []
    for i, _ in enumerate(topic_model.components_):
        # Get the top N words and their corresponding scores for the topic
        topic_word_scores = list(enumerate(topic_model.components_[i]))
        top_words = sorted(topic_word_scores, key=lambda x: x[1], reverse=True)[:3]

        # Extract the words and their scores
        words = [vectorizer.get_feature_names_out()[word_idx] for word_idx, _ in top_words][::-1]
        names.append(f"{i}_" + "_".join(words))
        
    topic_labels = {name.split('_')[0]: name for name in names}

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = [-1]
    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc,
            hoverinfo="text",
            mode='markers+text',
            name="other",
            showlegend=False,
            marker=dict(color='#CFD8DC', size=5, opacity=0.5)
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc,
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5)
                )
            )

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig, topic_labels