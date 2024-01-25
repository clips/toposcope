import gradio as gr
import pandas as pd
import topic_modeling_app

css = """
h1 {
    display: block;
    text-align: center;
    font-size: 32pt;
}
.progress-bar-wrap.progress-bar-wrap.progress-bar-wrap
{
	border-radius: var(--input-radius);
	height: 1.25rem;
	margin-top: 1rem;
	overflow: hidden;
	width: 70%;
}
"""
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="amber",
).set(
    button_primary_background_fill='*secondary_500',
    button_primary_background_fill_hover='*secondary_400',
    block_label_background_fill='*primary_50',
)

def show_model_params(choice):

    """
    Used to toggle parameters depending on the algorithm selection of the user.
    """

    if choice == "Top2Vec":
        return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
    elif choice == "BERTopic":
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]
    elif choice == "NMF":
        return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)]
    else: #LDA
        return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]

def visible_output(input_text):
    """
    Used to make the downloadable output widget visible after the submit button is clicked.
    This shows the progress bar (and the output zip when the main script has finished running).
    """
    return gr.update(visible=True), gr.update(visible=False)

def visible_plots(file):
    """
    Used to make the output plots widgets visible after the main script has finished running.
    """
    return gr.update(visible=True), gr.update(visible=True)

with gr.Blocks(title="CLARIAH-VL Topic Modeling Pipeline", theme=theme, css=css) as demo:
    title = gr.Markdown("""# CLARIAH-VL Topic Modeling Pipeline""")
    
    with gr.Tab("Pipeline"):

        # components
        with gr.Row(variant='panel'):
            file = gr.File(file_types = ['.csv', '.zip'], file_count = "single")

        with gr.Row(variant='panel'):
            lang = gr.Dropdown(["Dutch", "English", "French", "German"], label="Language", value="English", interactive=True)

        with gr.Row(variant='panel'):
            timestamp_col = gr.Textbox(label="Timestamp column name (optional)", info="If input file is .csv, pass timestamp column name to compute topics over time", interactive=True)

        # Preprocessing parameters
        with gr.Row(variant='panel'):
            preprocessing_steps = gr.CheckboxGroup(["tokenize", "lemmatize", "remove NLTK stopwords", "remove custom stopwords", "remove punctuation", "lowercase"], label="Text preprocessing steps"), 

        with gr.Row(variant='panel'):
            algorithm = gr.Dropdown(["Top2Vec", "BERTopic", "NMF", "LDA"], label="Algorithm", interactive=True)

        # Model parameters
        with gr.Row(visible=False, variant='panel') as top2vec_params:
            model = gr.Dropdown(["doc2vec (language-independant)", "universal-sentence-encoder (English)", "universal-sentence-encoder-multilingual", "all-MiniLM-L6-v2 (English)", "distiluse-base-multilingual-cased", "paraphrase-multilingual-MiniLM-L12-v2"], label="Embedding model", info="Model for generating doc representations", interactive=True)
            n_topics = gr.Textbox(label="Number of topics", info="'0' -> algorithm determines num. topics", interactive=True, value='0')
            min_topic_size = gr.Textbox(label="Min. number of docs per topic", info="Must be > 1", interactive=True, value='2')
            upper_ngram_range = gr.Textbox(label="Upper token n-gram range", info="<4 is recommended", value='1', interactive=True)
        
        with gr.Row(visible=False, variant='panel') as bertopic_params:
            model = gr.Textbox(label="Embedding model", value="all-MiniLM-L6-v2 (English)", info="https://huggingface.co/models", interactive=True)
            n_topics = gr.Textbox(label="Number of topics", info="'0' -> algorithm determines num. topics", value='0', interactive=True)
            min_topic_size = gr.Textbox(label="Min. number of docs per topic", info="Must be > 1", interactive=True, value='2')
            upper_ngram_range = gr.Textbox(label="Upper token n-gram range", info="<4 is recommended", value='1', interactive=True)

        with gr.Row(visible=False, variant='panel') as nmf_params:
            n_topics = gr.Textbox(label="Number of topics", info="Must be > 1", value='10', interactive=True)
            upper_ngram_range = gr.Textbox(label="Upper token n-gram range", info="<4 is recommended", value='1', interactive=True)

        with gr.Row(visible=False, variant='panel') as lda_params:
            n_topics = gr.Textbox(label="Number of topics", info="Must be > 1", value='10', interactive=True)
            upper_ngram_range = gr.Textbox(label="Upper token n-gram range", info="<4 is recommended", value='1', interactive=True)

        # Toggle parameters depending on algorithm selection
        algorithm.change(
            show_model_params, algorithm, [top2vec_params, bertopic_params, nmf_params, lda_params]
        )

        # Start running pipeline
        submit_button = gr.Button('Submit', variant='primary')

        # outputs
        zip_out = gr.File(label='Output', visible=False)
        doc_plot = gr.Plot(label='Topic-document plot', show_label=True, visible=False)
        
        submit_button.click( # first make zip output component visible (so that progress bar is visible)
            visible_output, 
            inputs=file, 
            outputs=[zip_out, doc_plot]
            ).then( # then run pipeline
                topic_modeling_app.main, 
                inputs=[file, lang, algorithm, preprocessing_steps[0], model, min_topic_size, timestamp_col, n_topics, upper_ngram_range], 
                outputs=[zip_out, doc_plot]
                ).then( # then make output plots visible
                    visible_plots,
                    inputs=file,
                    outputs=[doc_plot]
                )
        
    with gr.Tab("User guidelines"):
        gr.Markdown("""
### Algorithm selection
- **Top2Vec** (Angelov, 2020): State-of-the-art neural algorithm (default option). Text representations can be based on a pre-trained transformer, or new embeddings can be generated from the input corpus (by setting base model to "Doc2Vec" in the configuration). The latter can be useful if the corpus has a substantial size and when it is expected that the type of language used in the corpus is not represented in the pre-trained language models that Top2Vec features: 'universal-sentence-encoder', 'universal-sentence-encoder-multilingual', 'distiluse-base-multilingual-cased', 'all-MiniLM-L6-v2', and 'paraphrase-multilingual-MiniLM-L12-v2'. 
- **BERTopic** (Grootendorst, 2022): State-of-the-art neural algorithm that can be used with any publicly-available pre-trained transformer model on the HuggingFace hub: https://huggingface.co/models. This is useful when you want to use a model that is trained on the same (or similar) data as your corpus. The default model that BERTopic uses is "all-MiniLM-L6-v2" (for English data only).
- **NMF/LDA**: Classical machine learning algorithms, which generally perform worse than Top2Vec and BERTopic, but worth investigating when working with a large corpus that contains relatively long texts.

The selection of the embedding model used in Top2Vec or BERTopic should, a.o. depend on the language of your data. If your corpus contains texts written in multiple languages, it is recommended to use a multilingual model. When no model is available for a specific language, it is best to not use a pre-trained model, i.e. use Top2Vec with Doc2Vec as embedding method (or use NMF/LDA), although models pre-trained on similar languages might be worth trying.

### Preprocessing
When using the classical machine learning algorithms (NMF/LDA), it is recommended to apply all preprocessing steps provided in the pipeline (tokenization, lemmatization, lowercasing, and removing stopwords and punctuation). For the neural models, it is not required, since they rely on more sophisticated methods, but experimenting with different preprocessing steps could still result in improvements. Note that when selecting lemmatization, it is important to also apply tokenization.

### Model parameter tuning and evaluation of the results
Which model and hyperparameters are optimal depends on the data that is used. Therefore, optimization experiments are necessary to find the best configuration. To evaluate the results of the topic modeling algorithm, it is important to investigate both the quantitative results - the diversity and coherence scores - but also the qualitative results by looking at the individual topic predictions, visualizations, and the most important keywords per topic. 

### Automatic evaluation
In order to help the user evaluate the topic model, diversity and coherence scores are computed. The diversity score, i.e. the proportion of unique words, indicates how "diverse" the different topics are from one another. A diversity of score close to 0 indicates redundant topics, whereas a score of 1 indicates high diversity between topics. Coherence, on the other hand, indicates how frequently words within a topic co-occur. The coherence score can range from -14 to +14 (higher is better).
        
### References
        Dimo Angelov: 2020. Top2Vec: Distributed Representations of Topics. arXiv: 2008.09470
        Maarten Grootendorst: 2022. BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794
        """)
    
    with gr.Tab("About"):
        gr.Markdown("""        
        ### Project
        The topic modeling pipeline was developed by [CLiPS](https://www.uantwerpen.be/en/research-groups/clips/) ([University of Antwerp](https://www.uantwerpen.be/en/)) during the [CLARIAH-VL](https://clariahvl.hypotheses.org/) project.
        The code is available here: https://github.com/LemmensJens/CLARIAH-topic

        ### Contact
        If you have questions, please send them to [Jens Lemmens](mailto:jens.lemmens@uantwerpen.be) or [Walter Daelemans](mailto:walter.daelemans@uantwerpen.be)
        """)
    
    with gr.Row():
        gr.Markdown("""<center><img src="https://platformdh.uantwerpen.be/wp-content/uploads/2019/03/clariah_def.png" alt="Image" width="200"/></center>""")
        gr.Markdown("""<center><img src="https://thomasmore.be/sites/default/files/2022-11/UA-hor-1-nl-rgb.jpg" alt="Image" width="175"/></center>""")


demo.queue(default_concurrency_limit=10).launch(share=True, allowed_paths=['./assets/'])