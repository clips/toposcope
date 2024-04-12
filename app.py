import gradio as gr
import pandas as pd
import topic_modeling_app
import os, uuid

import smtplib 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

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

def generate_run_id():
    run_id = str(uuid.uuid4())
    return gr.update(value=run_id, visible=True)

def send_mail(receiver, run_id):
  
  """Sends email with pipeline output attached to user."""

  if receiver.strip(): #check if user actually wants to receive output

    # specify header
    msg = MIMEMultipart()
    msg['From'] = 'toposcope.ua@gmail.com'
    msg['To'] = receiver.strip()
    msg['Subject'] = f'Output {run_id}'

    # add body of message
    body = f'Dear Toposcope user,\n\nAttached you can find the output of run {run_id}.\n\nKind regards.'
    msg.attach(MIMEText(body, 'plain'))

    # attach output
    with open(f'./outputs/{run_id}.zip', 'rb') as file:
      msg.attach(MIMEApplication(file.read(), Name=f'{run_id}.zip'))
      text = msg.as_string()

    # set up mailing server
    path = '/var/www/CLARIAH-topic/app_password.txt' 
    if os.path.exists(path):
        with open(path) as f:
            app_password = f.read()

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('toposcope.ua@gmail.com', app_password)
        server.sendmail('toposcope.ua@gmail.com', receiver, text)
        server.quit()

def show_input(input_type):
    """
    Used to control which input widget is being shown.
    """
    if input_type == "Corpus":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def show_stopword_widget(preprocessing_steps):
    checked = True if 'remove custom stopwords' in preprocessing_steps else False
    if checked:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def show_model_params(choice):

    """
    Used to toggle parameters depending on the algorithm selection of the user.
    """

    if choice == "Top2Vec":
        return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]
    elif choice == "BERTopic":
        return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]
    elif choice == "NMF":
        return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)]
    else: #LDA
        return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)]

def visible_output(input_text):
    """
    Used to make the downloadable output widget visible after the submit button is clicked.
    This shows the progress bar (and the output zip when the main script has finished running).
    """
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

def visible_plots(_):
    """
    Used to make the output plots widgets visible after the main script has finished running.
    """
    return gr.update(visible=True)

with gr.Blocks(title="Toposcope", theme=theme, css=css) as demo:
    title = gr.Markdown("""# Toposcope""")
    
    with gr.Tab("Pipeline"):
        gr.Markdown("#### Provide input data")
        with gr.Row(variant='panel'):
            input_type = gr.Radio(choices=['Corpus', 'HuggingFace dataset'], value='Corpus', label='Type', interactive=True,
            info="""Upload your own corpus or use a publicly available dataset from the HuggingFace hub.""")

        # components
        with gr.Row(variant='panel'):
            with gr.Column(visible=True) as corpus_input:
                file = gr.File(file_types = ['.csv', '.zip'], file_count = "single")
            with gr.Column(visible=False) as huggingface_input:
                dataset = gr.Textbox(label="Name", info="Dataset identifier mentioned on the HuggingFace hub.")
                subset = gr.Textbox(label="Subset", info="Mandatory if dataset contains subsets.")
                split = gr.Textbox(label="Split", info="Mandatory if dataset contains splits.")

        input_type.change(
            show_input, input_type, [corpus_input, huggingface_input]
        )

        with gr.Row(variant='panel'):
            lang = gr.Dropdown(["Dutch", "English", "French", "German"], label="Language", value="English", info='Language of the data (only relevant when applying tokenization, lemmatization or stopword removal).', interactive=True)
            text_col = gr.Textbox(label="Text column name", info="Name of column containing the documents (if .csv file or Huggingface dataset).", interactive=True)
            timestamp_col = gr.Textbox(label="Timestamp column name (optional)", info="If input is .csv file or Huggingface dataset, pass timestamp column name to compute topics over time", interactive=True)

        # Preprocessing parameters
        gr.Markdown("#### Preprocess the corpus")
        with gr.Row(variant='panel'):
            preprocessing_steps = gr.CheckboxGroup(["tokenize", "lemmatize", "remove NLTK stopwords", "remove custom stopwords", "remove punctuation", "lowercase"], label="Steps", info="Tokenization is mandatory when 'lemmatize' is selected. If tokenization is not used, text will be split on whitespace to obtain tokens."), 

        with gr.Row(variant='panel', visible=False) as stopword_widget:
            stopword_file = gr.File(file_types = ['.txt'], file_count = "single", label='Stopword file', show_label=True)

        gr.Markdown("#### Customize the algorithm")
        with gr.Row(variant='panel'):
            algorithm = gr.Dropdown(["Top2Vec", "BERTopic", "NMF", "LDA"], label="Algorithm", interactive=True)

        # Model parameters
        with gr.Row(visible=False, variant='panel') as params:
            model = gr.Textbox(label="Embedding model", value="all-MiniLM-L6-v2", info="See user guidelines for options\n(default='all-MiniLM-L6-v2')", interactive=True, visible=False)
            n_topics = gr.Textbox(label="Number of topics", info="When using BERTopic/Top2Vec, passing '0' will automatically optimize this value.", value='10', interactive=True, visible=False)
            min_topic_size = gr.Textbox(label="Min. number of texts per topic", info="Must be > 1; Higher values will result in fewer topics.", interactive=True, value='2', visible=False)
            upper_ngram_range = gr.Dropdown(["1", "2", "3", "4"], value="1", label="Upper n-gram range", info="Note that higher range increases processing time.", interactive=True)

        with gr.Row(variant="panel"):
            receiver = gr.Textbox(label='E-mail', info="Please provide your e-mail address to receive the output in your mailbox (optional). Personal info will not be saved or used for any other purpose than this application.")

        # Toggle stopword file widget
        preprocessing_steps[0].change(
            show_stopword_widget, preprocessing_steps[0], [stopword_widget]
        )

        # Toggle parameters depending on algorithm selection
        algorithm.change(
            show_model_params, algorithm, [params, model, n_topics, min_topic_size, upper_ngram_range]
        )

        # Start running pipeline
        submit_button = gr.Button('Submit', variant='primary')

        # outputs
        with gr.Row():
            run_id = gr.Textbox(label='Run index', visible=False, interactive=False)

        zip_out = gr.File(label='Output', visible=False)
        doc_plot = gr.Plot(label='Topic-document plot', show_label=True, visible=False)

        submit_button.click( # first generate run index
            generate_run_id,
            outputs=run_id,
            trigger_mode="once",
        ).then( # then make output components visible
            visible_output, 
            inputs=file, 
            outputs=[run_id, zip_out, doc_plot]
            ).then( # then run pipeline
                topic_modeling_app.main, 
                inputs=[input_type, file, dataset, subset, split, text_col, stopword_file, lang, algorithm, preprocessing_steps[0], model, min_topic_size, timestamp_col, n_topics, upper_ngram_range, run_id], 
                outputs=[zip_out, doc_plot]
                ).then( # then make output plot visible
                    visible_plots,
                    inputs=doc_plot,
                    outputs=doc_plot
                ).then(
                    send_mail,
                    inputs=[receiver, run_id]
                )
        
    with gr.Tab("User guidelines"):
        gr.Markdown("""
### Algorithm selection
- **Top2Vec** (Angelov, 2020): State-of-the-art neural algorithm (default option). Text representations can be based on a pre-trained transformer, or new embeddings can be generated from the input corpus (by setting base model to "Doc2Vec" in the configuration). The latter can be useful if the corpus has a substantial size and when it is expected that the type of language used in the corpus is not represented in the pre-trained language models that Top2Vec features: 'universal-sentence-encoder' (English), 'universal-sentence-encoder-multilingual', 'distiluse-base-multilingual-cased', 'all-MiniLM-L6-v2' (English), and 'paraphrase-multilingual-MiniLM-L12-v2'. Note, however, that training an embedding model with Doc2Vec is substantially slower than using one of the pre-trained models.
- **BERTopic** (Grootendorst, 2022): State-of-the-art neural algorithm that can be used with any pre-trained transformer model that is publicly available on the [Huggingface hub](https://huggingface.co/models). This is useful when you want to use a model that is trained on the same (or similar) data as your corpus. The default model that BERTopic uses is "all-MiniLM-L6-v2" (for English data only). Other models that can be used are, for instance, [BERTje](https://huggingface.co/GroNLP/bert-base-dutch-cased)/[RobBERT](https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-base) (Dutch), [CamemBERT](https://huggingface.co/almanach/camembert-base) (French), [German BERT](https://huggingface.co/google-bert/bert-base-german-cased), and [multi-lingual BERT](https://huggingface.co/google-bert/bert-base-multilingual-cased) (other languages)
- **NMF/LDA**: Classical machine learning algorithms, which generally perform worse than Top2Vec and BERTopic, but worth investigating when working with a large corpus that contains relatively long texts.

The selection of the embedding model used in Top2Vec or BERTopic should, a.o. depend on the language of your data. If your corpus contains texts written in multiple languages, it is recommended to use a multilingual model. It is also recommended to use a multi-lingual model if the corpus contains texts written in any language other than Dutch, English, French, or German. When no pre-trained mono- or multi-lingual model that was trained on the relevant language (or dialect / historical variant) exists, it is best to either train a new model with Top2Vec using Doc2Vec to generate embeddings, or use a model that was pre-trained on a structurally similar language (e.g. use a Dutch model for Afrikaans).

### Preprocessing
When using the classical machine learning algorithms (NMF/LDA), it is recommended to apply all preprocessing steps provided in the pipeline (tokenization, lemmatization, lowercasing, and removing stopwords and punctuation). For the neural models, it is not required, since they rely on more sophisticated methods, but experimenting with different preprocessing steps could still result in improvements. Note that when selecting lemmatization, it is important to also apply tokenization. Note that multi-lingual preprocessing is currently not supported.     

### Model parameter tuning and evaluation of the results
Which model and hyperparameters are optimal depends on the data that is used. Therefore, optimization experiments are necessary to find the best configuration. To evaluate the results of the topic modeling algorithm, it is important to investigate both the quantitative results - the diversity and coherence scores - but also the qualitative results by looking at the individual topic predictions, visualizations, and the most important keywords per topic. 

### Automatic evaluation
In order to help the user evaluate the topic model, diversity and coherence scores are computed. The diversity score, i.e. the proportion of unique words, indicates how "diverse" the different topics are from one another. A diversity of score close to 0 indicates redundant topics, whereas a score of 1 indicates high diversity between topics. Coherence, on the other hand, indicates how frequently words within a topic co-occur. The coherence score can range from -14 to +14 (higher is better).
        
### References
        Dimo Angelov (2020). Top2Vec: Distributed Representations of Topics. arXiv:2008.09470
        Maarten Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794
        """)
    
    with gr.Tab("About"):
        gr.Markdown("""        
        ### Project
        Toposcope is a topic modeling pipeline that was developed by [CLiPS](https://www.uantwerpen.be/en/research-groups/clips/) ([University of Antwerp](https://www.uantwerpen.be/en/)) during the [CLARIAH-VL](https://clariahvl.hypotheses.org/) project.
        The code is available here: https://github.com/clips/toposcope.

        ### Contact
        If you have questions, please send them to [Jens Lemmens](mailto:jens.lemmens@uantwerpen.be) or [Walter Daelemans](mailto:walter.daelemans@uantwerpen.be).
        """)
    
    with gr.Row():
        gr.Markdown("""<center><img src="https://platformdh.uantwerpen.be/wp-content/uploads/2019/03/clariah_def.png" alt="Image" width="200"/></center>""")
        gr.Markdown("""<center><img src="https://thomasmore.be/sites/default/files/2022-11/UA-hor-1-nl-rgb.jpg" alt="Image" width="175"/></center>""")


demo.launch(share=False, server_port=7861, server_name='0.0.0.0')
