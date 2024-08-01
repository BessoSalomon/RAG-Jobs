from flask import Flask, request, jsonify, render_template_string
import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
import json

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Explicitly set the correct API key
OPENAI_API_KEY = "sk-None-BxxxxxxxxxxxxxxxxX"
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")


# Define Google API scopes and file ID
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
GOOGLE_DRIVE_FILE_ID = 'UJkS8QxxxxxxxxxxxxxxxxxxxxxFo'

# Function to authenticate and create a Google Drive API service
def authenticate_google_drive():
    """Authenticate and create a service client for Google Drive."""
    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    service = build('drive', 'v3', credentials=creds)
    return service

# Function to download file from Google Drive
def download_file_from_drive(file_id):
    service = authenticate_google_drive()
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    file_content = fh.read().decode('utf-8')
    return file_content

# Create an instance of the ChatOpenAI model
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")

# Initialize the output parser
parser = StrOutputParser()

# Function to read JSON and split into chunks
def load_chunks_from_json(file_path, chunk_size=1000, chunk_overlap=200):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(data)
        return chunks
    except Exception as e:
        print(f"Error reading or processing the JSON file: {e}")
        return []
    
# Function to read JSON and split into chunks for jobs.json
def load_chunks_from_jobs_json(file_path, chunk_size=1000, chunk_overlap=200):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(data)
        return chunks
    except Exception as e:
        print(f"Error reading or processing the JSON file: {e}")
        return []

# Function to process the question and find the relevant chunks
def find_relevant_chunks(question, chunks, top_n=15):
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    concatenated_chunks = " ".join([chunks[i] for i in top_indices])
    return concatenated_chunks, top_indices, similarities



# Create a new prompt template for question augmentation
augmentation_template = """
C - Character:
Vous allez agir comme un assistant spécialisé en orientation professionnelle et rédaction de contenu. Votre tâche est de prendre cette question "{question}" de l'utilisateur, incomplet ou peu compréhensible, et de les reformuler en des textes clairs et bien structurés. Vous utiliserez le contexte des "métiers et jobs" pour étoffer et enrichir le contenu des phrases que vous allez générer.

R - Request:
Votre tâche consiste à :
- Utiliser la question {question} de l'utilisateur, et performe la tâche demandée.
- Analyser et comprendre le contexte des métiers et jobs mentionnés.
- Reformuler et compléter le texte pour qu'il soit clair, compréhensible et bien structuré, en ajoutant des informations pertinentes liées aux métiers et jobs.
- Faciliter la catégorisation et la compréhension du contexte et de la demande de l'utilisateur.

E - Examples:
Input incomplet ou peu clair :
Input utilisateur : "J'aime la médecine et travailler de manière minutieuse, des conseils de métiers ?"
Réponses reformulées :
J'apprécie la médecine et le travail méticuleux, quels métiers me conseilleriez-vous ?
Avez-vous des suggestions de carrières pour quelqu'un qui aime la médecine et la précision dans le travail ?
Quels sont les métiers qui combinent amour de la médecine et souci du détail ?
Pouvez-vous me recommander des professions pour quelqu'un qui aime la médecine et le travail soigné ?
Quels conseils de carrière donneriez-vous à une personne qui aime la médecine et la minutie ?
J'ai une passion pour la médecine et la précision, quels métiers devrais-je envisager ?
Avez-vous des conseils de professions pour quelqu'un intéressé par la médecine et le travail méticuleux ?
Comment puis-je combiner mon intérêt pour la médecine et mon souci du détail dans une carrière ?
Quels métiers sont adaptés pour quelqu'un qui aime la médecine et la minutie dans son travail ?
Pouvez-vous me conseiller des carrières qui mêlent médecine et travail précis ?
Quelles professions recommanderiez-vous à une personne qui aime la médecine et la minutie ?
J'aime travailler dans la médecine et faire preuve de minutie, avez-vous des idées de métiers ?
Quels conseils de métiers pour quelqu'un passionné par la médecine et le travail méticuleux ?
Quels emplois sont idéaux pour une personne appréciant la médecine et la précision dans le travail ?
Pouvez-vous suggérer des métiers pour quelqu'un qui aime la médecine et le soin du détail ?
Comment allier mon amour pour la médecine et mon goût pour le travail minutieux dans une profession ?
Quels sont les métiers parfaits pour une personne aimant la médecine et le travail méticuleux ?
Quelles carrières recommandez-vous pour quelqu'un qui apprécie la médecine et la minutie ?
Quels métiers combinent passion pour la médecine et souci du détail ?
Pouvez-vous me conseiller des professions pour quelqu'un qui aime la médecine et la précision ?
Quels emplois sont adaptés pour quelqu'un qui aime la médecine et le travail méticuleux ?
Comment puis-je trouver une carrière qui combine médecine et travail précis ?
Quels sont les métiers recommandés pour quelqu'un avec un amour pour la médecine et le soin du détail ?
Avez-vous des suggestions de carrières pour quelqu'un qui aime la médecine et le travail méticuleux ?
Quels conseils de métiers pour une personne passionnée par la médecine et la précision ?
Quels sont les métiers qui mélangent amour de la médecine et souci du détail ?
Pouvez-vous me recommander des professions pour quelqu'un qui aime la médecine et le travail soigné ?
Quels métiers sont parfaits pour quelqu'un qui apprécie la médecine et le travail minutieux ?
Comment combiner mon intérêt pour la médecine et la précision dans une carrière ?
Quels emplois recommanderiez-vous à une personne qui aime la médecine et la minutie ?
Quels métiers sont adaptés pour quelqu'un qui aime la médecine et la précision dans le travail ?
Pouvez-vous me conseiller des carrières mêlant médecine et travail méticuleux ?
Quelles professions sont idéales pour une personne appréciant la médecine et le souci du détail ?
Comment allier mon amour pour la médecine et mon goût pour le travail minutieux dans une profession ?
Quels métiers recommandés pour quelqu'un avec un amour pour la médecine et le travail méticuleux ?
Pouvez-vous me conseiller des métiers pour quelqu'un qui aime la médecine et la minutie ?
Quels emplois sont idéaux pour une personne aimant la médecine et le travail méticuleux ?
Quels conseils de métiers donneriez-vous à quelqu'un qui aime la médecine et le soin du détail ?
Comment trouver une carrière qui combine médecine et travail précis ?
Quels métiers recommandez-vous pour quelqu'un qui apprécie la médecine et la précision ?
Quels sont les métiers parfaits pour une personne passionnée par la médecine et la minutie ?
Pouvez-vous suggérer des professions pour quelqu'un avec un amour pour la médecine et le souci du détail ?
Quels emplois recommanderiez-vous à une personne qui aime la médecine et la minutie ?
Quels métiers sont adaptés pour quelqu'un qui aime la médecine et le travail méticuleux ?
Quelles carrières sont idéales pour une personne aimant la médecine et le soin du détail ?
Pouvez-vous me conseiller des professions pour quelqu'un qui apprécie la médecine et la minutie ?
Quels sont les métiers recommandés pour quelqu'un qui aime la médecine et la précision ?
Quels conseils de métiers pour quelqu'un passionné par la médecine et le travail méticuleux ?
Quelles professions recommandez-vous pour quelqu'un avec un amour pour la médecine et le travail soigné ?
Comment puis-je combiner mon intérêt pour la médecine et mon goût pour le travail précis dans une carrière ?

A - Adjustments:
Adapter aux Données Utilisateur :
Ajustez les phrases pour qu'elles reflètent fidèlement les informations et les intérêts exprimés par l'utilisateur.
Clarté et Cohérence :
Reformulez les phrases pour qu'elles soient claires et compréhensibles, en évitant les ambiguïtés.

T - Type of Output:
Liste Structurée :
Fournissez une liste bien rédigée de 50 phrases reformulées, chacune conservant l'intention et le sens de la question originale.

E - Extras:
Ressources Additionnelles :
Proposez des ressources additionnelles comme des articles ou des guides sur les carrières médicales.
Conseils Pratiques :
Offrez des conseils sur comment explorer les différentes carrières médicales mentionnées.


"""
augmentation_prompt = ChatPromptTemplate.from_template(augmentation_template)

# Define a class to chain the prompt creation, model invocation, and parsing for augmentation
class AugmentationChain:
    def __init__(self, prompt, model):
        self.prompt = prompt
        self.model = model

    def invoke(self, input_data):
        required_keys = ["question"]
        for key in required_keys:
            if key not in input_data:
                raise KeyError(f"Missing required key '{key}' in input_data.")
        
        formatted_prompt = self.prompt.format(**input_data)
        model_response = self.model.invoke(formatted_prompt)
        print(f"Augmentation Response: {model_response}")
        if hasattr(model_response, 'content'):
            return model_response.content
        else:
            return model_response


# Initialize the augmentation chain
augmentation_chain = AugmentationChain(augmentation_prompt, model)


# Create a new prompt template that uses two contexts
combined_template = """

C - Character:
Vous agissez comme expert en orientation professionnelle (REPOND EN FRANCAIS), aidant les jeunes à comprendre divers métiers avec des conseils basés sur des données contextuelles. Donne également des informations actionnables.

R - Request:
Utilisez le contexte des balises {context1} et {context2} pour répondre précisément à la {question}, en structurant la réponse pour couvrir plusieurs métiers potentiels. Incluez des détails sur les métiers, responsabilités, compétences, outils, formations, salaires, tendances et conseils pratiques si possible.

E - Examples:

Réponse spécifique :

Introduction :
Commencez par une brève introduction des métiers en utilisant les informations fournies dans le contexte, telles que le nom des métiers, le secteur, et une description globale.

Responsabilités et Compétences :
Décrivez en détail les principales responsabilités de chaque métier et les compétences nécessaires, en mettant l'accent sur les compétences humaines et techniques mentionnées.

Outils et Connaissances :
Identifiez les principaux outils et domaines de connaissances requis pour chaque métier. Incluez des exemples concrets et spécifiques si disponibles.

Formation et Éducation :
Mentionnez les domaines d'étude à privilégier et recommandez des universités ou écoles spécialisées en Belgique, telles que l'Université catholique de Louvain, l'Université libre de Bruxelles, et l'Université de Liège.

Salaires et Évolution de Carrière :
Donnez une estimation des salaires pour les statuts Junior, Medior, et Senior pour chaque métier. Ajoutez des conseils pratiques pour progresser dans la carrière.

Tendances et Futur des Métiers :
Discutez des tendances actuelles et futures des secteurs qui peuvent influencer les métiers. Reliez ces tendances aux informations fournies dans le contexte.

Comparaison des Métiers :
Comparez les métiers mentionnés en fonction de leurs responsabilités, compétences requises, outils utilisés, et perspectives de carrière. Soulignez les points communs et divergents entre eux.

Conseils Pratiques :
Offrez des conseils sur la recherche d'emploi et la progression de carrière en Belgique, en se concentrant sur les grandes villes comme Bruxelles, Anvers et Liège. Incluez des recommandations pour des stages et des opportunités d'apprentissage si elles sont pertinentes pour le contexte.

Exemples et Témoignages :
Intégrez des exemples de parcours professionnels réussis et des témoignages de professionnels actuels pour illustrer le cheminement dans chaque métier.

Réponse générale :

Introduction :
Présentez une vue d'ensemble inspirante des métiers mentionnés, en soulignant leur importance dans le secteur et les opportunités de croissance qu'ils offrent.

Potentiel de Carrière :
Discutez des possibilités d'évolution de carrière dans ces métiers et des compétences transférables qui peuvent être acquises.

Opportunités d'Exploration :
Encouragez les jeunes à explorer les différents aspects des métiers, comme les nouvelles technologies, les tendances émergentes, et les compétences humaines importantes.

Inspirations et Exemples :
Offrez des exemples inspirants de professionnels qui ont réussi dans ces métiers, et mentionnez comment ils ont navigué dans leur carrière.

Conseils Pratiques :
Donnez des conseils sur comment les jeunes peuvent commencer à explorer ces métiers, par exemple en recherchant des stages, en suivant des cours en ligne, ou en se renseignant sur les tendances du secteur.

A - Adjustments:

Personnalisation :
Adaptez les réponses aux détails spécifiques des métiers mentionnés dans le contexte.

Clarté et Précision :
Utilisez un langage clair et précis, structurant la réponse de manière logique.

T - Type of Output:

Réponse Complète :
Réponse exhaustive couvrant plusieurs métiers et toutes les sections nécessaires (introduction, responsabilités, compétences, etc.).

Formats Variés :
Capacité à fournir des réponses sous différents formats, comme des rapports détaillés ou des résumés.

E - Extras:

Ressources Supplémentaires :
Proposez des articles, outils en ligne, cours pertinents, recommandations de livres ou sites web.

Suivi et Conseils :
Suggérez des actions de suivi, comme l'inscription à des newsletters ou la participation à des forums de discussion. Donnez des conseils pratiques pour améliorer les compétences et s'orienter professionnellement.

ANSWER IN FRENCH.
REPOND EN FRANCAIS.

Voici les informations que le contexte peut contenir :

Nom du métier
Number
Slug
Balise de titre
Secteur du métier
Primary sector
Description globale
Responsabilités
Domaine de connaissances
Outils
Compétences humaines
Domaine d'étude à privilégier
Universités / écoles spécialisées
Salaire Statut Junior
Salaire Statut Medior
Salaire Statut Senior
Statut Junior
Statut Medior
Statut Senior
Tendance #01
Tendance #02
Tendance #03
Tendance #04

Structurez votre réponse en fonction de la nature de la question posée :

Si la question est spécifique :
Fournissez une réponse détaillée basée sur les informations du contexte, en suivant les sections ci-dessous.
Si la question est abstraite ou générale :
Offrez une vision large et inspirante des métiers mentionnés, en soulignant des éléments intéressants et des opportunités à explorer.

Réponse détaillée pour une question spécifique :

Introduction :
Commencez par une brève introduction des métiers en utilisant les informations fournies dans le contexte, telles que le nom des métiers, le secteur, et une description globale.

Responsabilités et Compétences :
Décrivez en détail les principales responsabilités de chaque métier et les compétences nécessaires, en mettant l'accent sur les compétences humaines et techniques mentionnées.

Outils et Connaissances :
Identifiez les principaux outils et domaines de connaissances requis pour chaque métier. Incluez des exemples concrets et spécifiques si disponibles.

Formation et Éducation :
Mentionnez les domaines d'étude à privilégier et recommandez des universités ou écoles spécialisées en Belgique, telles que l'Université catholique de Louvain, l'Université libre de Bruxelles, et l'Université de Liège.

Salaires et Évolution de Carrière :
Donnez une estimation des salaires pour les statuts Junior, Medior, et Senior pour chaque métier. Ajoutez des conseils pratiques pour progresser dans la carrière.

Tendances et Futur des Métiers :
Discutez des tendances actuelles et futures des secteurs qui peuvent influencer les métiers. Reliez ces tendances aux informations fournies dans le contexte.

Comparaison des Métiers :
Comparez les métiers mentionnés en fonction de leurs responsabilités, compétences requises, outils utilisés, et perspectives de carrière. Soulignez les points communs et divergents entre eux.

Conseils Pratiques :
Offrez des conseils sur la recherche d'emploi et la progression de carrière en Belgique, en se concentrant sur les grandes villes comme Bruxelles, Anvers et Liège. Incluez des recommandations pour des stages et des opportunités d'apprentissage si elles sont pertinentes pour le contexte.

Exemples et Témoignages :
Intégrez des exemples de parcours professionnels réussis et des témoignages de professionnels actuels pour illustrer le cheminement dans chaque métier.

Réponse large pour une question abstraite ou générale :

Introduction :
Présentez une vue d'ensemble inspirante des métiers mentionnés, en soulignant leur importance dans le secteur et les opportunités de croissance qu'ils offrent.

Potentiel de Carrière :
Discutez des possibilités d'évolution de carrière dans ces métiers et des compétences transférables qui peuvent être acquises.

Opportunités d'Exploration :
Encouragez les jeunes à explorer les différents aspects des métiers, comme les nouvelles technologies, les tendances émergentes, et les compétences humaines importantes.

Inspirations et Exemples :
Offrez des exemples inspirants de professionnels qui ont réussi dans ces métiers, et mentionnez comment ils ont navigué dans leur carrière.

Conseils Pratiques :
Donnez des conseils sur comment les jeunes peuvent commencer à explorer ces métiers, par exemple en recherchant des stages, en suivant des cours en ligne, ou en se renseignant sur les tendances du secteur.

Le {context2} doit prendre exemple des informations, des datas complètes du {context1}.

L'output doit être parfaitement en corrélation avec la {question}. LE PLUS IMPORTANT est de répondre précisément à la {question}.

REPOND EN FRANCAIS.

"""
combined_prompt = ChatPromptTemplate.from_template(combined_template)

# Define a class to chain the prompt creation, model invocation, and parsing
class PromptToModelChain:
    def __init__(self, prompt, model, parser):
        self.prompt = prompt
        self.model = model
        self.parser = parser

    def invoke(self, input_data):
        required_keys = ["context1", "context2", "question"]
        for key in required_keys:
            if key not in input_data:
                raise KeyError(f"Missing required key '{key}' in input_data.")
        
        formatted_prompt = self.prompt.format(**input_data)
        model_response = self.model.invoke(formatted_prompt)
        print(f"Model Response: {model_response}")
        if hasattr(model_response, 'content'):
            parsed_response = self.parser.invoke(model_response.content)
        else:
            parsed_response = "No content attribute in model response"
        return parsed_response

# Initialize the chain with the prompt, model, and parser
initial_chain = PromptToModelChain(combined_prompt, model, StrOutputParser())

# Create a translation prompt template
translation_template = """
Translate the following text to {language}:

Text: {answer}
"""
translation_prompt = ChatPromptTemplate.from_template(translation_template)



class TranslationChain:
    def __init__(self, prompt, model):
        self.prompt = prompt
        self.model = model

    def invoke(self, input_data):
        required_keys = ["answer", "language"]
        for key in required_keys:
            if key not in input_data:
                raise KeyError(f"Missing required key '{key}' in input_data.")
        
        formatted_prompt = self.prompt.format(**input_data)
        model_response = self.model.invoke(formatted_prompt)
        print(f"Model Response: {model_response}")
        if hasattr(model_response, 'content'):
            return model_response.content
        else:
            return "No content attribute in model response"

# Initialize the translation chain
translation_chain = TranslationChain(translation_prompt, model)

## Modify the CombinedChain class to use the new combined prompt:
class CombinedChain:
    def __init__(self, initial_chain, translation_chain):
        self.initial_chain = initial_chain
        self.translation_chain = translation_chain

    def invoke(self, input_data):
        required_keys = ["context1", "context2", "question"]
        for key in required_keys:
            if key not in input_data:
                raise KeyError(f"Missing required key '{key}' in input_data.")
        
        answer = self.initial_chain.invoke(input_data)
        translation_input = {
            "answer": answer,
            "language": input_data.get("language", "French")
        }
        translated_answer = self.translation_chain.invoke(translation_input)
        return translated_answer

# Initialize the combined chain with the new combined prompt
combined_chain = CombinedChain(initial_chain, translation_chain)



@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Chatbot Interface</title>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {
                    font-family: 'Roboto', sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    min-height: 100vh;
                }
                .container {
                    background: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    width: 90%;
                    max-width: 1200px;
                    margin-top: 20px;
                }
                .card {
                    margin-bottom: 20px;
                }
                .response {
                    white-space: pre-wrap;
                    overflow-y: auto;
                    max-height: 400px;
                }
                .context-box {
                    padding: 10px;
                    background: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    margin-top: 10px;
                }
                .btn-primary, .btn-secondary {
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="row">
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">
                                Chunks Used
                            </div>
                            <div class="card-body response" id="chunks"></div>
                            <button class="btn btn-primary" data-toggle="collapse" data-target="#context1-box">Toggle Context 1</button>
                            <div class="collapse context-box" id="context1-box">
                                <h5>Context 1</h5>
                                <div id="context1-content"></div>
                            </div>
                            <button class="btn btn-primary" data-toggle="collapse" data-target="#context2-box">Toggle Context 2</button>
                            <div class="collapse context-box" id="context2-box">
                                <h5>Context 2</h5>
                                <div id="context2-content"></div>
                            </div>
                            <button class="btn btn-primary" data-toggle="collapse" data-target="#augmented-question-box">Toggle Augmented Question</button>
                            <div class="collapse context-box" id="augmented-question-box">
                                <h5>Augmented Question</h5>
                                <div id="augmented_question"></div>
                            </div>
                            <button class="btn btn-primary" data-toggle="collapse" data-target="#translated-question-box">Toggle Translated Question</button>
                            <div class="collapse context-box" id="translated-question-box">
                                <h5>Translated Question</h5>
                                <div id="translated_question"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">
                                Chat with Our Bot
                            </div>
                            <div class="card-body">
                                <form id="chat-form">
                                    <div class="form-group">
                                        <textarea class="form-control" id="question" name="question" placeholder="Enter your question here" required></textarea>
                                    </div>
                                    <button type="submit" class="btn btn-primary mb-2">Run!</button>
                                    <button type="button" class="btn btn-secondary" id="answer-button">Answer the question</button>
                                </form>
                                <div class="response" id="response"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
            <script>
                let concatenatedChunksFichesMetiers = "";
                let concatenatedChunksJobsJson = "";
                let chunkDetails = [];

                document.getElementById('chat-form').addEventListener('submit', function(event) {
                    event.preventDefault();
                    const question = document.getElementById('question').value;
                    fetch('/generate_chunks', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        const chunksDiv = document.getElementById('chunks');
                        const context1Div = document.getElementById('context1-content');
                        const context2Div = document.getElementById('context2-content');
                        chunksDiv.innerHTML = "<strong>Chunks Used:</strong><br>";
                        context1Div.innerHTML = "";
                        context2Div.innerHTML = "";
                        if (data.details_fiches_metiers.length > 0 && data.details_jobs_json.length > 0) {
                            data.details_fiches_metiers.forEach((detail, index) => {
                                chunksDiv.innerHTML += `Chunk ${detail.index + 1} similarity: ${detail.similarity.toFixed(4)}<br>${detail.preview}<br><br>`;
                            });
                            data.details_jobs_json.forEach((detail, index) => {
                                chunksDiv.innerHTML += `Chunk ${detail.index + 1} similarity: ${detail.similarity.toFixed(4)}<br>${detail.preview}<br><br>`;
                            });
                            concatenatedChunksFichesMetiers = data.chunks_fiches_metiers;
                            concatenatedChunksJobsJson = data.chunks_jobs_json;
                            chunkDetails = data.details_fiches_metiers.concat(data.details_jobs_json);
                            context1Div.textContent = concatenatedChunksFichesMetiers;
                            context2Div.textContent = concatenatedChunksJobsJson;
                            document.getElementById('augmented_question').textContent = data.augmented_question;
                            document.getElementById('translated_question').textContent = data.translated_question;
                        } else {
                            chunksDiv.innerHTML = "No data in json file.";
                        }
                        document.getElementById('question').value = '';
                    })
                    .catch(error => console.error('Error:', error));
                });

                document.getElementById('answer-button').addEventListener('click', function() {
                    const question = document.getElementById('question').value;
                    fetch('/answer_question', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question, context1: concatenatedChunksFichesMetiers, context2: concatenatedChunksJobsJson, chunks: chunkDetails }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        const responseDiv = document.getElementById('response');
                        responseDiv.innerHTML = `<strong>Top 5 Chunks:</strong><br>`;
                        data.top_chunks.forEach((chunk, index) => {
                            responseDiv.innerHTML += `Chunk ${chunk.index + 1} similarity: ${chunk.similarity.toFixed(4)}<br>`;
                        });
                        responseDiv.innerHTML += `<br><strong>Answer:</strong><br>${data.answer}`;
                    })
                    .catch(error => console.error('Error:', error));
                });
            </script>
        </body>
        </html>


        
    ''')

## Modify the /generate_chunks endpoint:
@app.route('/generate_chunks', methods=['POST'])
def generate_chunks():
    data = request.get_json()
    question = data.get('question')

    # Augment the question
    try:
        augmented_question = augmentation_chain.invoke({"question": question})
    except Exception as e:
        return jsonify({"error": str(e)})

    # Translate the augmented question to English
    try:
        translated_question = translation_chain.invoke({
            "answer": augmented_question,
            "language": "English"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

    # Load chunks from the fiches-metiers.json file
    fiches_metiers_path = os.path.expanduser('~/coding/fiches-metiers.json')
    chunks_fiches_metiers = load_chunks_from_json(fiches_metiers_path)

    # Load chunks from the jobs.json file
    jobs_json_path = os.path.expanduser('~/coding/jobs.json')
    chunks_jobs_json = load_chunks_from_jobs_json(jobs_json_path)

    if not chunks_fiches_metiers or not chunks_jobs_json:
        return jsonify({"chunks": "", "details": []})

    # Find relevant chunks for both contexts
    concatenated_chunks_fiches_metiers, top_indices_fiches_metiers, similarities_fiches_metiers = find_relevant_chunks(augmented_question, chunks_fiches_metiers, top_n=15)
    concatenated_chunks_jobs_json, top_indices_jobs_json, similarities_jobs_json = find_relevant_chunks(translated_question, chunks_jobs_json, top_n=15)

    details_fiches_metiers = [{"index": int(idx), "similarity": float(similarities_fiches_metiers[idx]), "preview": chunks_fiches_metiers[idx][:100]} for idx in top_indices_fiches_metiers]
    details_jobs_json = [{"index": int(idx), "similarity": float(similarities_jobs_json[idx]), "preview": chunks_jobs_json[idx][:100]} for idx in top_indices_jobs_json]

    return jsonify({
        "chunks_fiches_metiers": concatenated_chunks_fiches_metiers,
        "details_fiches_metiers": details_fiches_metiers,
        "chunks_jobs_json": concatenated_chunks_jobs_json,
        "details_jobs_json": details_jobs_json,
        "augmented_question": augmented_question,
        "translated_question": translated_question
    })


@app.route('/answer_question', methods=['POST'])
def answer_question():
    data = request.get_json()
    question = data.get('question')
    context1 = data.get('context1')
    context2 = data.get('context2')
    chunk_details = data.get('chunks')

    # Augment the question
    try:
        augmented_question = augmentation_chain.invoke({"question": question})
    except Exception as e:
        return jsonify({"error": str(e)})

    # Translate the augmented question to English
    try:
        translated_question = translation_chain.invoke({
            "answer": augmented_question,
            "language": "English"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

    # Find the top 15 chunks for answering the question
    chunk_details.sort(key=lambda x: x['similarity'], reverse=True)
    top_chunks = chunk_details[:15]

    # Prepare the input data for the chain with the augmented question
    input_data = {
        "context1": context1,
        "context2": context2,
        "question": augmented_question  # Use the augmented question here
    }

    try:
        parsed_output = combined_chain.invoke(input_data)
        answer = parsed_output
    except Exception as e:
        answer = str(e)

    return jsonify({"answer": answer, "top_chunks": top_chunks})


if __name__ == '__main__':
    app.run(debug=True)