# RAG-Jobs
 Talk to a chatbot and get market job information from 2 different sources.

### Career Guidance Chatbot
This project is a career guidance chatbot developed using Flask and natural language processing (NLP) models provided by OpenAI. The chatbot is able to reformulate and enrich career guidance questions, find relevant information in JSON documents, and provide detailed and useful answers for users.

### Features
1. Question processing: Uses NLP templates to reformulate and enrich career guidance questions.
2. Document search: Loads and splits JSON documents into chunks for easy searching.
3. Similarity analysis: Uses TF-IDF and cosine similarity to find the chunks most relevant to the question.
4. Answer generation: Generates detailed, structured answers based on the information found.

### Prerequisites
Python 3.6+
Flask
OpenAI API Key (not displayed)

# Detailed features
### Document processing and retrieval
The load_chunks_from_json and load_chunks_from_jobs_json functions read and divide JSON documents into chunks to facilitate searching. 
The find_relevant_chunks function uses TF-IDF and cosine similarity to find the most relevant chunks based on the question.

### Question augmentation and translation
The chatbot uses the OpenAI GPT model to reformulate and enrich questions. The AugmentationChain class manages this process. 
In addition, a translation chain (TranslationChain) is used to translate questions and answers if necessary.

### Answer generation
The chatbot generates detailed, structured answers using the contexts provided. The CombinedChain class combines initial answer generation and translation to provide complete answers.

### Endpoints API
- / : Displays the user interface.
- /generate_chunks (POST): Generates relevant chunks from JSON documents for a given question.
- /answer_question (POST): Answers a question using the generated chunks and the GPT template.

### Example workflow
1. The user submits a question via the user interface form.
2. The backend uses augmentation_chain to reformulate the question.
3. JSON documents are loaded and divided into chunks.
4. The most relevant chunks are found using cosine similarity.
5. A detailed answer is generated using the contexts of the chunks found.

### User interface
The user interface is built with HTML, CSS, and Bootstrap for a clean, responsive presentation. 
It includes features such as buttons to toggle context sections and dynamic displays of chunks and answers.

### Contribute
Contributions are welcome! Please submit pull requests or open issues to discuss changes you'd like to make.

## THANKS
Thank you to public resources such as ESCO, and Betterteam.
MANY thanks to Interskillar allowing me to explore technical use-cases.
