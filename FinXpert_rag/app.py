from flask import Flask, session, request, render_template
import sqlite3
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = os.urandom(24)  # Securely generate a key or use a static one

# Initialize OpenAI API key
openai.api_key = 'your_api_key'

# Load vectorizer and database
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)  # Load the vectorizer object

db_name = 'privacy_policies.db'
table_name = 'documents'

def vectorize_query(query, vectorizer):
    query_vector = vectorizer.transform([query])
    return query_vector

def retrieve_documents(query, vectorizer, db_name, table_name, top_n=3):
    query_vector = vectorize_query(query, vectorizer)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, text, vector FROM {table_name}")
    rows = cursor.fetchall()

    doc_scores = []
    for row in rows:
        doc_id, text, vector_blob = row
        doc_vector = pickle.loads(vector_blob)
        similarity = cosine_similarity(query_vector, doc_vector)
        doc_scores.append((similarity, text))

    doc_scores.sort(reverse=True, key=lambda x: x[0])
    top_documents = [doc[1] for doc in doc_scores[:top_n]]
    conn.close()
    return top_documents

def generate_response_with_gpt3(query, documents, history):
    prompt = "You are a helpful assistant.\n\n"
    if history:
        for entry in history:
            prompt += f"Previous Query: {entry['query']}\nPrevious Response: {entry['response']}\n\n"
    prompt += f"Current Query: {query}\n"
    for i, doc in enumerate(documents, 1):
        prompt += f"Context {i}: {doc}\n"
    prompt += "Response:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7
    )

    answer = response['choices'][0]['message']['content'].strip()
    return answer

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []  # Initialize history if it doesn't exist

    if request.method == 'POST':
        query = request.form['query']
        top_documents = retrieve_documents(query, vectorizer, db_name, table_name)
        response = generate_response_with_gpt3(query, top_documents, session['history'])
        
        # Store the new query and response in the session history
        session['history'].append({'query': query, 'response': response})
        # Limit the history to a reasonable number of entries (e.g., 10)
        if len(session['history']) > 10:
            session['history'].pop(0)
        
        return render_template('index.html', query=query, response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
