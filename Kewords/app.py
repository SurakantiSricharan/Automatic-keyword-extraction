#cd Kewords
#python app.py
import pickle
import os
from flask import Flask, render_template, request
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_pickle(filename):
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)


try:
    cv = load_pickle('count_vectorizer.pkl')
    tfidf_transformer = load_pickle('tfidf_transformer.pkl')
    feature_names = load_pickle('feature_names.pkl')
except FileNotFoundError as e:
    print(e)
    exit(1)  


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
new_stop_words = ["fig", "figure", "image", "sample", "using",
                  "show", "result", "large", "also", "one", "two",
                  "three", "four", "five", "seven", "eight", "nine"]
stop_words = list(stop_words.union(new_stop_words))

def preprocess_text(txt):
   
    txt = txt.lower()
    
    txt = re.sub(r"<.*?>", " ", txt)
    
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    
    txt = nltk.word_tokenize(txt)
    
    txt = [word for word in txt if word not in stop_words]
    
    txt = [word for word in txt if len(word) >= 3]
    
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]

    return " ".join(txt)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    results = {feature_names[idx]: round(score, 3) for idx, score in sorted_items}
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    document = request.files['file']
    if not document.filename:
        return render_template('index.html', error='No document selected')

    text = document.read().decode('utf-8', errors='ignore')
    preprocessed_text = preprocess_text(text)
    tf_idf_vector = tfidf_transformer.transform(cv.transform([preprocessed_text]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, 20)

    return render_template('keywords.html', keywords=keywords)

@app.route('/search_keywords', methods=['POST'])
def search_keywords():
    search_query = request.form['search']
    if search_query:
        keywords = [keyword for keyword in feature_names if search_query.lower() in keyword.lower()]
        return render_template('keywordslist.html', keywords=keywords[:20])  # Limit to 20 keywords

    return render_template('index.html')

if __name__ == '__main__':
    print("Current Directory:", os.getcwd())  
    app.run(debug=True)