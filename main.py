import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import process

# --- Load and preprocess data ---
data = pd.read_csv("books_data.csv")
data = data.dropna(subset=['title', 'authors'], how='all')
data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce')
data['book_content'] = data['title'].fillna('') + ' ' + data['authors'].fillna('')

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['book_content'])

# --- Search and Recommendation Functions ---

def multi_field_search(query, data):
    mask = (
        data['title'].str.lower().str.contains(query.lower()) |
        data['authors'].str.lower().str.contains(query.lower())
    )
    return data[mask]

def fuzzy_search(query, data, limit=3):
    choices = (data['title'] + ' ' + data['authors']).tolist()
    results = process.extract(query, choices, limit=limit, score_cutoff=60)
    matched_indices = [idx for _, _, idx in results]
    return data.iloc[matched_indices]

def find_best_match(query, data):
    matches = multi_field_search(query, data)
    if not matches.empty:
        return matches
    matches = fuzzy_search(query, data)
    return matches

def get_recommendations(book_query, tfidf_matrix=tfidf_matrix, data=data):
    matches = find_best_match(book_query, data)
    if matches.empty:
        return []
    idx = matches.index[0]
    query_vec = tfidf_matrix[idx]
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    recommendations = data.iloc[book_indices][['title', 'authors', 'average_rating']]
    return recommendations.to_dict(orient='records'), matches.iloc[0]['title']

# --- FastAPI App ---

app = FastAPI(title="Book Recommendation API")

class BookRecommendationResponse(BaseModel):
    matched_title: str
    recommendations: List[dict]

@app.get("/recommend", response_model=BookRecommendationResponse)
def recommend_books(q: str = Query(..., description="Book title or author to search")):
    recs, matched_title = get_recommendations(q)
    if not recs:
        return {"matched_title": "", "recommendations": []}
    return {"matched_title": matched_title, "recommendations": recs}
