from fastapi import FastAPI
from visual import search_text

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "CCTV API Working"
    }

@app.get("/search")
def search(query: str):

    results = search_text(query)
    return {
        "query": query,
        "matches_found": len(results)
    }