# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from index.search_engine import hybrid_search

app = FastAPI(title="SHL Assessment Recommender", version="1.0")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    w_semantic: float = 0.7

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    try:
        results = hybrid_search(req.query, top_k=req.top_k, w_semantic=req.w_semantic)
        return {"query": req.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "SHL Recommender API is running. Use /docs or POST /recommend."}
