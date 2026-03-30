from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from rag import LangChain

app = FastAPI()
lang_chain = LangChain()

@app.post("/query")
async def query_api(file:UploadFile = File(...), query:str = Form(...)) :

    if not file :
        raise HTTPException(status_code=400, detail ="File is required")
    
    if not query :
        raise HTTPException(status_code=400, detail ="Query is required")
    
    vector_store = lang_chain.ingest_document(file, query)

    if type(vector_store) == str and 'Answer:' in vector_store :
        vector_store = vector_store.replace("Answer:", "").strip()

    result = {
        "query" : query,
        "answer" : vector_store
    }

    return result