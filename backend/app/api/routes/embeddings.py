from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional,List
from app.services.embedding_service import embed_text, embed_documents
from app.core.config import settings
router=APIRouter(prefix="/embeddings",tags=["embeddings"])

class EmbeddingTestResponse(BaseModel):
    text:str
    embedding:List[float]
    embedding_dimensions:int
    vector_length:int

@router.post("/test",response_model=EmbeddingTestResponse)
def test_embedding(
    text:str=Query(..., description="Text to embed")
    ):  
    try:
        embedding=embed_text(text)
        return EmbeddingTestResponse(
            text=text,
            embedding=embedding,
            embedding_dimensions=len(embedding),
            vector_length=len(embedding)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

class EmbeddingBatchResponse(BaseModel):
    texts:List[str]
    embeddings:List[List[float]]
    total_text:int
    total_embeddings:int
    dimensions:Optional[int]

@router.post("/test-batch",response_model=EmbeddingBatchResponse)
def test_embedding_batch(
    request:List[str]
    ):
    try:
        embeddings=embed_documents(request)
        return EmbeddingBatchResponse(
            texts=request,
            embeddings=embeddings,
            total_text=len(request),
            total_embeddings=len(embeddings),
            dimensions=len(embeddings[0]) if embeddings else None
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )