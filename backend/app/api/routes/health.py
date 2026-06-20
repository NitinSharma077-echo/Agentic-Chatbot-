from fastapi import APIRouter
from app.core.config import settings

router=APIRouter(prefix='/health',tags=['health'])

@router.get("/")
def health_check():
    return{
        "status":"ok",
        "app_name":settings.APP_NAME,
        "version":settings.APP_VERSION,
        "environment":settings.APP_ENV
    }
