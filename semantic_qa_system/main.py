import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.routes:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,  # Set to False in production
        workers=4  # Adjust based on your needs
    )