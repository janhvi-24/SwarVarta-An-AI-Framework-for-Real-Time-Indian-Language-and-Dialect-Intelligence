import uvicorn
from gen_app import app  # Import FastAPI app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
