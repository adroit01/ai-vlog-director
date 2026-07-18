import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.db import init_db
from api.routes import router

logger = logging.getLogger(__name__)

def setup_logging():
    import logging
    from logging.handlers import RotatingFileHandler
    
    # Create file handler
    file_handler = RotatingFileHandler("server.log", maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Add file handler to root logger
    if file_handler not in root_logger.handlers:
        root_logger.addHandler(file_handler)
        
    # Add file handler to Uvicorn loggers
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        log = logging.getLogger(logger_name)
        # Prevent duplicate handlers on reload
        has_file_handler = any(isinstance(h, RotatingFileHandler) and h.baseFilename.endswith("server.log") for h in log.handlers)
        if not has_file_handler:
            log.addHandler(file_handler)

# Configure logging at startup
setup_logging()

app = FastAPI(
    title="GenAI Travel Vlog Director API",
    description="Backend API services for automating travel vlog visual logs, research, voice generation, and editing.",
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, configure to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(router)

@app.on_event("startup")
def on_startup():
    import os
    if not os.getenv("TESTING"):
        logger.info("Initializing database SQLite tables...")
        init_db()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "GenAI Travel Vlog Director API",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
