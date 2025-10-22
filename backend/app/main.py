"""
Entry point for FastAPI backend
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api import search
from app.config import settings
# from app.utils.logger import StandardLogger
#from app.db.database import create_tables, check_database_connection

# Setup logging
# StandardLogger()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI Video Search Backend - Multi-modal Vietnamese video search system"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static data directory (e.g., frames/images) at /static
app.mount("/static", StaticFiles(directory="data"), name="static")

# Initialize database
# @app.on_event("startup")
# async def startup_event():
#     """Initialize database and other services on startup"""
#     logger.info("Initializing application...")
    
#     # Create database tables
#     try:
#         create_tables()
#         logger.info("Database tables created successfully")
#     except Exception as e:
#         logger.error(f"Failed to create database tables: {str(e)}")
    
#     # Check database connection
#     if check_database_connection():
#         logger.info("Database connection verified")
#     else:
#         logger.warning("Database connection failed")

# Include routers
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
#app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
#app.include_router(health.router, prefix="/api/v1/health", tags=["health"])


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "AI Video Search Backend",
        "version": settings.VERSION,
        "status": "running"
    }


@app.get("/api/v1")
def api_info():
    """API information endpoint"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "Multi-modal Vietnamese video search system",
        "endpoints": {
            "search": "/api/v1/search",
            "ingest": "/api/v1/ingest",
            "health": "/api/v1/health"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
