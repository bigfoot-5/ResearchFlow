import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.hubspot_etl import load_to_db

# Add project root to Python path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Potentially import routers later
# from src.api.endpoints import data_endpoints, insights_endpoints # Example
from src.api.endpoints import data_endpoints # Import the data_endpoints router
from src.api.endpoints import agent_endpoints # Import the agent_endpoints router

# Potentially import database initialization functions
from src.database.database_setup import create_db_and_tables

from src.config import settings
from src.database.hubspot_etl import load_to_db

# Create FastAPI app
app = FastAPI(
    title="Cognition Engine API",
    description="API for the HubSpot CRM Cognition Engine for ICP segmentation insights",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Create database tables if they don't exist on startup
    """
    create_db_and_tables()
    # load_to_db()

@app.get("/")
async def root():
    """
    Root endpoint to check API status
    """
    return {
        "status": "online",
        "api_version": app.version,
        "api_title": app.title,
    }

# Include routers for different sets of endpoints
app.include_router(data_endpoints.router, prefix="/api/v1/data", tags=["Data Access"])

# Example: 
# app.include_router(insights_endpoints.router, prefix="/api/v1/insights", tags=["Insights"])
# app.include_router(insights_endpoints.router, prefix="/api/v1/segments", tags=["Segments"]) # For /segments/dead-zones etc.

# Import and include API routers
from src.api.endpoints.icp_endpoints import router as icp_router
app.include_router(icp_router, prefix="/api/v1/icp")

# Include agent endpoints router
app.include_router(agent_endpoints.router, prefix="/api/v1", tags=["Agents"])

# Include any other API routers here
# from src.api.endpoints.data_endpoints import router as data_router
# app.include_router(data_router)

if __name__ == "__main__":
    import uvicorn
    # Initialize database before starting the server
    create_db_and_tables()
    load_to_db()
    # For development only. In production, use a proper ASGI server like Gunicorn with Uvicorn workers
    uvicorn.run(
        "src.main:app", 
        host="0.0.0.0", 
        port=8005,  # Using port 8005 to avoid conflicts
        reload=True
    ) 