import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from auto_ai.app.config import settings
from auto_ai.app.infra.db import init_db
from auto_ai.app.api.projects import router as projects_router
from auto_ai.app.api.settings import router as settings_router

# 1. Initialize FastAPI Application
app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-Agent AI Research Platform for Tabular Datasets",
    version="1.0.0"
)

# 2. Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Register Routers
app.include_router(projects_router, prefix="/api")
app.include_router(settings_router, prefix="/api")

# 4. Initialize Database on Startup
@app.on_event("startup")
def startup_event():
    init_db()

# 5. Serve SPA Frontend
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
runs_path = settings.DATA_DIR / "runs"

# Ensure directories exist
frontend_path.mkdir(parents=True, exist_ok=True)
(frontend_path / "css").mkdir(exist_ok=True)
(frontend_path / "js").mkdir(exist_ok=True)
runs_path.mkdir(parents=True, exist_ok=True)

# Mount asset folders statically
app.mount("/css", StaticFiles(directory=str(frontend_path / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(frontend_path / "js")), name="js")
app.mount("/runs", StaticFiles(directory=str(runs_path)), name="runs")

@app.get("/")
def read_root():
    """Serve the single-page application dashboard."""
    index_file = frontend_path / "index.html"
    if not index_file.exists():
        # Create a basic placeholder just in case, though we will overwrite it next
        with open(index_file, "w") as f:
            f.write("<h1>ModelSmith AI Dashboard Loading...</h1>")
    return FileResponse(str(index_file))

if __name__ == "__main__":
    # Standard development server run
    uvicorn.run("auto_ai.app.main:app", host="0.0.0.0", port=8000, reload=True)
