"""
FastAPI backend for Romanian LLM Fine-tuning Interface
Provides REST API endpoints for training, testing, and managing models
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import yaml

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.config_validator import validate_config
from scripts.prepare_data import RomanianDataProcessor
from scripts.test_model import ModelTester
from scripts.evaluate import RomanianModelEvaluator

app = FastAPI(
    title="Romanian LLM Training API",
    description="API for fine-tuning and testing Romanian language models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking jobs
training_jobs = {}
evaluation_jobs = {}

# Pydantic models
class TrainingConfig(BaseModel):
    model_name: str = Field(default="meta-llama/Llama-3.1-8B")
    max_steps: int = Field(default=1000, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, le=1)
    warmup_steps: int = Field(default=100, ge=0)
    output_dir: str = Field(default="./checkpoints")
    dataset_path: str = Field(...)

class TrainingJob(BaseModel):
    job_id: str
    status: str
    config: TrainingConfig
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TestPrompt(BaseModel):
    prompt: str
    max_length: int = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)

class TestResponse(BaseModel):
    prompt: str
    response: str
    generation_time: float

class DatasetInfo(BaseModel):
    name: str
    path: str
    num_examples: int
    size_mb: float
    created_at: str

class EvaluationRequest(BaseModel):
    checkpoint_path: str
    test_suite: str = "default"

class EvaluationResult(BaseModel):
    job_id: str
    status: str
    metrics: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None


# Helper functions
def get_project_root():
    return Path(__file__).parent.parent.parent

def get_data_dir():
    return get_project_root() / "data"

def get_checkpoints_dir():
    return get_project_root() / "checkpoints"

def get_config_path():
    return get_project_root() / "configs" / "hyperparams.yaml"


# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Romanian LLM Training API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "training": "/api/training",
            "datasets": "/api/datasets",
            "testing": "/api/test",
            "evaluation": "/api/evaluate",
            "checkpoints": "/api/checkpoints"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "data_dir": str(get_data_dir().exists()),
            "checkpoints_dir": str(get_checkpoints_dir().exists())
        }
    }

# Training endpoints
@app.post("/api/training/start", response_model=TrainingJob)
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start a new training job"""
    job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    job = TrainingJob(
        job_id=job_id,
        status="queued",
        config=config,
        created_at=datetime.now().isoformat()
    )

    training_jobs[job_id] = job

    # In a real implementation, this would start the training in the background
    # For now, we'll just simulate it
    background_tasks.add_task(run_training_job, job_id)

    return job

async def run_training_job(job_id: str):
    """Background task to run training"""
    job = training_jobs[job_id]
    job.status = "running"
    job.started_at = datetime.now().isoformat()

    try:
        # This would call the actual training script
        # For now, simulate it
        await asyncio.sleep(2)

        job.status = "completed"
        job.completed_at = datetime.now().isoformat()
        job.metrics = {
            "final_loss": 2.456,
            "steps_completed": job.config.max_steps
        }
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now().isoformat()

@app.get("/api/training/jobs", response_model=List[TrainingJob])
async def list_training_jobs():
    """List all training jobs"""
    return list(training_jobs.values())

@app.get("/api/training/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str):
    """Get specific training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]

@app.delete("/api/training/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]
    if job.status == "running":
        job.status = "cancelled"
        job.completed_at = datetime.now().isoformat()

    return {"message": f"Job {job_id} cancelled"}

# Dataset endpoints
@app.get("/api/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List available datasets"""
    datasets = []
    data_dir = get_data_dir()

    if not data_dir.exists():
        return datasets

    for item in data_dir.iterdir():
        if item.is_file() and item.suffix in ['.jsonl', '.json']:
            stat = item.stat()
            datasets.append(DatasetInfo(
                name=item.name,
                path=str(item),
                num_examples=count_jsonl_lines(item) if item.suffix == '.jsonl' else 0,
                size_mb=round(stat.st_size / (1024 * 1024), 2),
                created_at=datetime.fromtimestamp(stat.st_ctime).isoformat()
            ))

    return datasets

def count_jsonl_lines(filepath: Path) -> int:
    """Count lines in JSONL file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except:
        return 0

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset"""
    if not file.filename.endswith(('.jsonl', '.json')):
        raise HTTPException(status_code=400, detail="Only JSONL and JSON files are supported")

    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / file.filename

    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    return {
        "message": "Dataset uploaded successfully",
        "filename": file.filename,
        "path": str(file_path),
        "size_mb": round(len(content) / (1024 * 1024), 2)
    }

@app.get("/api/datasets/{dataset_name}")
async def get_dataset(dataset_name: str, limit: int = 10):
    """Get dataset preview"""
    data_dir = get_data_dir()
    file_path = data_dir / dataset_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    examples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                examples.append(json.loads(line))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

    return {
        "dataset": dataset_name,
        "preview": examples,
        "total_shown": len(examples)
    }

# Testing endpoints
@app.post("/api/test/prompt", response_model=TestResponse)
async def test_prompt(request: TestPrompt):
    """Test the model with a prompt"""
    # This would use the actual model tester
    # For now, simulate a response
    import time
    start_time = time.time()

    # Simulated response
    response_text = f"Aceasta este un răspuns simulat pentru: {request.prompt}"

    generation_time = time.time() - start_time

    return TestResponse(
        prompt=request.prompt,
        response=response_text,
        generation_time=generation_time
    )

@app.get("/api/test/examples")
async def get_test_examples():
    """Get predefined test examples"""
    return {
        "examples": [
            "Care este capitala României?",
            "Explică-mi ce este inteligența artificială.",
            "Scrie o poezie despre munții Carpați.",
            "Enumeră cele mai mari orașe din România.",
            "Rezumă istoria României într-un paragraf."
        ]
    }

# Evaluation endpoints
@app.post("/api/evaluate/start", response_model=EvaluationResult)
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start model evaluation"""
    job_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    result = EvaluationResult(
        job_id=job_id,
        status="running",
        created_at=datetime.now().isoformat()
    )

    evaluation_jobs[job_id] = result
    background_tasks.add_task(run_evaluation_job, job_id, request)

    return result

async def run_evaluation_job(job_id: str, request: EvaluationRequest):
    """Background task to run evaluation"""
    result = evaluation_jobs[job_id]

    try:
        # Simulate evaluation
        await asyncio.sleep(2)

        result.status = "completed"
        result.completed_at = datetime.now().isoformat()
        result.metrics = {
            "accuracy": 0.85,
            "fluency": 0.92,
            "instruction_following": 0.88
        }
    except Exception as e:
        result.status = "failed"
        result.metrics = {"error": str(e)}

@app.get("/api/evaluate/jobs/{job_id}", response_model=EvaluationResult)
async def get_evaluation_job(job_id: str):
    """Get evaluation job status"""
    if job_id not in evaluation_jobs:
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    return evaluation_jobs[job_id]

# Checkpoint endpoints
@app.get("/api/checkpoints")
async def list_checkpoints():
    """List available checkpoints"""
    checkpoints_dir = get_checkpoints_dir()

    if not checkpoints_dir.exists():
        return {"checkpoints": []}

    checkpoints = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir():
            stat = item.stat()
            checkpoints.append({
                "name": item.name,
                "path": str(item),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "size_mb": get_dir_size(item)
            })

    return {"checkpoints": checkpoints}

def get_dir_size(path: Path) -> float:
    """Get directory size in MB"""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except:
        pass
    return round(total / (1024 * 1024), 2)

@app.get("/api/config")
async def get_config():
    """Get current training configuration"""
    config_path = get_config_path()

    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config file not found")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

@app.put("/api/config")
async def update_config(config: Dict[str, Any]):
    """Update training configuration"""
    config_path = get_config_path()

    # Validate config
    try:
        validate_config(config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return {"message": "Configuration updated successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
