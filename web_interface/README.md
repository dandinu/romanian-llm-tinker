# Romanian LLM Web Interface

A modern web interface for fine-tuning Llama 3.1 8B for Romanian language using the Tinker framework.

## Features

- **Dashboard** - Monitor training jobs, datasets, and system status
- **Training** - Configure and start fine-tuning jobs with custom parameters
- **Datasets** - Upload and manage JSONL training datasets
- **Testing** - Interactive chat interface to test your fine-tuned models
- **Settings** - View and modify training configurations

## Tech Stack

### Frontend
- **React 18** - Modern UI library
- **Tailwind CSS 4.1.13** - Utility-first CSS framework
- **shadcn/ui** - High-quality UI components
- **Radix UI** - Accessible component primitives
- **Recharts** - Data visualization
- **Lucide Icons** - Beautiful icon set
- **Vite** - Fast build tool

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **PyYAML** - Configuration management

### Deployment
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **Nginx** - Web server for frontend

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- `.env` file configured (copy from `.env.example` in project root)

### Deploy with Docker

1. **Navigate to web interface directory:**
   ```bash
   cd web_interface
   ```

2. **Build and start containers:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop containers:**
   ```bash
   docker-compose down
   ```

### Development Mode

#### Backend Development

```bash
cd web_interface/backend

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Development

```bash
cd web_interface/frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Run development server
npm run dev
```

## Architecture

```
web_interface/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile          # Backend container config
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── ui/        # Reusable UI components
│   │   │   └── Layout.jsx # App layout
│   │   ├── pages/         # Page components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Training.jsx
│   │   │   ├── Datasets.jsx
│   │   │   ├── Testing.jsx
│   │   │   └── Settings.jsx
│   │   ├── lib/
│   │   │   ├── api.js     # API client
│   │   │   └── utils.js   # Utility functions
│   │   ├── App.jsx        # Main app component
│   │   ├── main.jsx       # Entry point
│   │   └── index.css      # Global styles
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│   ├── vite.config.js
│   ├── nginx.conf         # Nginx configuration
│   └── Dockerfile        # Frontend container config
└── docker-compose.yml    # Container orchestration
```

## API Endpoints

### Training
- `POST /api/training/start` - Start a new training job
- `GET /api/training/jobs` - List all training jobs
- `GET /api/training/jobs/{job_id}` - Get specific job status
- `DELETE /api/training/jobs/{job_id}` - Cancel a training job

### Datasets
- `GET /api/datasets` - List available datasets
- `POST /api/datasets/upload` - Upload a new dataset
- `GET /api/datasets/{name}` - Preview dataset examples

### Testing
- `POST /api/test/prompt` - Test model with a prompt
- `GET /api/test/examples` - Get example prompts

### Evaluation
- `POST /api/evaluate/start` - Start model evaluation
- `GET /api/evaluate/jobs/{job_id}` - Get evaluation results

### Configuration
- `GET /api/config` - Get current configuration
- `PUT /api/config` - Update configuration

### System
- `GET /health` - Health check endpoint
- `GET /api/checkpoints` - List saved model checkpoints

## Dataset Format

Upload JSONL files with the following structure:

```json
{
  "messages": [
    {"role": "user", "content": "Care este capitala României?"},
    {"role": "assistant", "content": "Capitala României este București, cel mai mare oraș din țară..."}
  ]
}
```

Each line should be a valid JSON object containing a conversation.

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Tinker API Configuration
TINKER_API_KEY=your_tinker_api_key_here

# HuggingFace Configuration
HF_TOKEN=your_huggingface_token_here

# Optional: Weights & Biases
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=romanian-llm
```

### Frontend Environment

Create `.env` in `frontend/` directory:

```env
VITE_API_URL=http://localhost:8000
```

## Design System

The interface follows a modern enterprise SaaS design language:

- **Color Palette:**
  - Primary: Blue (#0B99FF)
  - Secondary: Orange (#f97316)
  - Success: Emerald Green
  - Destructive: Red (#F45757)
  - Warning: Amber

- **Typography:**
  - Font: Inter (Google Font)
  - Sizes: 12px - 36px modular scale

- **Layout:**
  - Sidebar: 256px (collapsible to 48px)
  - Border Radius: 10px default
  - Base spacing: 4px increments

## Production Deployment

### Using Docker Compose (Recommended)

```bash
# Build production images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Manual Deployment

1. **Build frontend:**
   ```bash
   cd frontend
   npm run build
   # Serve dist/ folder with nginx or similar
   ```

2. **Deploy backend:**
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Production Considerations

- Set appropriate CORS origins in `backend/main.py`
- Use environment-specific `.env` files
- Enable HTTPS with reverse proxy (nginx/traefik)
- Configure proper logging and monitoring
- Set up backup strategies for data and checkpoints
- Use production-grade database for job tracking (currently in-memory)

## Troubleshooting

### Backend issues

```bash
# Check backend logs
docker-compose logs backend

# Restart backend
docker-compose restart backend

# Access backend shell
docker-compose exec backend bash
```

### Frontend issues

```bash
# Check frontend logs
docker-compose logs frontend

# Rebuild frontend
docker-compose up -d --build frontend

# Access frontend container
docker-compose exec frontend sh
```

### Port conflicts

If ports 3000 or 8000 are in use, modify `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8080:8000"  # Change host port
  frontend:
    ports:
      - "3001:3000"  # Change host port
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

Same as the main project.

## Support

For issues and questions, please open an issue in the main repository.
