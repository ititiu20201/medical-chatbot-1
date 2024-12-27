# Medical Consultation Chatbot

A Vietnamese medical consultation chatbot using NLP for specialty prediction and queue management. The system uses PhoBERT for natural language processing and provides medical specialty recommendations based on patient symptoms.

## Features

- Vietnamese language medical consultation
- Real-time chat interface
- Medical specialty prediction
- Automatic medical record generation
- Queue management system
- Patient information collection
- Symptom analysis
- Treatment recommendations

## System Requirements

### Development Environment

- Python 3.8+
- Node.js 14+
- Redis 6.0+
- Docker 20.10+

### GPU Requirements

- Minimum: 8GB GPU Memory
- Recommended: NVIDIA A100 (40GB/80GB)
- CUDA support required

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ititiu20201/medical-chatbot.git
cd medical-chatbot
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Setup directory structure:

```bash
mkdir -p data/{raw,processed,models}
mkdir -p logs
```

5. Configuration:

```bash
cp configs/config.example.json configs/config.json
# Edit configs/config.json with your settings
```

## Data Preparation

1. Place your data files in data/raw/:

- alpaca_data.json
- chatdoctor5k.json
- disease_database_mini.csv
- disease_symptom.csv
- format_dataset.csv
- other data files...

2. Process the data:

```bash
python src/data/run_pipeline.py
```

## Model Training

1. Fine-tune PhoBERT model:

```bash
python src/training/run_training.py
```

Training parameters can be configured in configs/config.json.

## Running the Application

### Using Docker (Recommended)

1. Build and start containers:

```bash
docker-compose up -d
```

2. Check status:

```bash
docker-compose ps
```

### Manual Setup

1. Start the backend:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

2. Access the API:

- API Documentation: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws

## Project Structure

```
medical-chatbot/
├── src/
│   ├── api/            # FastAPI backend
│   ├── models/         # ML models
│   ├── data/           # Data processing
│   └── training/       # Training scripts
├── data/
│   ├── raw/           # Raw data files
│   ├── processed/     # Processed data
│   └── models/        # Trained models
├── configs/           # Configuration files
├── tests/            # Test files
├── docs/             # Documentation
└── docker/           # Docker files
```

## Testing

Run the test suite:

```bash
python tests/run_tests.py
```

## Documentation

- System Documentation: docs/system_documentation.md
- API Documentation: docs/api_documentation.md
- User Manual: docs/user_manual.md
- Installation & Maintenance: docs/installation_maintenance.md

## Monitoring

Access monitoring dashboards:

- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

## Contact

- Email: dqhien.ityu@example.com
- Issue Tracker: https://github.com/ititiu20201/medical-chatbot/issues

## Acknowledgments

- VINAI for PhoBERT model
