# AI Server - Question Processing System

Server AI yang memproses pertanyaan menggunakan Brain.js untuk analisis dataset, LangChain untuk template, dan Ollama untuk pemrosesan AI.

## Flowchart Sistem

```
GET Question → Get Context from Dataset (Brain.js) → Tag Question → Process with LangChain + Ollama → Send Response
```

## Fitur Utama

- ✅ **Asynchronous Processing**: Pertanyaan diproses secara asinkron dengan tracking status
- ✅ **Brain.js Integration**: Neural network untuk analisis relevansi dari dataset
- ✅ **LangChain Templates**: Template yang fleksibel untuk prompt engineering
- ✅ **Ollama Integration**: Local LLM processing dengan model yang dapat dikonfigurasi
- ✅ **Question Tagging**: Otomatis menentukan tag berdasarkan konteks
- ✅ **Status Tracking**: Real-time monitoring proses pertanyaan
- ✅ **Comprehensive Logging**: Logging lengkap untuk debugging dan monitoring

## Struktur Folder

```
servers_ai-toOrder/
├── src/
│   ├── server.js                 # Main server file
│   ├── routes/
│   │   └── questionRoutes.js     # API routes untuk pertanyaan
│   ├── services/
│   │   └── QuestionProcessor.js  # Core logic Brain.js + LangChain + Ollama
│   ├── middleware/
│   │   ├── validateQuestion.js   # Validasi input pertanyaan
│   │   └── errorHandler.js       # Global error handling
│   └── utils/
│       └── logger.js             # Winston logging utility
├── data/
│   └── dataset.json              # Dataset untuk training Brain.js
├── logs/                         # Log files (auto-generated)
├── package.json                  # Dependencies dan scripts
├── .env.example                  # Environment variables template
└── README.md                     # Dokumentasi ini
```

## Prerequisites

1. **Node.js** (v16 atau lebih tinggi)
2. **Ollama** terinstall dan berjalan di sistem
3. **Model Ollama** (contoh: llama2, mistral, dll)

### Install Ollama

```bash
# Windows/Mac/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model (contoh: llama2)
ollama pull llama2

# Start Ollama service
ollama serve
```

## Installation

1. **Clone atau copy project ini**

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Setup environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` sesuai kebutuhan:
   ```env
   PORT=3000
   NODE_ENV=development
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2
   DATASET_PATH=./data/dataset.json
   LOG_LEVEL=info
   ```

4. **Start server**
   ```bash
   # Development mode
   npm run dev
   
   # Production mode
   npm start
   ```

## API Endpoints

### 1. Submit Question

**POST** `/api/questions/ask`

```json
{
  "question": "What is artificial intelligence?",
  "context": {
    "user_id": "123",
    "session_id": "abc"
  }
}
```

**Response:**
```json
{
  "success": true,
  "questionId": "uuid-here",
  "message": "Question received and being processed",
  "statusUrl": "/api/questions/status/uuid-here"
}
```

### 2. Check Status

**GET** `/api/questions/status/:questionId`

**Response (Processing):**
```json
{
  "success": true,
  "status": "processing",
  "stage": "getting_context",
  "message": "Retrieving context from dataset",
  "startTime": "2024-01-01T10:00:00.000Z",
  "lastUpdated": "2024-01-01T10:00:05.000Z"
}
```

**Response (Completed):**
```json
{
  "success": true,
  "status": "completed",
  "stage": "completed",
  "message": "Processing completed",
  "response": {
    "answer": "Artificial intelligence (AI) is...",
    "confidence": 0.95,
    "tags": ["artificial intelligence", "technology", "AI"],
    "sources": 3
  },
  "processingTime": 5000
}
```

### 3. Health Check

**GET** `/health`

```json
{
  "status": "OK",
  "timestamp": "2024-01-01T10:00:00.000Z",
  "uptime": 3600
}
```

## Processing Stages

1. **initializing**: Menerima dan memvalidasi pertanyaan
2. **getting_context**: Menggunakan Brain.js untuk mendapatkan konteks dari dataset
3. **tagging**: Menentukan tag berdasarkan analisis pertanyaan
4. **processing_ai**: Memproses dengan LangChain template dan Ollama
5. **completed**: Proses selesai dengan hasil
6. **error**: Terjadi error dalam proses

## Dataset Format

File `data/dataset.json` berisi data training untuk Brain.js:

```json
[
  {
    "input": {
      "question": "what is artificial intelligence",
      "category": "technology"
    },
    "output": {
      "relevance": 0.95,
      "tags": ["artificial intelligence", "technology", "AI"]
    }
  }
]
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3000 | Server port |
| `NODE_ENV` | development | Environment mode |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama service URL |
| `OLLAMA_MODEL` | llama2 | Model yang digunakan |
| `DATASET_PATH` | ./data/dataset.json | Path ke dataset |
| `LOG_LEVEL` | info | Level logging |
| `API_TIMEOUT` | 30000 | Timeout untuk API calls |

## Logging

Server menggunakan Winston untuk logging:

- `logs/combined.log`: Semua log
- `logs/error.log`: Error logs saja
- `logs/exceptions.log`: Unhandled exceptions
- `logs/rejections.log`: Unhandled promise rejections

## Error Handling

Server menangani berbagai jenis error:

- **Validation errors**: Input tidak valid
- **Service unavailable**: Ollama tidak tersedia
- **Timeout errors**: Request timeout
- **Neural network errors**: Error dari Brain.js
- **AI service errors**: Error dari Ollama

## Development

```bash
# Install dependencies
npm install

# Run in development mode with auto-reload
npm run dev

# Run tests (jika ada)
npm test
```

## Production Deployment

1. Set `NODE_ENV=production`
2. Pastikan Ollama service berjalan
3. Configure reverse proxy (nginx/apache)
4. Setup process manager (PM2)
5. Configure monitoring dan logging

```bash
# Using PM2
npm install -g pm2
pm2 start src/server.js --name "ai-server"
pm2 startup
pm2 save
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
ollama serve

# Check available models
ollama list
```

### Brain.js Training Issues

- Pastikan dataset.json format benar
- Check log untuk error training
- Adjust training parameters di QuestionProcessor.js

### Memory Issues

- Monitor memory usage dengan `htop` atau Task Manager
- Adjust Ollama model size
- Implement request rate limiting

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License - lihat file LICENSE untuk detail.

## Support

Untuk pertanyaan dan dukungan, silakan buat issue di repository ini.