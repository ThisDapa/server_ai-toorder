# Training and Model Management Guide

This guide explains how to train Brain.js models, manage training data, and handle model versioning in the AI server.

## Overview

The AI server uses a separated architecture:
- **BrainService**: Handles context extraction and question tagging using Brain.js
- **OllamaService**: Processes questions using LangChain templates and Ollama
- **Training Scripts**: Manage training data and model persistence

## Architecture Separation

### BrainService (`src/services/BrainService.js`)
- Context extraction from dataset
- Question tagging and categorization
- Similarity calculations
- Model training and persistence

### OllamaService (`src/services/OllamaService.js`)
- LangChain template management
- Ollama model integration
- AI response generation
- Template selection based on context

### QuestionProcessor (`src/services/QuestionProcessor.js`)
- Orchestrates BrainService and OllamaService
- Manages the complete question processing pipeline

## Training Data Management

### Dataset Structure

Training data should follow this structure:

```json
[
  {
    "question": "What is artificial intelligence?",
    "output": {
      "category": "technology",
      "tags": ["AI", "technology", "machine learning"],
      "context": "Description of the topic",
      "difficulty": "beginner",
      "topics": ["definition", "overview"]
    }
  }
]
```

### Training Scripts

#### 1. Create Sample Dataset
```bash
npm run train:create
```
Creates a sample dataset with various categories and examples.

#### 2. Train Model
```bash
npm run train:data
```
Trains the Brain.js model with the current dataset and saves it.

#### 3. Test Model
```bash
npm run train:test
```
Tests the trained model with sample questions.

#### 4. View Statistics
```bash
npm run train:stats
```
Shows training statistics including dataset size, categories, and model info.

#### 5. Create Backup
```bash
npm run train:backup
```
Creates a timestamped backup of the current dataset.

### Manual Training Data Management

```javascript
const TrainingDataManager = require('./scripts/train-data.js');
const manager = new TrainingDataManager();

// Add new training data
const newData = [
  {
    question: "How to deploy a web application?",
    output: {
      category: "technology",
      tags: ["deployment", "web", "hosting"],
      context: "Web application deployment involves...",
      difficulty: "intermediate",
      topics: ["deployment", "hosting", "DevOps"]
    }
  }
];

await manager.addTrainingData(newData);
await manager.trainModel();
```

## Model Management

### Model Scripts

#### 1. List Models
```bash
npm run model:list
```
Shows all saved models with their metadata.

#### 2. Model Statistics
```bash
npm run model:stats
```
Displays comprehensive model statistics.

#### 3. Model History
```bash
npm run model:history
```
Shows the history of model training sessions.

#### 4. Save Named Model
```bash
npm run model:save production 1.0.0
```
Saves the current model with a specific name and version.

#### 5. Load Named Model
```bash
npm run model:load production 1.0.0
```
Loads a specific model version.

#### 6. Cleanup Old Models
```bash
npm run model:cleanup 3
```
Keeps only the latest 3 versions of each model name.

#### 7. Export Model
```bash
npm run model:export model_id ./exports/production-model.json
```
Exports a model for deployment.

### Manual Model Management

```javascript
const ModelManager = require('./scripts/model-manager.js');
const manager = new ModelManager();

// Save current model
const modelInfo = await manager.saveCurrentModel({
  version: '2.0.0',
  description: 'Improved accuracy model',
  performance: {
    accuracy: 0.95,
    precision: 0.92,
    recall: 0.89
  }
});

// Compare models
const comparison = await manager.compareModels(modelId1, modelId2);
console.log(comparison);
```

## Training Workflow

### 1. Initial Setup
```bash
# Create initial dataset
npm run train:create

# Train the model
npm run train:data

# Test the model
npm run train:test
```

### 2. Iterative Improvement
```bash
# Create backup before changes
npm run train:backup

# Add new training data (manually edit dataset.json)
# Or use the TrainingDataManager programmatically

# Retrain with new data
npm run train:data

# Test improvements
npm run train:test

# Save as new version if satisfied
npm run model:save improved 1.1.0
```

### 3. Production Deployment
```bash
# Save production model
npm run model:save production 1.0.0

# Export for deployment
npm run model:export model_id ./production/model.json

# Clean up old versions
npm run model:cleanup 5
```

## Environment Variables

```env
# Training Configuration
DATASET_PATH=./data/dataset.json
TRAINED_MODEL_PATH=./data/trained-model.json
MODEL_BACKUP_DIR=./data/backups

# Brain.js Configuration
BRAIN_LEARNING_RATE=0.3
BRAIN_ITERATIONS=20000
BRAIN_ERROR_THRESH=0.005
BRAIN_TIMEOUT=20000

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TEMPERATURE=0.7
OLLAMA_TOP_P=0.9
OLLAMA_TOP_K=40
```

## Data Directory Structure

```
data/
├── dataset.json              # Main training dataset
├── trained-model.json        # Current trained model
├── backups/                  # Dataset backups
│   ├── dataset-backup-2024-01-01T10-00-00.json
│   └── dataset-backup-2024-01-02T15-30-00.json
└── models/                   # Saved models
    ├── current-model.json    # Current active model
    ├── model-history.json    # Model training history
    ├── production-v1.0.0.json
    └── improved-v1.1.0.json
```

## API Integration

The trained models are automatically loaded by the services:

```javascript
// BrainService automatically loads the trained model
const brainService = new BrainService();
await brainService.init(); // Loads saved model if available

// Process questions using trained model
const result = await brainService.processContext(question);
const tags = await brainService.tagQuestion(question, result.relevantEntries);
```

## Performance Monitoring

### Training Metrics
- **Error Rate**: Lower is better (target < 0.01)
- **Iterations**: Number of training cycles
- **Training Time**: Time taken to train
- **Dataset Size**: Number of training examples

### Model Performance
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Monitoring Commands
```bash
# Check training statistics
npm run train:stats

# Check model performance
npm run model:stats

# View training history
npm run model:history
```

## Best Practices

### 1. Data Quality
- Ensure diverse and representative training data
- Include various question types and categories
- Maintain consistent tagging and categorization
- Regular data validation and cleanup

### 2. Model Versioning
- Use semantic versioning (major.minor.patch)
- Save models before significant changes
- Keep production models separate from experimental ones
- Document model changes and improvements

### 3. Training Process
- Create backups before retraining
- Test models thoroughly before deployment
- Monitor performance metrics over time
- Implement gradual rollout for new models

### 4. Performance Optimization
- Balance dataset size with training time
- Adjust Brain.js parameters for optimal performance
- Use appropriate similarity thresholds
- Regular model cleanup to save storage

## Troubleshooting

### Common Issues

#### 1. Training Fails
```bash
# Check dataset validity
npm run train:stats

# Verify dataset structure
node -e "console.log(JSON.parse(require('fs').readFileSync('./data/dataset.json', 'utf8')))"
```

#### 2. Model Not Loading
```bash
# Check if model file exists
ls -la data/

# Verify model structure
npm run model:list
```

#### 3. Poor Performance
- Increase dataset size
- Improve data quality
- Adjust Brain.js parameters
- Add more diverse examples

#### 4. Memory Issues
- Reduce dataset size for testing
- Adjust Brain.js timeout settings
- Monitor system resources during training

## Advanced Usage

### Custom Training Data Sources
```javascript
// Load data from external API
const response = await fetch('https://api.example.com/training-data');
const externalData = await response.json();

// Transform and add to dataset
const transformedData = externalData.map(item => ({
  question: item.query,
  output: {
    category: item.type,
    tags: item.keywords,
    context: item.description,
    difficulty: item.level,
    topics: item.subjects
  }
}));

await manager.addTrainingData(transformedData);
```

### Automated Training Pipeline
```javascript
// Scheduled retraining
const cron = require('node-cron');

cron.schedule('0 2 * * 0', async () => {
  console.log('Starting weekly model retraining...');
  
  // Create backup
  await manager.createBackup();
  
  // Retrain model
  const result = await manager.trainModel();
  
  // Test performance
  const testResult = await manager.testModel();
  
  // Save if performance is good
  if (testResult.success) {
    await modelManager.saveCurrentModel({
      version: `auto-${Date.now()}`,
      description: 'Automated weekly training',
      performance: testResult
    });
  }
});
```

This comprehensive training and model management system provides a robust foundation for maintaining and improving the AI server's performance over time.