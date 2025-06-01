const fs = require('fs').promises;
const path = require('path');
const BrainService = require('../src/services/BrainService');
const logger = require('../src/utils/logger');

/**
 * Model Management Script
 * This script handles saving, loading, and managing trained Brain.js models
 */

class ModelManager {
  constructor() {
    this.brainService = new BrainService();
    this.modelsDir = path.join(__dirname, '../data/models');
    this.currentModelPath = path.join(this.modelsDir, 'current-model.json');
    this.modelHistoryPath = path.join(this.modelsDir, 'model-history.json');
  }

  /**
   * Initialize model manager
   */
  async init() {
    try {
      // Ensure models directory exists
      await fs.mkdir(this.modelsDir, { recursive: true });
      logger.info('ModelManager initialized');
    } catch (error) {
      logger.error(`Error initializing ModelManager: ${error.message}`);
      throw error;
    }
  }

  /**
   * Save current trained model
   */
  async saveCurrentModel(metadata = {}) {
    try {
      await this.init();
      
      // Get model data from BrainService
      const modelData = await this.brainService.exportModel();
      const stats = this.brainService.getServiceStats();
      
      const modelInfo = {
        id: this.generateModelId(),
        timestamp: new Date().toISOString(),
        version: metadata.version || '1.0.0',
        description: metadata.description || 'Trained Brain.js model',
        modelData: modelData,
        stats: stats,
        performance: metadata.performance || {},
        trainingConfig: {
          datasetSize: metadata.datasetSize || 0,
          trainingTime: metadata.trainingTime || 0,
          iterations: metadata.iterations || 0,
          error: metadata.error || 0
        },
        metadata: metadata
      };
      
      // Save current model
      await fs.writeFile(this.currentModelPath, JSON.stringify(modelInfo, null, 2));
      
      // Update model history
      await this.updateModelHistory(modelInfo);
      
      logger.info(`Model saved with ID: ${modelInfo.id}`);
      return modelInfo;
    } catch (error) {
      logger.error(`Error saving model: ${error.message}`);
      throw error;
    }
  }

  /**
   * Load current model
   */
  async loadCurrentModel() {
    try {
      const data = await fs.readFile(this.currentModelPath, 'utf8');
      const modelInfo = JSON.parse(data);
      
      // Load model into BrainService
      await this.brainService.loadModel(modelInfo.modelData);
      
      logger.info(`Loaded model: ${modelInfo.id} (${modelInfo.timestamp})`);
      return modelInfo;
    } catch (error) {
      logger.warn(`No current model found: ${error.message}`);
      return null;
    }
  }

  /**
   * Save model with specific name/version
   */
  async saveNamedModel(name, version, metadata = {}) {
    try {
      await this.init();
      
      const modelData = await this.brainService.exportModel();
      const stats = this.brainService.getServiceStats();
      
      const modelInfo = {
        id: this.generateModelId(),
        name: name,
        version: version,
        timestamp: new Date().toISOString(),
        description: metadata.description || `${name} v${version}`,
        modelData: modelData,
        stats: stats,
        performance: metadata.performance || {},
        trainingConfig: metadata.trainingConfig || {},
        metadata: metadata
      };
      
      const namedModelPath = path.join(this.modelsDir, `${name}-v${version}.json`);
      await fs.writeFile(namedModelPath, JSON.stringify(modelInfo, null, 2));
      
      // Update model history
      await this.updateModelHistory(modelInfo);
      
      logger.info(`Named model saved: ${name} v${version}`);
      return modelInfo;
    } catch (error) {
      logger.error(`Error saving named model: ${error.message}`);
      throw error;
    }
  }

  /**
   * Load named model
   */
  async loadNamedModel(name, version) {
    try {
      const namedModelPath = path.join(this.modelsDir, `${name}-v${version}.json`);
      const data = await fs.readFile(namedModelPath, 'utf8');
      const modelInfo = JSON.parse(data);
      
      // Load model into BrainService
      await this.brainService.loadModel(modelInfo.modelData);
      
      logger.info(`Loaded named model: ${name} v${version}`);
      return modelInfo;
    } catch (error) {
      logger.error(`Error loading named model ${name} v${version}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Update model history
   */
  async updateModelHistory(modelInfo) {
    try {
      let history = [];
      
      // Load existing history
      try {
        const data = await fs.readFile(this.modelHistoryPath, 'utf8');
        history = JSON.parse(data);
      } catch (error) {
        // History file doesn't exist, start with empty array
      }
      
      // Add new model to history
      const historyEntry = {
        id: modelInfo.id,
        name: modelInfo.name || 'unnamed',
        version: modelInfo.version,
        timestamp: modelInfo.timestamp,
        description: modelInfo.description,
        performance: modelInfo.performance,
        trainingConfig: modelInfo.trainingConfig
      };
      
      history.unshift(historyEntry); // Add to beginning
      
      // Keep only last 50 entries
      if (history.length > 50) {
        history = history.slice(0, 50);
      }
      
      // Save updated history
      await fs.writeFile(this.modelHistoryPath, JSON.stringify(history, null, 2));
      
      logger.info('Model history updated');
    } catch (error) {
      logger.error(`Error updating model history: ${error.message}`);
    }
  }

  /**
   * Get model history
   */
  async getModelHistory() {
    try {
      const data = await fs.readFile(this.modelHistoryPath, 'utf8');
      return JSON.parse(data);
    } catch (error) {
      logger.warn(`No model history found: ${error.message}`);
      return [];
    }
  }

  /**
   * List all saved models
   */
  async listModels() {
    try {
      await this.init();
      
      const files = await fs.readdir(this.modelsDir);
      const modelFiles = files.filter(file => file.endsWith('.json') && file !== 'model-history.json');
      
      const models = [];
      
      for (const file of modelFiles) {
        try {
          const filePath = path.join(this.modelsDir, file);
          const data = await fs.readFile(filePath, 'utf8');
          const modelInfo = JSON.parse(data);
          
          models.push({
            filename: file,
            id: modelInfo.id,
            name: modelInfo.name || 'unnamed',
            version: modelInfo.version,
            timestamp: modelInfo.timestamp,
            description: modelInfo.description,
            size: data.length
          });
        } catch (error) {
          logger.warn(`Error reading model file ${file}: ${error.message}`);
        }
      }
      
      // Sort by timestamp (newest first)
      models.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      
      return models;
    } catch (error) {
      logger.error(`Error listing models: ${error.message}`);
      throw error;
    }
  }

  /**
   * Delete a model
   */
  async deleteModel(filename) {
    try {
      const filePath = path.join(this.modelsDir, filename);
      await fs.unlink(filePath);
      logger.info(`Model deleted: ${filename}`);
      return true;
    } catch (error) {
      logger.error(`Error deleting model ${filename}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Compare two models
   */
  async compareModels(modelId1, modelId2) {
    try {
      const models = await this.listModels();
      const model1 = models.find(m => m.id === modelId1);
      const model2 = models.find(m => m.id === modelId2);
      
      if (!model1 || !model2) {
        throw new Error('One or both models not found');
      }
      
      // Load full model data for comparison
      const model1Path = path.join(this.modelsDir, model1.filename);
      const model2Path = path.join(this.modelsDir, model2.filename);
      
      const model1Data = JSON.parse(await fs.readFile(model1Path, 'utf8'));
      const model2Data = JSON.parse(await fs.readFile(model2Path, 'utf8'));
      
      const comparison = {
        model1: {
          id: model1Data.id,
          name: model1Data.name,
          version: model1Data.version,
          timestamp: model1Data.timestamp,
          performance: model1Data.performance,
          trainingConfig: model1Data.trainingConfig
        },
        model2: {
          id: model2Data.id,
          name: model2Data.name,
          version: model2Data.version,
          timestamp: model2Data.timestamp,
          performance: model2Data.performance,
          trainingConfig: model2Data.trainingConfig
        },
        differences: {
          timeDiff: new Date(model2Data.timestamp) - new Date(model1Data.timestamp),
          versionDiff: model2Data.version !== model1Data.version,
          performanceDiff: this.comparePerformance(model1Data.performance, model2Data.performance)
        }
      };
      
      return comparison;
    } catch (error) {
      logger.error(`Error comparing models: ${error.message}`);
      throw error;
    }
  }

  /**
   * Compare performance metrics
   */
  comparePerformance(perf1, perf2) {
    const metrics = ['accuracy', 'precision', 'recall', 'f1Score', 'error'];
    const comparison = {};
    
    metrics.forEach(metric => {
      if (perf1[metric] !== undefined && perf2[metric] !== undefined) {
        comparison[metric] = {
          model1: perf1[metric],
          model2: perf2[metric],
          difference: perf2[metric] - perf1[metric],
          improvement: perf2[metric] > perf1[metric]
        };
      }
    });
    
    return comparison;
  }

  /**
   * Export model for deployment
   */
  async exportModelForDeployment(modelId, exportPath) {
    try {
      const models = await this.listModels();
      const model = models.find(m => m.id === modelId);
      
      if (!model) {
        throw new Error(`Model with ID ${modelId} not found`);
      }
      
      const modelPath = path.join(this.modelsDir, model.filename);
      const modelData = JSON.parse(await fs.readFile(modelPath, 'utf8'));
      
      // Create deployment package
      const deploymentPackage = {
        model: modelData.modelData,
        metadata: {
          id: modelData.id,
          name: modelData.name,
          version: modelData.version,
          timestamp: modelData.timestamp,
          description: modelData.description
        },
        config: {
          inputFormat: 'vector',
          outputFormat: 'object',
          requiredLibraries: ['brain.js']
        }
      };
      
      // Ensure export directory exists
      const exportDir = path.dirname(exportPath);
      await fs.mkdir(exportDir, { recursive: true });
      
      // Save deployment package
      await fs.writeFile(exportPath, JSON.stringify(deploymentPackage, null, 2));
      
      logger.info(`Model exported for deployment: ${exportPath}`);
      return deploymentPackage;
    } catch (error) {
      logger.error(`Error exporting model: ${error.message}`);
      throw error;
    }
  }

  /**
   * Generate unique model ID
   */
  generateModelId() {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 8);
    return `model_${timestamp}_${random}`;
  }

  /**
   * Get model statistics
   */
  async getModelStats() {
    try {
      const models = await this.listModels();
      const history = await this.getModelHistory();
      
      // Calculate total size
      const totalSize = models.reduce((sum, model) => sum + model.size, 0);
      
      // Group by name
      const modelsByName = {};
      models.forEach(model => {
        const name = model.name || 'unnamed';
        if (!modelsByName[name]) {
          modelsByName[name] = [];
        }
        modelsByName[name].push(model);
      });
      
      return {
        totalModels: models.length,
        totalSize: totalSize,
        totalSizeFormatted: this.formatBytes(totalSize),
        modelsByName: Object.keys(modelsByName).map(name => ({
          name,
          count: modelsByName[name].length,
          latestVersion: modelsByName[name][0].version
        })),
        historyEntries: history.length,
        oldestModel: models.length > 0 ? models[models.length - 1].timestamp : null,
        newestModel: models.length > 0 ? models[0].timestamp : null
      };
    } catch (error) {
      logger.error(`Error getting model stats: ${error.message}`);
      throw error;
    }
  }

  /**
   * Format bytes to human readable format
   */
  formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Clean up old models (keep only latest N versions per name)
   */
  async cleanupOldModels(keepPerName = 5) {
    try {
      const models = await this.listModels();
      
      // Group by name
      const modelsByName = {};
      models.forEach(model => {
        const name = model.name || 'unnamed';
        if (!modelsByName[name]) {
          modelsByName[name] = [];
        }
        modelsByName[name].push(model);
      });
      
      const deletedModels = [];
      
      // For each name, keep only the latest N versions
      for (const [name, nameModels] of Object.entries(modelsByName)) {
        if (nameModels.length > keepPerName) {
          const toDelete = nameModels.slice(keepPerName);
          
          for (const model of toDelete) {
            await this.deleteModel(model.filename);
            deletedModels.push(model);
          }
        }
      }
      
      logger.info(`Cleanup completed. Deleted ${deletedModels.length} old models`);
      return {
        deletedCount: deletedModels.length,
        deletedModels: deletedModels.map(m => ({ name: m.name, version: m.version, timestamp: m.timestamp }))
      };
    } catch (error) {
      logger.error(`Error during cleanup: ${error.message}`);
      throw error;
    }
  }
}

// CLI Interface
if (require.main === module) {
  const manager = new ModelManager();
  const command = process.argv[2];
  const arg1 = process.argv[3];
  const arg2 = process.argv[4];
  
  async function runCommand() {
    try {
      switch (command) {
        case 'list':
          console.log('Listing all models...');
          const models = await manager.listModels();
          console.table(models);
          break;
          
        case 'stats':
          console.log('Getting model statistics...');
          const stats = await manager.getModelStats();
          console.log('Model Statistics:', JSON.stringify(stats, null, 2));
          break;
          
        case 'history':
          console.log('Getting model history...');
          const history = await manager.getModelHistory();
          console.table(history);
          break;
          
        case 'save':
          if (!arg1 || !arg2) {
            console.log('Usage: node model-manager.js save <name> <version>');
            return;
          }
          console.log(`Saving model as ${arg1} v${arg2}...`);
          const savedModel = await manager.saveNamedModel(arg1, arg2);
          console.log('Model saved:', savedModel.id);
          break;
          
        case 'load':
          if (!arg1 || !arg2) {
            console.log('Usage: node model-manager.js load <name> <version>');
            return;
          }
          console.log(`Loading model ${arg1} v${arg2}...`);
          await manager.loadNamedModel(arg1, arg2);
          console.log('Model loaded successfully');
          break;
          
        case 'delete':
          if (!arg1) {
            console.log('Usage: node model-manager.js delete <filename>');
            return;
          }
          console.log(`Deleting model ${arg1}...`);
          await manager.deleteModel(arg1);
          console.log('Model deleted successfully');
          break;
          
        case 'cleanup':
          const keepCount = arg1 ? parseInt(arg1) : 5;
          console.log(`Cleaning up old models (keeping ${keepCount} per name)...`);
          const cleanupResult = await manager.cleanupOldModels(keepCount);
          console.log('Cleanup result:', cleanupResult);
          break;
          
        case 'export':
          if (!arg1 || !arg2) {
            console.log('Usage: node model-manager.js export <modelId> <exportPath>');
            return;
          }
          console.log(`Exporting model ${arg1} to ${arg2}...`);
          await manager.exportModelForDeployment(arg1, arg2);
          console.log('Model exported successfully');
          break;
          
        default:
          console.log(`
Model Management Script

Usage: node model-manager.js <command> [args]

Commands:
  list                    - List all saved models
  stats                   - Show model statistics
  history                 - Show model history
  save <name> <version>   - Save current model with name and version
  load <name> <version>   - Load a specific model
  delete <filename>       - Delete a model file
  cleanup [keepCount]     - Clean up old models (default: keep 5 per name)
  export <id> <path>      - Export model for deployment

Examples:
  node model-manager.js list
  node model-manager.js save production 1.0.0
  node model-manager.js load production 1.0.0
  node model-manager.js cleanup 3
`);
      }
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  }
  
  runCommand();
}

module.exports = ModelManager;