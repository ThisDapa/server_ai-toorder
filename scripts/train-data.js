const fs = require('fs').promises;
const path = require('path');
const BrainService = require('../src/services/BrainService');
const logger = require('../src/utils/logger');

/**
 * Training Data Management Script
 * This script handles training data creation, validation, and model training
 */

class TrainingDataManager {
  constructor() {
    this.brainService = new BrainService();
    this.datasetPath = path.join(__dirname, '../data/dataset.json');
    this.trainedModelPath = path.join(__dirname, '../data/trained-model.json');
    this.backupPath = path.join(__dirname, '../data/backups');
  }

  createSampleData() {
    return [
      {
        question: "Selamat pagi, saya ingin bertanya",
        answer: "Selamat pagi! Ada yang bisa saya bantu?",
        tags: ["greeting"]
      },
      {
        question: "Berapa harga produk premium ini?",
        answer: "Untuk produk premium kami, harganya bervariasi tergantung fitur yang Anda pilih. Bisa tolong sebutkan produk spesifik yang Anda minati?",
        tags: ["price_inquiry"]
      },
      {
        question: "Apakah stok masih tersedia?",
        answer: "Saya akan periksa ketersediaan stoknya. Mohon sebutkan produk yang Anda cari.",
        tags: ["available"]
      },
      {
        question: "Saya butuh bantuan dengan pesanan",
        answer: "Tentu, saya siap membantu. Apa yang bisa saya bantu terkait pesanan Anda?",
        tags: ["help"]
      },
      {
        question: "Bagaimana cara melakukan pembayaran?",
        answer: "Kami menerima berbagai metode pembayaran termasuk transfer bank, kartu kredit, dan e-wallet. Metode pembayaran mana yang Anda prefer?",
        tags: ["payment"]
      },
      {
        question: "Terima kasih, sampai jumpa!",
        answer: "Terima kasih kembali! Semoga harinya menyenangkan!",
        tags: ["goodbye"]
      },
      {
        question: "Ada diskon untuk pembelian dalam jumlah besar?",
        answer: "Ya, kami menyediakan harga khusus untuk pembelian dalam jumlah besar. Berapa unit yang Anda rencanakan untuk dibeli?",
        tags: "price_inquiry"
      },
      {
        question: "Kapan pesanan saya akan sampai?",
        answer: "Saya akan bantu cek status pengiriman pesanan Anda. Mohon berikan nomor pesanannya.",
        tags: "help"
      },
      {
        question: "Saya ingin membatalkan pesanan",
        answer: "Saya akan bantu proses pembatalan pesanan. Mohon berikan nomor pesanan Anda.",
        tags: "help"
      },
      {
        question: "Apa kebijakan pengembalian barang?",
        answer: "Kebijakan pengembalian kami berlaku 30 hari sejak pembelian. Apakah Anda ingin saya jelaskan prosesnya?",
        tags: ["help"]
      },
      {
        question: "Apakah bisa kirim ke luar negeri?",
        answer: "Ya, kami melayani pengiriman internasional. Ke negara mana Anda ingin kami kirimkan?",
        tags: "help"
      },
      {
        question: "Ada masalah dengan pembayaran saya",
        answer: "Saya akan bantu menyelesaikan masalah pembayaran Anda. Bisa dijelaskan masalah yang Anda alami?",
        tags: "payment"
      },
      {
        question: "Jam berapa layanan customer service buka?",
        answer: "Layanan customer service kami tersedia 24/7 untuk membantu Anda.",
        tags: ["help"]
      },
      {
        question: "Saya ingin bicara dengan manager",
        answer: "Saya mengerti Anda ingin berbicara dengan manager. Mohon jelaskan keluhannya agar saya bisa bantu atau hubungkan dengan pihak yang tepat.",
        tags: ["help"]
      },
      {
        question: "asdfghjkl",
        answer: "Maaf, saya tidak memahami pesan Anda. Bisakah Anda menjelaskan kembali pertanyaannya?",
        tags: "unknown"
      }
    ];
  }

  /**
   * Validate training data structure
   */
  validateTrainingData(data) {
    const errors = [];
    
    if (!Array.isArray(data)) {
      errors.push('Training data must be an array');
      return { isValid: false, errors };
    }

    const validTags = ['greeting', 'price_inquiry', 'available', 'help', 'payment', 'goodbye', 'unknown'];

    data.forEach((item, index) => {
      if (!item.question || typeof item.question !== 'string') {
        errors.push(`Item ${index}: Missing or invalid question`);
      }
      
      if (!item.answer || typeof item.answer !== 'string') {
        errors.push(`Item ${index}: Missing or invalid answer`);
      }

      // Handle tags as array or string
      if (!item.tags) {
        errors.push(`Item ${index}: Missing tags`);
      } else if (Array.isArray(item.tags)) {
        const invalidTags = item.tags.filter(tag => !validTags.includes(tag));
        if (invalidTags.length > 0) {
          errors.push(`Item ${index}: Invalid tags: ${invalidTags.join(', ')}`);
        }
      } else if (typeof item.tags === 'string') {
        // If tags is a string, check if it's a valid tag
        if (!validTags.includes(item.tags)) {
          errors.push(`Item ${index}: Invalid tag: ${item.tags}`);
        }
      } else {
        errors.push(`Item ${index}: Tags must be an array or a string`);
      }
    });

    return {
      isValid: errors.length === 0,
      errors,
      itemCount: data.length
    };
  }

  /**
   * Load existing dataset or create new one
   */
  async loadOrCreateDataset() {
    try {
      // Try to load existing dataset
      const data = await fs.readFile(this.datasetPath, 'utf8');
      const dataset = JSON.parse(data);
      logger.info(`Loaded existing dataset with ${dataset.length} entries`);
      return dataset;
    } catch (error) {
      // Create new dataset with sample data
      logger.info('Creating new dataset with sample data');
      const sampleData = this.createSampleData();
      await this.saveDataset(sampleData);
      return sampleData;
    }
  }

  /**
   * Save dataset to file
   */
  async saveDataset(data) {
    try {
      // Ensure data directory exists
      const dataDir = path.dirname(this.datasetPath);
      await fs.mkdir(dataDir, { recursive: true });
      
      // Validate data before saving
      const validation = this.validateTrainingData(data);
      if (!validation.isValid) {
        throw new Error(`Invalid training data: ${validation.errors.join(', ')}`);
      }

      // Save dataset
      await fs.writeFile(this.datasetPath, JSON.stringify(data, null, 2));
      logger.info(`Dataset saved with ${data.length} entries`);
      return true;
    } catch (error) {
      logger.error(`Error saving dataset: ${error.message}`);
      throw error;
    }
  }

  /**
   * Add new training data to existing dataset
   */
  async addTrainingData(newData) {
    try {
      const existingData = await this.loadOrCreateDataset();
      const updatedData = [...existingData, ...newData];
      
      // Remove duplicates based on question
      const uniqueData = updatedData.filter((item, index, self) => 
        index === self.findIndex(t => t.question.toLowerCase() === item.question.toLowerCase())
      );
      
      await this.saveDataset(uniqueData);
      logger.info(`Added ${newData.length} new entries. Total: ${uniqueData.length}`);
      return uniqueData;
    } catch (error) {
      logger.error(`Error adding training data: ${error.message}`);
      throw error;
    }
  }

  /**
   * Create backup of current dataset
   */
  async createBackup() {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const backupDir = this.backupPath;
      const backupFile = path.join(backupDir, `dataset-backup-${timestamp}.json`);
      
      // Ensure backup directory exists
      await fs.mkdir(backupDir, { recursive: true });
      
      // Copy current dataset to backup
      const currentData = await fs.readFile(this.datasetPath, 'utf8');
      await fs.writeFile(backupFile, currentData);
      
      logger.info(`Backup created: ${backupFile}`);
      return backupFile;
    } catch (error) {
      logger.error(`Error creating backup: ${error.message}`);
      throw error;
    }
  }

  /**
   * Train the Brain.js model with current dataset
   */
  async trainModel() {
    try {
      logger.info('Starting model training...');
      
      // Load dataset
      const dataset = await this.loadOrCreateDataset();
      
      // Initialize and train Brain.js service
      await this.brainService.init();
      const trainingResult = await this.brainService.trainWithDataset(dataset);
      
      // Save trained model
      const modelData = await this.brainService.exportModel();
      await this.saveTrainedModel(modelData);
      
      logger.info('Model training completed successfully');
      return {
        success: true,
        trainingResult,
        datasetSize: dataset.length,
        modelPath: this.trainedModelPath
      };
    } catch (error) {
      logger.error(`Error training model: ${error.message}`);
      throw error;
    }
  }

  /**
   * Save trained model to file
   */
  async saveTrainedModel(modelData) {
    try {
      const dataDir = path.dirname(this.trainedModelPath);
      await fs.mkdir(dataDir, { recursive: true });
      
      const modelInfo = {
        model: modelData,
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        trainingStats: modelData.stats || {}
      };
      
      await fs.writeFile(this.trainedModelPath, JSON.stringify(modelInfo, null, 2));
      logger.info(`Trained model saved to ${this.trainedModelPath}`);
      return true;
    } catch (error) {
      logger.error(`Error saving trained model: ${error.message}`);
      throw error;
    }
  }

  /**
   * Load trained model
   */
  async loadTrainedModel() {
    try {
      const data = await fs.readFile(this.trainedModelPath, 'utf8');
      const modelInfo = JSON.parse(data);
      logger.info(`Loaded trained model from ${modelInfo.timestamp}`);
      return modelInfo;
    } catch (error) {
      logger.warn(`No trained model found: ${error.message}`);
      return null;
    }
  }

  /**
   * Test the trained model
   */
  async testModel() {
    try {
      const testQuestions = [
        'Bisa QRIS gk?',
        'Minta pricelist ny dong'
      ];
      
      const results = [];
      
      for (const question of testQuestions) {
        const result = await this.brainService.getContextFromDataset(question);
        
        results.push({
          question,
          predictedTags: result.predictedTags,
          context: result.context,
          relevantEntriesCount: result.relevantEntries.length
        });
      }
      
      logger.info('Model testing completed');
      return {
        success: true,
        testResults: results
      };
    } catch (error) {
      logger.error(`Error testing model: ${error.message}`);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get training statistics
   */
  async getTrainingStats() {
    try {
      const dataset = await this.loadOrCreateDataset();
      const trainedModel = await this.loadTrainedModel();
      
      // Analyze dataset
      const tagCounts = {};
      const tags = new Set();
      
      dataset.forEach(item => {
        // Handle both array and string tag formats
        if (item.tags) {
          if (Array.isArray(item.tags)) {
            item.tags.forEach(tag => {
              tags.add(tag);
              tagCounts[tag] = (tagCounts[tag] || 0) + 1;
            });
          } else if (typeof item.tags === 'string') {
            tags.add(item.tags);
            tagCounts[item.tags] = (tagCounts[item.tags] || 0) + 1;
          }
        }
      });
      
      // Sort tags by frequency
      const sortedTagCounts = {};
      Object.entries(tagCounts)
        .sort(([,a], [,b]) => b - a)
        .forEach(([tag, count]) => {
          sortedTagCounts[tag] = count;
        });

      // Calculate percentages
      const tagPercentages = {};
      Object.entries(sortedTagCounts).forEach(([tag, count]) => {
        tagPercentages[tag] = ((count / dataset.length) * 100).toFixed(2) + '%';
      });

      return {
        datasetSize: dataset.length,
        tagDistribution: sortedTagCounts,
        tagPercentages,
        uniqueTags: tags.size,
        hasTrainedModel: trainedModel !== null,
        modelTimestamp: trainedModel?.timestamp,
        modelVersion: trainedModel?.version
      };
    } catch (error) {
      logger.error(`Error getting training stats: ${error.message}`);
      throw error;
    }
  }
}

// CLI Interface
if (require.main === module) {
  const manager = new TrainingDataManager();
  const command = process.argv[2];
  
  async function runCommand() {
    try {
      switch (command) {
        case 'create':
          console.log('Creating sample dataset...');
          const sampleData = manager.createSampleData();
          await manager.saveDataset(sampleData);
          console.log('Sample dataset created successfully!');
          break;
          
        case 'train':
          console.log('Training model...');
          await manager.createBackup();
          const result = await manager.trainModel();
          console.log('Training completed:', result);
          break;
          
        case 'test':
          console.log('Testing model...');
          const testResult = await manager.testModel();
          console.log('Test results:', JSON.stringify(testResult, null, 2));
          break;
          
        case 'stats':
          console.log('Getting training statistics...');
          const stats = await manager.getTrainingStats();
          console.log('\nTraining Statistics:\n');
          console.log(`Total Dataset Size: ${stats.datasetSize} entries`);
          console.log(`Unique Tags: ${stats.uniqueTags}\n`);
          console.log('Tag Distribution:');
          Object.entries(stats.tagDistribution).forEach(([tag, count]) => {
            const percentage = stats.tagPercentages[tag];
            console.log(`- ${tag}: ${count} entries (${percentage})`);
          });
          console.log('\nModel Information:');
          console.log(`- Trained Model Available: ${stats.hasTrainedModel}`);
          if (stats.hasTrainedModel) {
            console.log(`- Last Training: ${stats.modelTimestamp}`);
            console.log(`- Model Version: ${stats.modelVersion}`);
          }
          break;
          
        case 'backup':
          console.log('Creating backup...');
          const backupPath = await manager.createBackup();
          console.log(`Backup created: ${backupPath}`);
          break;
          
        default:
          console.log(`
Training Data Management Script

Usage: node train-data.js <command>

Commands:
  create  - Create sample dataset
  train   - Train the Brain.js model
  test    - Test the trained model
  stats   - Show training statistics
  backup  - Create dataset backup

Examples:
  node train-data.js create
  node train-data.js train
  node train-data.js test
`);
      }
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  }
  
  runCommand();
}

module.exports = TrainingDataManager;