/**
 * Model utilities for BrainService
 * Extracted from BrainService.js to improve maintainability
 */

const brain = require('brain.js');
const fs = require('fs');
const path = require('path');
const textUtils = require('./textUtils');

module.exports = {
  /**
   * Initializes a new neural network
   * @returns {Object} - Initialized neural network
   */
  initializeNetwork() {
    return new brain.NeuralNetwork({
      hiddenLayers: [10, 8],
      activation: 'sigmoid',
      learningRate: 0.1,
      errorThresh: 0.005
    });
  },

  /**
   * Loads a trained model from file
   * @param {string} modelPath - Path to model file
   * @returns {Object|null} - Loaded neural network or null if error
   */
  loadTrainedModel(modelPath) {
    try {
      if (fs.existsSync(modelPath)) {
        const modelData = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
        const network = this.initializeNetwork();
        network.fromJSON(modelData);
        return network;
      }
      return null;
    } catch (error) {
      console.error('Error loading trained model:', error);
      return null;
    }
  },

  /**
   * Saves a trained model to file
   * @param {Object} network - Trained neural network
   * @param {string} modelPath - Path to save model
   * @returns {boolean} - True if successful
   */
  saveTrainedModel(network, modelPath) {
    try {
      const modelData = network.toJSON();
      const modelDir = path.dirname(modelPath);
      
      if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir, { recursive: true });
      }
      
      fs.writeFileSync(modelPath, JSON.stringify(modelData), 'utf8');
      return true;
    } catch (error) {
      console.error('Error saving trained model:', error);
      return false;
    }
  },

  /**
   * Trains a neural network with dataset
   * @param {Array<Object>} dataset - Training dataset
   * @param {Object} options - Training options
   * @returns {Object} - Trained network and stats
   */
  trainBrainNetwork(dataset, options = {}) {
    if (!dataset || dataset.length === 0) {
      throw new Error('Cannot train with empty dataset');
    }
    
    const {
      iterations = 20000,
      errorThresh = 0.005,
      logPeriod = 1000,
      logCallback = null
    } = options;
    
    // Prepare training data
    const trainingData = dataset.map(item => ({
      input: this.prepareNetworkInput(item.question),
      output: this.prepareNetworkOutput(item.tags)
    }));
    
    // Initialize and train network
    const network = this.initializeNetwork();
    
    const trainingOptions = {
      iterations,
      errorThresh,
      log: logCallback !== null,
      logPeriod
    };
    
    if (logCallback) {
      trainingOptions.log = true;
      trainingOptions.logPeriod = logPeriod;
    }
    
    const stats = network.train(trainingData, trainingOptions);
    
    return { network, stats };
  },

  /**
   * Prepares input for neural network from question text
   * @param {string} question - Question text
   * @returns {Object} - Input object for neural network
   */
  prepareNetworkInput(question) {
    if (!question) return {};
    
    // Normalize and stem the question
    const normalizedQuestion = textUtils.normalizeText(question);
    const stemmedQuestion = textUtils.stemText(normalizedQuestion);
    
    // Create input object with word presence
    const words = stemmedQuestion.split(' ');
    const input = {};
    
    words.forEach(word => {
      if (word.length > 2) { // Ignore very short words
        input[word] = 1;
      }
    });
    
    return input;
  },

  /**
   * Prepares output for neural network from tags
   * @param {Array<string>} tags - Array of tags
   * @returns {Object} - Output object for neural network
   */
  prepareNetworkOutput(tags) {
    if (!tags || !Array.isArray(tags)) return {};
    
    const output = {};
    tags.forEach(tag => {
      output[tag] = 1;
    });
    
    return output;
  },

  /**
   * Gets predicted tags from neural network output
   * @param {Object} networkOutput - Neural network output
   * @param {number} confidenceThreshold - Minimum confidence threshold
   * @returns {Array<Object>} - Array of predicted tags with confidence
   */
  getPredictedTags(networkOutput, confidenceThreshold = 0.5) {
    if (!networkOutput) return [];
    
    const predictions = [];
    
    for (const [tag, confidence] of Object.entries(networkOutput)) {
      if (confidence >= confidenceThreshold) {
        predictions.push({ tag, confidence });
      }
    }
    
    // Sort by confidence (highest first)
    return predictions.sort((a, b) => b.confidence - a.confidence);
  },

  /**
   * Extracts unique tags from dataset
   * @param {Array<Object>} dataset - Dataset with tags
   * @returns {Array<string>} - Array of unique tags
   */
  extractUniqueTags(dataset) {
    if (!dataset || !Array.isArray(dataset)) return [];
    
    const tagSet = new Set();
    
    dataset.forEach(item => {
      if (item.tags && Array.isArray(item.tags)) {
        item.tags.forEach(tag => tagSet.add(tag));
      }
    });
    
    return Array.from(tagSet);
  },

  /**
   * Generates augmented dataset with variations
   * @param {Array<Object>} dataset - Original dataset
   * @param {Object} options - Augmentation options
   * @returns {Array<Object>} - Augmented dataset
   */
  generateAugmentedDataset(dataset, options = {}) {
    if (!dataset || !Array.isArray(dataset)) return [];
    
    const {
      includeTypos = true,
      includeWordOrder = true,
      typoRate = 0.1,
      maxVariationsPerItem = 2
    } = options;
    
    const augmentedDataset = [...dataset];
    
    dataset.forEach(item => {
      if (!item.question) return;
      
      const variations = [];
      
      // Generate typo variations
      if (includeTypos) {
        const typoVariations = this.generateTypoVariations(
          item.question, 
          Math.min(maxVariationsPerItem, 2),
          typoRate
        );
        
        typoVariations.forEach(variation => {
          variations.push({
            ...item,
            question: variation,
            isAugmented: true
          });
        });
      }
      
      // Generate word order variations
      if (includeWordOrder && variations.length < maxVariationsPerItem) {
        const orderVariations = this.generateWordOrderVariations(
          item.question,
          Math.min(maxVariationsPerItem - variations.length, 2)
        );
        
        orderVariations.forEach(variation => {
          variations.push({
            ...item,
            question: variation,
            isAugmented: true
          });
        });
      }
      
      // Add variations to augmented dataset
      augmentedDataset.push(...variations);
    });
    
    return augmentedDataset;
  },

  /**
   * Generates variations with typos
   * @param {string} text - Original text
   * @param {number} count - Number of variations to generate
   * @param {number} typoRate - Rate of typos to introduce
   * @returns {Array<string>} - Variations with typos
   */
  generateTypoVariations(text, count = 1, typoRate = 0.1) {
    if (!text) return [];
    
    const variations = [];
    const words = text.split(' ');
    
    for (let i = 0; i < count; i++) {
      const newWords = [...words];
      
      // Determine how many words to modify
      const wordsToModify = Math.max(1, Math.floor(words.length * typoRate));
      
      for (let j = 0; j < wordsToModify; j++) {
        const randomIndex = Math.floor(Math.random() * words.length);
        const word = words[randomIndex];
        
        if (word.length <= 2) continue;
        
        // Apply random typo transformation
        const typoType = Math.floor(Math.random() * 3);
        
        switch (typoType) {
          case 0: // Character swap
            if (word.length >= 3) {
              const pos = Math.floor(Math.random() * (word.length - 2)) + 1;
              newWords[randomIndex] = 
                word.substring(0, pos - 1) + 
                word.charAt(pos) + 
                word.charAt(pos - 1) + 
                word.substring(pos + 1);
            }
            break;
            
          case 1: // Character deletion
            const delPos = Math.floor(Math.random() * word.length);
            newWords[randomIndex] = 
              word.substring(0, delPos) + 
              word.substring(delPos + 1);
            break;
            
          case 2: // Character duplication
            const dupPos = Math.floor(Math.random() * word.length);
            newWords[randomIndex] = 
              word.substring(0, dupPos) + 
              word.charAt(dupPos) + 
              word.substring(dupPos);
            break;
        }
      }
      
      variations.push(newWords.join(' '));
    }
    
    return variations;
  },

  /**
   * Generates variations with different word orders
   * @param {string} text - Original text
   * @param {number} count - Number of variations to generate
   * @returns {Array<string>} - Variations with different word orders
   */
  generateWordOrderVariations(text, count = 1) {
    if (!text) return [];
    
    const variations = [];
    const words = text.split(' ');
    
    // Only generate variations for texts with enough words
    if (words.length <= 3) return [];
    
    for (let i = 0; i < count; i++) {
      const newWords = [...words];
      
      // Swap random adjacent words
      const pos = Math.floor(Math.random() * (words.length - 1));
      const temp = newWords[pos];
      newWords[pos] = newWords[pos + 1];
      newWords[pos + 1] = temp;
      
      variations.push(newWords.join(' '));
    }
    
    return variations;
  },

  /**
   * Gets model statistics
   * @param {Object} network - Neural network
   * @param {Array<Object>} dataset - Dataset for testing
   * @returns {Object} - Model statistics
   */
  getModelStats(network, dataset) {
    if (!network || !dataset || dataset.length === 0) {
      return { accuracy: 0, precision: 0, recall: 0, f1Score: 0 };
    }
    
    let correct = 0;
    let total = 0;
    let truePositives = 0;
    let falsePositives = 0;
    let falseNegatives = 0;
    
    dataset.forEach(item => {
      if (!item.question || !item.tags || !Array.isArray(item.tags)) return;
      
      const input = this.prepareNetworkInput(item.question);
      const output = network.run(input);
      const predictions = this.getPredictedTags(output, 0.5);
      const predictedTags = predictions.map(p => p.tag);
      
      // Count correct predictions
      item.tags.forEach(tag => {
        total++;
        
        if (predictedTags.includes(tag)) {
          correct++;
          truePositives++;
        } else {
          falseNegatives++;
        }
      });
      
      // Count false positives
      predictedTags.forEach(tag => {
        if (!item.tags.includes(tag)) {
          falsePositives++;
        }
      });
    });
    
    // Calculate metrics
    const accuracy = total > 0 ? correct / total : 0;
    const precision = (truePositives + falsePositives) > 0 ? 
                     truePositives / (truePositives + falsePositives) : 0;
    const recall = (truePositives + falseNegatives) > 0 ? 
                  truePositives / (truePositives + falseNegatives) : 0;
    const f1Score = (precision + recall) > 0 ? 
                   2 * (precision * recall) / (precision + recall) : 0;
    
    return { accuracy, precision, recall, f1Score };
  }
};