/**
 * BrainService Module
 * Provides vector-based context search using FAISSStore and Ollama embeddings
 * Replaces previous brain.js neural network implementation
 */

'use strict';

const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');
const vectorUtils = require('../utils/vectorUtils');
const textUtils = require('../utils/textUtils');
const { FaissStore } = require('@langchain/community/vectorstores/faiss');
const { OllamaEmbeddings } = require('@langchain/ollama');

class BrainService {
  constructor() {
    this.vectorStore = null;
    this.embeddings = null;
    this.dataset = null;
    this.isInitialized = false;
    this.vectorStorePath = path.resolve("./models/vector-store");
    this.datasetPath = process.env.DATASET_PATH || "./data/dataset.json";
    this.embeddingsModel = process.env.OLLAMA_EMBEDDINGS_MODEL || "bge-m3";
  }

  /**
   * Initialize BrainService with FAISSStore and Ollama embeddings
   */
  async init() {
    try {
      logger.info("Initializing BrainService", {
        embeddingsModel: this.embeddingsModel,
        datasetPath: this.datasetPath,
        vectorStorePath: this.vectorStorePath,
      });

      // Load dataset first (we need this regardless of Ollama status)
      const datasetStartTime = Date.now();
      await this.loadDataset();
      logger.info("Dataset loaded", {
        loadTimeMs: Date.now() - datasetStartTime,
        entriesCount: this.dataset ? this.dataset.length : 0,
      });

      // Initialize Ollama embeddings with retry and timeout
      try {
        this.embeddings = new OllamaEmbeddings({
          model: this.embeddingsModel,
          timeout: 10000, // 10 second timeout
        });
        logger.info("Ollama embeddings initialized", {
          model: this.embeddingsModel,
        });

        // Try to load existing vector store first
        const vectorStoreLoaded = await this.loadVectorStore();

        if (!vectorStoreLoaded) {
          // Create new vector store if none exists
          if (this.dataset && this.dataset.length > 0) {
            logger.info("Creating new vector store from dataset");
            await this.updateVectorStore();
          } else {
            logger.warn("No dataset available to create vector store");
          }
        }
      } catch (ollamaError) {
        logger.error(
          `Failed to initialize Ollama embeddings: ${ollamaError.message}`,
          {
            stack: ollamaError.stack,
            ollamaBaseUrl: this.ollamaBaseUrl,
            embeddingsModel: this.embeddingsModel,
          }
        );
        logger.warn(
          "BrainService will operate in fallback mode without vector search"
        );
      }

      this.isInitialized = true;
      logger.info("BrainService initialized successfully", {
        datasetSize: this.dataset ? this.dataset.length : 0,
        vectorStoreAvailable: !!this.vectorStore,
      });
    } catch (error) {
      logger.error(`Failed to initialize BrainService: ${error.message}`, {
        stack: error.stack,
      });
      // Don't throw error, allow service to start in degraded mode
      this.isInitialized = true;
    }
  }

  /**
   * Load dataset from file
   */
  async loadDataset() {
    try {
      const fullPath = path.resolve(this.datasetPath);
      logger.info("Loading dataset", { path: fullPath });

      // Check if dataset file exists
      try {
        await fs.access(fullPath);
        const startTime = Date.now();
        const data = await fs.readFile(fullPath, "utf8");
        const parseStartTime = Date.now();
        this.dataset = JSON.parse(data);
        const parseEndTime = Date.now();

        // Extract unique tags for logging
        const uniqueTags = vectorUtils.extractUniqueTags(this.dataset);

        logger.info("Dataset loaded successfully", {
          path: fullPath,
          entries: this.dataset.length,
          sizeBytes: data.length,
          readTimeMs: parseStartTime - startTime,
          parseTimeMs: parseEndTime - parseStartTime,
          uniqueTagsCount: uniqueTags.length,
          uniqueTags,
        });
      } catch (fileError) {
        logger.warn(`Dataset file not found or invalid at ${fullPath}`, {
          error: fileError.message,
          usingDefault: true,
        });
        this.dataset = this.getDefaultDataset();
        logger.info("Using default dataset", { entries: this.dataset.length });
      }
    } catch (error) {
      logger.error(`Error loading dataset`, {
        error: error.message,
        stack: error.stack,
        path: this.datasetPath,
        usingDefault: true,
      });
      this.dataset = this.getDefaultDataset();
      logger.info("Using default dataset", { entries: this.dataset.length });
    }
  }

  /**
   * Get default dataset for training
   */
  getDefaultDataset() {
    return [
      {
        question: "Hi, good morning!",
        answer: "Good morning! How can I assist you today?",
        tags: ["greeting"],
      },
      {
        question: "How much does this product cost?",
        answer:
          "I'll help you check the price. Could you please specify which product you're interested in?",
        tags: ["price_inquiry"],
      },
      {
        question: "Do you have this item in stock?",
        answer:
          "I'll check the availability for you. Which specific item are you looking for?",
        tags: ["availability"],
      },
      {
        question: "Can you help me with my order?",
        answer:
          "Of course! I'm here to help. What kind of assistance do you need with your order?",
        tags: ["order"],
      },
    ];
  }

  /**
   * Load vector store from disk
   */
  async loadVectorStore() {
    try {
      logger.info("Attempting to load vector store", {
        path: this.vectorStorePath,
      });

      // Check if vector store directory exists
      await fs.access(this.vectorStorePath);

      // Load vector store from disk
      const startTime = Date.now();
      this.vectorStore = await FaissStore.load(
        this.vectorStorePath,
        this.embeddings
      );
      const loadTime = Date.now() - startTime;

      // Get vector store stats
      const stats = this.getStats();

      logger.info("Vector store loaded successfully", {
        path: this.vectorStorePath,
        loadTimeMs: loadTime,
        ...stats,
      });
      return true;
    } catch (error) {
      logger.info("No existing vector store found, will create new one", {
        path: this.vectorStorePath,
        error: error.message,
      });
      return false;
    }
  }

  /**
   * Update vector store with current dataset
   */
  async updateVectorStore() {
    try {
      logger.info('Updating vector store with current dataset');
      // Convert dataset to documents for vector store
      const startTime = Date.now();
      const documents = vectorUtils.datasetToDocuments(this.dataset);
      const conversionTime = Date.now() - startTime;
      logger.info('Dataset converted to documents', { 
        documentsCount: documents.length,
        conversionTimeMs: conversionTime 
      });
      if (documents.length === 0) {
        logger.warn('No documents to create vector store', { datasetLength: this.dataset.length });
        return false;
      }
      // Batching embedding to avoid client closing connection
      const batchSize = 8; // You can increase/decrease as needed
      let allStores = [];
      for (const batch of chunkArray(documents, batchSize)) {
        logger.info('Embedding batch', { batchSize: batch.length });
        const store = await FaissStore.fromDocuments(batch, this.embeddings);
        allStores.push(store);
      }
      // Merge all stores if more than one batch
      if (allStores.length === 1) {
        this.vectorStore = allStores[0];
      } else {
        // Merge all vector stores
        this.vectorStore = allStores[0];
        for (let i = 1; i < allStores.length; i++) {
          await this.vectorStore.mergeFrom(allStores[i]);
        }
      }
      const createTime = Date.now() - startTime;
      logger.info('Vector store created', { 
        documentsCount: documents.length,
        createTimeMs: createTime 
      });
      // Save vector store to disk
      await this.saveVectorStore();
      // Get vector store stats
      const stats = this.getStats();
      logger.info('Vector store updated successfully', { 
        documentsCount: documents.length,
        totalTimeMs: Date.now() - startTime,
        ...stats
      });
      return true;
    } catch (error) {
      logger.error('Error updating vector store', {
        error: error.message,
        stack: error.stack,
        datasetLength: this.dataset ? this.dataset.length : 0
      });
      return false;
    }
  }

  /**
   * Save vector store to disk
   */
  async saveVectorStore() {
    try {
      if (!this.vectorStore) {
        logger.warn("No vector store to save");
        return false;
      }

      logger.info("Saving vector store to disk", {
        path: this.vectorStorePath,
      });

      // Ensure directory exists
      await fs.mkdir(this.vectorStorePath, { recursive: true });

      // Save vector store to disk
      const startTime = Date.now();
      await this.vectorStore.save(this.vectorStorePath);
      const saveTime = Date.now() - startTime;

      // Get vector store stats
      const stats = this.getStats();

      logger.info("Vector store saved successfully", {
        path: this.vectorStorePath,
        saveTimeMs: saveTime,
        ...stats,
      });
      return true;
    } catch (error) {
      logger.error(`Error saving vector store`, {
        error: error.message,
        stack: error.stack,
        path: this.vectorStorePath,
      });
      return false;
    }
  }

  /**
   * Process context for a given question
   * @param {string} question - User question
   * @param {number} maxResults - Maximum number of results to return (default: 32)
   * @returns {Array<Object>} - Array of context entries
   */
  async processContext(question, maxResults = 32) {
    try {
      if (!this.isInitialized) {
        await this.init();
      }

      // If vector store is not available, use fallback method
      if (!this.vectorStore || !this.embeddings) {
        logger.warn("Vector store not available, using fallback method", {
          question,
        });
        return this.getFallbackContext(question, maxResults);
      }

      // Prepare question for embedding
      const preparedQuestion = vectorUtils.prepareTextForEmbedding(question);
      logger.info("Processing context for question", {
        question,
        preparedQuestion,
        maxResults,
      });

      try {
        // Search for similar documents
        const startTime = Date.now();
        const results = await this.vectorStore.similaritySearch(
          preparedQuestion,
          maxResults
        );
        const searchTime = Date.now() - startTime;

        logger.info("Context search completed", {
          resultsCount: results.length,
          searchTimeMs: searchTime,
        });

        // Extract context from results
        return results.map((result) => ({
          question: result.metadata.question,
          answer: result.metadata.answer,
          tags: result.metadata.tags,
          similarity: result._distance || 0, // Some implementations include distance
        }));
      } catch (searchError) {
        logger.error(`Error in vector search: ${searchError.message}`, {
          question,
          stack: searchError.stack,
        });
        // Fallback to basic search if vector search fails
        return this.getFallbackContext(question, maxResults);
      }
    } catch (error) {
      logger.error(`Error processing context: ${error.message}`, {
        question,
        stack: error.stack,
      });
      return [];
    }
  }

  /**
   * Get fallback context when vector store is not available
   * @param {string} question - User question
   * @param {number} maxResults - Maximum number of results to return
   * @returns {Array<Object>} - Array of context entries
   */
  getFallbackContext(question, maxResults = 32) {
    try {
      if (
        !this.dataset ||
        !Array.isArray(this.dataset) ||
        this.dataset.length === 0
      ) {
        logger.warn("No dataset available for fallback context", { question });
        return [];
      }

      logger.info("Using fallback context method", { question, maxResults });

      // Normalize question for comparison
      const normalizedQuestion = textUtils
        .normalizeText(question)
        .toLowerCase();

      // Simple text similarity search
      const results = this.dataset
        .map((entry) => {
          const normalizedEntryQuestion = textUtils
            .normalizeText(entry.question)
            .toLowerCase();
          const similarity = textUtils.calculateJaccardSimilarity(
            normalizedQuestion,
            normalizedEntryQuestion
          );
          return { ...entry, similarity };
        })
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, maxResults);

      logger.info("Fallback context search completed", {
        resultsCount: results.length,
      });

      return results;
    } catch (error) {
      logger.error(`Error in fallback context: ${error.message}`, {
        question,
        stack: error.stack,
      });
      return [];
    }
  }

  /**
   * Get predicted tags for a question
   * @param {string} question - User question
   * @returns {Array<string>} - Array of predicted tags
   */
  async getPredictedTags(question) {
    try {
      if (!this.isInitialized) {
        await this.init();
      }

      logger.info("Getting predicted tags for question", { question });

      // Get context for question (using default 32 results)
      const context = await this.processContext(question);

      if (!context || !Array.isArray(context) || context.length === 0) {
        logger.warn("No context found for question", { question });
        return ["unknown"];
      }

      // Extract tags from context
      const tagCounts = {};

      // Count tag occurrences in context with similarity weighting
      for (const entry of context) {
        // Skip entries without tags
        if (!entry.tags || !Array.isArray(entry.tags)) {
          continue;
        }

        // Calculate weight based on similarity (higher similarity = higher weight)
        const weight = 1 - (entry.similarity || 0);

        for (const tag of entry.tags) {
          if (tag) {
            // Skip empty tags
            tagCounts[tag] = (tagCounts[tag] || 0) + weight;
          }
        }
      }

      // Sort tags by weighted count
      const sortedTags = Object.entries(tagCounts)
        .sort((a, b) => b[1] - a[1])
        .map((entry) => entry[0]);

      logger.info("Predicted tags", {
        question,
        tags: sortedTags,
        contextSize: context.length,
        topContextSimilarity: context.length > 0 ? context[0].similarity : null,
      });

      return sortedTags.length > 0 ? sortedTags : ["unknown"];
    } catch (error) {
      logger.error(`Error getting predicted tags: ${error.message}`, {
        question,
        stack: error.stack,
      });
      return ["unknown"];
    }
  }

  /**
   * Find answer for a question
   * @param {string} question - User question
   * @param {number} topResults - Number of top results to consider (default: 3)
   * @returns {string} - Best matching answer
   */
  async findAnswer(question, topResults = 3) {
    try {
      if (!this.isInitialized) {
        await this.init();
      }

      logger.info("Finding answer for question", { question, topResults });

      // Get context for question (using specified top results)
      const context = await this.processContext(question, topResults);

      if (!context || !Array.isArray(context) || context.length === 0) {
        logger.warn("No context found for question", { question });
        return "I don't have enough information to answer that question.";
      }

      // Get the best matching answer (highest similarity)
      const bestMatch = context[0];

      // Validate best match has required fields
      if (!bestMatch || !bestMatch.answer) {
        logger.warn("Invalid best match found", {
          question,
          bestMatch: bestMatch ? JSON.stringify(bestMatch) : "null",
        });
        return "I found some information but couldn't formulate a proper answer.";
      }

      logger.info("Found best matching answer", {
        question,
        matchedQuestion: bestMatch.question,
        similarity: bestMatch.similarity,
        tags: Array.isArray(bestMatch.tags) ? bestMatch.tags : [],
      });

      // Return best matching answer
      return bestMatch.answer;
    } catch (error) {
      logger.error(`Error finding answer: ${error.message}`, {
        question,
        stack: error.stack,
      });
      return "Sorry, I encountered an error while processing your question.";
    }
  }

  /**
   * Extract entities from text
   * @param {string} text - Text to extract entities from
   * @returns {Array<string>} - Array of extracted entities
   */
  extractEntities(text) {
    const productEntities = [
      "netflix",
      "spotify",
      "youtube",
      "disney",
      "canva",
      "vidio",
      "amazon",
      "hbo",
      "game pass",
      "chatgpt",
      "loklok",
      "prime",
      "viu",
      "wetv",
      "iqiyi",
      "mola tv",
      "apple music",
      "deezer",
      "tidal",
      "crunchyroll",
      "hulu",
      "paramount",
      "peacock",
    ];

    const normalizedText = textUtils.normalizeText(text).toLowerCase();
    const foundEntities = [];

    for (const entity of productEntities) {
      if (normalizedText.includes(entity)) {
        foundEntities.push(entity);
      }
    }

    return foundEntities;
  }

  /**
   * Calculate entity similarity
   * @param {string} entity1 - First entity
   * @param {string} entity2 - Second entity
   * @returns {number} - Similarity score
   */
  calculateEntitySimilarity(entity1, entity2) {
    const normalized1 = textUtils.normalizeText(entity1).toLowerCase();
    const normalized2 = textUtils.normalizeText(entity2).toLowerCase();

    // Exact match
    if (normalized1 === normalized2) {
      return 1.0;
    }

    // One is substring of the other
    if (
      normalized1.includes(normalized2) ||
      normalized2.includes(normalized1)
    ) {
      return 0.8;
    }

    // Levenshtein distance for fuzzy matching
    const distance = textUtils.levenshteinDistance(normalized1, normalized2);
    const maxLength = Math.max(normalized1.length, normalized2.length);

    if (maxLength === 0) return 0;

    const normalizedDistance = 1 - distance / maxLength;
    return normalizedDistance > 0.7 ? normalizedDistance : 0;
  }

  /**
   * Get vector store statistics
   * @returns {Object} - Vector store statistics
   */
  getStats() {
    return vectorUtils.getVectorStoreStats(this.vectorStore, this.dataset);
  }
}

// Helper function for batching
function chunkArray(array, size) {
  const result = [];
  for (let i = 0; i < array.length; i += size) {
    result.push(array.slice(i, i + size));
  }
  return result;
}

module.exports = BrainService;