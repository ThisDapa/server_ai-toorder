/**
 * QuestionProcessor Module
 * Handles the processing of questions through the AI pipeline
 * Integrates BrainService for context retrieval and OllamaService for AI processing
 */

'use strict';

const logger = require('../utils/logger');
const BrainService = require('./BrainService');
const OllamaService = require('./OllamaService');

/**
 * QuestionProcessor class
 * Coordinates the processing of questions through multiple AI services
 */
class QuestionProcessor {
  /**
   * Creates a new QuestionProcessor instance
   * Initializes required services and starts initialization process
   */
  constructor() {
    this.brainService = new BrainService();
    this.ollamaService = new OllamaService();
    this.isInitialized = false;
    this.init();
  }

  /**
   * Initializes all required services
   * @returns {Promise<void>} Resolves when initialization is complete
   * @throws {Error} If initialization fails
   */
  async init() {
    try {
      // Initialize services in parallel for better performance
      await Promise.all([
        this.brainService.init(),
        this.ollamaService.init()
      ]);

      this.isInitialized = true;
      logger.info('QuestionProcessor berhasil diinisialisasi');
    } catch (error) {
      logger.error(`Gagal menginisialisasi QuestionProcessor: ${error.message}`);
      throw error;
    }
  }

  /**
   * Processes a question through the complete AI pipeline
   * @param {string} question - The question to process
   * @param {string} number_whatsapp - WhatsApp number for context
   * @returns {Promise<Object>} The processed result with metadata
   * @throws {Error} If processing fails
   */
  async processQuestion(question, number_whatsapp) {
    try {
      await this.ensureInitialized();
      logger.info(`Memproses pertanyaan: ${question}`);
      
      const startTime = Date.now();
      
      // Step 1: Get context from BrainService
      const brainResult = await this.retrieveContext(question);
      
      // Step 2: Process with OllamaService using LangChain template
      const aiResult = await this.generateAIResponse(question, brainResult, number_whatsapp);
      
      // Step 3: Prepare and return the final result
      return this.prepareProcessingResult(aiResult, brainResult, startTime);
    } catch (error) {
      logger.error(`Error memproses pertanyaan: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Ensures the processor is initialized
   * @returns {Promise<void>}
   * @private
   */
  async ensureInitialized() {
    if (!this.isInitialized) {
      await this.init();
    }
  }
  
  /**
   * Retrieves context for a question using BrainService
   * @param {string} question - The question to get context for
   * @returns {Promise<Object>} Context information
   * @private
   */
  async retrieveContext(question) {
    return await this.brainService.processContext(question);
  }
  
  /**
   * Generates AI response using OllamaService
   * @param {string} question - The original question
   * @param {Object} brainResult - Context from BrainService
   * @param {string} number_whatsapp - WhatsApp number
   * @returns {Promise<Object>} AI processing result
   * @private
   */
  async generateAIResponse(question, brainResult, number_whatsapp) {
    return await this.ollamaService.processWithAI(
      question, 
      brainResult,
      number_whatsapp
    );
  }
  
  /**
   * Prepares the final processing result with metadata
   * @param {Object} aiResult - Result from AI processing
   * @param {Object} brainResult - Result from context retrieval
   * @param {number} startTime - Processing start timestamp
   * @returns {Object} Final result with metadata
   * @private
   */
  prepareProcessingResult(aiResult, brainResult, startTime) {
    const processingTime = Date.now() - startTime;
    logger.info(`Pertanyaan diproses dalam ${processingTime}ms`);
    
    return {
      ...aiResult,
      processingTime,
      brainRelevance: brainResult.brainRelevance,
      relevantEntriesCount: brainResult.relevantEntries.length
    };
  }

  /**
   * Gets processor statistics
   * @returns {Object} Statistics about the processor and its services
   */
  getStats() {
    return {
      isInitialized: this.isInitialized,
      brainService: this.brainService.getServiceStats(),
      ollamaService: this.ollamaService.getServiceStats()
    };
  }

  /**
   * Retrains the Brain.js network with new data
   * @param {Array|null} newData - Optional new training data
   * @returns {Promise<Object|boolean>} Training result or false if failed
   */
  async retrain(newData = null) {
    try {
      const result = await this.brainService.retrain(newData);
      logger.info('Jaringan berhasil dilatih ulang');
      return result;
    } catch (error) {
      logger.error(`Error melatih ulang jaringan: ${error.message}`);
      return false;
    }
  }

  /**
   * Tests the processor with a sample question
   * @returns {Promise<Object>} Test results
   */
  async test() {
    try {
      const testQuestion = "What is artificial intelligence?";
      const result = await this.processQuestion(testQuestion);
      logger.info('Pengujian prosesor berhasil diselesaikan');
      return {
        success: true,
        testQuestion,
        result
      };
    } catch (error) {
      logger.error(`Pengujian prosesor gagal: ${error.message}`);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Tests individual services
   * @returns {Promise<Object>} Test results for each service
   */
  async testServices() {
    try {
      const brainTest = await this.brainService.test();
      const ollamaTest = await this.ollamaService.testConnection();
      
      return {
        brain: brainTest,
        ollama: ollamaTest,
        overall: brainTest.success && ollamaTest.success
      };
    } catch (error) {
      logger.error(`Pengujian layanan gagal: ${error.message}`);
      return {
        brain: { success: false, error: error.message },
        ollama: { success: false, error: error.message },
        overall: false
      };
    }
  }
}

module.exports = QuestionProcessor;