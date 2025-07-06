/**
 * Question Routes Module
 * Handles all API routes related to question processing
 */

'use strict';

const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');
const QuestionProcessor = require('../services/QuestionProcessor');
const BrainService = require('../services/BrainService');
const OllamaService = require('../services/OllamaService');
const validateQuestion = require('../middleware/validateQuestion');

// Initialize services
const questionProcessor = new QuestionProcessor();
const brainService = new BrainService();
const ollamaService = new OllamaService();

// In-memory storage for question processing status
const processingStatus = new Map();

/**
 * Fastify plugin for question routes
 * @param {FastifyInstance} fastify - Fastify instance
 * @param {Object} options - Plugin options
 */
async function questionRoutes(fastify, options) {
  // Register routes
  registerAskRoute(fastify);
  registerStatusRoute(fastify);
  registerTestRoute(fastify);
}

/**
 * Register the main question asking endpoint
 * @param {FastifyInstance} fastify - Fastify instance
 */
function registerAskRoute(fastify) {
  fastify.post('/ask', async (request, reply) => {
    const questionId = uuidv4();
    const { question, whatsapp_number } = request.body;
    
    try {
      // Validate the question
      const validationResult = validateQuestion(request, reply, () => {});
      if (validationResult === false) return; // validateQuestion handles reply

      // Log and initialize processing status
      logger.info(`Processing question ${questionId}: ${question}`);
      initializeProcessingStatus(questionId, question);
      
      // Process the question asynchronously
      processQuestionAsync(questionId, question, whatsapp_number);
      
      // Send immediate response
      return reply.send({
        success: true,
        questionId,
        message: 'Question received and being processed',
        statusUrl: `/api/questions/status/${questionId}`
      });
    } catch (error) {
      logger.error(`Error initiating question processing: ${error.message}`);
      return reply.status(500).send({
        success: false,
        error: 'Failed to initiate question processing'
      });
    }
  });
}

/**
 * Register the status checking endpoint
 * @param {FastifyInstance} fastify - Fastify instance
 */
function registerStatusRoute(fastify) {
  fastify.get('/status/:questionId', async (request, reply) => {
    const { questionId } = request.params;
    const status = processingStatus.get(questionId);
    
    if (!status) {
      return reply.status(404).send({
        success: false,
        error: 'Question ID not found'
      });
    }
    
    return reply.send({
      success: true,
      ...status
    });
  });
}

/**
 * Register the Ollama test endpoint
 * @param {FastifyInstance} fastify - Fastify instance
 */
function registerTestRoute(fastify) {
  fastify.get('/ollama-test', async (request, reply) => {
    try {
      const result = await ollamaService.testConnection();
      return reply.send(result);
    } catch (error) {
      return reply.status(500).send({ 
        success: false, 
        error: error.message 
      });
    }
  });
}

/**
 * Initialize the processing status for a new question
 * @param {string} questionId - UUID of the question
 * @param {string} question - The question text
 */
function initializeProcessingStatus(questionId, question) {
  processingStatus.set(questionId, {
    status: 'processing',
    stage: 'initializing',
    startTime: new Date(),
    question
  });
}

/**
 * Process a question asynchronously through the complete pipeline
 * @param {string} questionId - UUID of the question
 * @param {string} question - The question text
 * @param {string} whatsapp_number - WhatsApp number for response delivery
 */
async function processQuestionAsync(questionId, question, whatsapp_number) {
  try {
    // Step 1: Get context from dataset
    updateStatus(questionId, 'getting_context', 'Retrieving context from dataset');
    const datasetContext = await brainService.processContext(question);
    
    // Step 2: Analyze and tag the question
    updateStatus(questionId, 'tagging', 'Analyzing and tagging question');
    const tags = await brainService.tagQuestion(question, datasetContext);
    
    // Step 3: Process with AI model
    updateStatus(questionId, 'processing_ai', 'Processing with AI model');
    const response = await questionProcessor.processQuestion(question, whatsapp_number);
    
    // Step 4: Mark as completed
    const processingTime = Date.now() - processingStatus.get(questionId).startTime;
    updateStatus(questionId, 'completed', 'Processing completed', {
      response,
      tags,
      processingTime
    });
    
    logger.info(`Question ${questionId} processed successfully in ${processingTime}ms`);
  } catch (error) {
    logger.error(`Error processing question ${questionId}: ${error.message}`);
    updateStatus(questionId, 'error', error.message);
  }
}

/**
 * Update the processing status of a question
 * @param {string} questionId - UUID of the question
 * @param {string} stage - Current processing stage
 * @param {string} message - Status message
 * @param {Object} additionalData - Additional data to include in status
 */
function updateStatus(questionId, stage, message, additionalData = {}) {
  const currentStatus = processingStatus.get(questionId);
  if (currentStatus) {
    processingStatus.set(questionId, {
      ...currentStatus,
      stage,
      message,
      lastUpdated: new Date(),
      ...additionalData
    });
  }
}

// Set up automatic cleanup of old status entries
setupStatusCleanup();

/**
 * Set up periodic cleanup of old processing status entries
 * Runs every hour to remove entries older than one hour
 */
function setupStatusCleanup() {
  const ONE_HOUR_MS = 60 * 60 * 1000;
  
  setInterval(() => {
    const oneHourAgo = Date.now() - ONE_HOUR_MS;
    let cleanupCount = 0;
    
    for (const [questionId, status] of processingStatus.entries()) {
      if (status.startTime < oneHourAgo) {
        processingStatus.delete(questionId);
        cleanupCount++;
      }
    }
    
    if (cleanupCount > 0) {
      logger.debug(`Cleaned up ${cleanupCount} old question status entries`);
    }
  }, ONE_HOUR_MS);
}

module.exports = questionRoutes;