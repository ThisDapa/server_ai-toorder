const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');
const QuestionProcessor = require('../services/QuestionProcessor');
const BrainService = require('../services/BrainService');
const OllamaService = require('../services/OllamaService');
const validateQuestion = require('../middleware/validateQuestion');

const questionProcessor = new QuestionProcessor();
const processingStatus = new Map();
const brainService = new BrainService();
const ollamaService = new OllamaService();

/**
 * Fastify plugin for question routes
 */
async function questionRoutes(fastify, opts) {
  fastify.post('/ask', async (request, reply) => {
    const questionId = uuidv4();
    const { question, whatsapp_number } = request.body;
    try {
      // Manual validation (since no Express middleware)
      const validationResult = validateQuestion(request, reply, () => {});
      if (validationResult === false) return; // validateQuestion handles reply

      logger.info(`Processing question ${questionId}: ${question}`);
      processingStatus.set(questionId, {
        status: 'processing',
        stage: 'initializing',
        startTime: new Date(),
        question
      });
      processQuestionAsync(questionId, question, whatsapp_number);
      reply.send({
        success: true,
        questionId,
        message: 'Question received and being processed',
        statusUrl: `/api/questions/status/${questionId}`
      });
    } catch (error) {
      logger.error(`Error initiating question processing: ${error.message}`);
      reply.status(500).send({
        success: false,
        error: 'Failed to initiate question processing'
      });
    }
  });

  // GET /api/questions/status/:questionId
  fastify.get('/status/:questionId', async (request, reply) => {
    const { questionId } = request.params;
    const status = processingStatus.get(questionId);
    if (!status) {
      return reply.status(404).send({
        success: false,
        error: 'Question ID not found'
      });
    }
    reply.send({
      success: true,
      ...status
    });
  });

  // GET /api/questions/ollama-test
  fastify.get('/ollama-test', async (request, reply) => {
    try {
      const result = await ollamaService.testConnection();
      reply.send(result);
    } catch (error) {
      reply.status(500).send({ success: false, error: error.message });
    }
  });
}

// Async function to process question through the complete pipeline
async function processQuestionAsync(questionId, question, whatsapp_number) {
  try {
    updateStatus(questionId, 'getting_context', 'Retrieving context from dataset');
    const datasetContext = await brainService.processContext(question);
    updateStatus(questionId, 'tagging', 'Analyzing and tagging question');
    const tags = await brainService.tagQuestion(question, datasetContext);
    updateStatus(questionId, 'processing_ai', 'Processing with AI model');
    const response = await questionProcessor.processQuestion(question, whatsapp_number);
    updateStatus(questionId, 'completed', 'Processing completed', {
      response,
      tags,
      processingTime: Date.now() - processingStatus.get(questionId).startTime
    });
    logger.info(`Question ${questionId} processed successfully`);
  } catch (error) {
    logger.error(`Error processing question ${questionId}: ${error.message}`);
    updateStatus(questionId, 'error', error.message);
  }
}

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

// Cleanup old statuses (run every hour)
setInterval(() => {
  const oneHourAgo = Date.now() - (60 * 60 * 1000);
  for (const [questionId, status] of processingStatus.entries()) {
    if (status.startTime < oneHourAgo) {
      processingStatus.delete(questionId);
    }
  }
}, 60 * 60 * 1000);

module.exports = questionRoutes;