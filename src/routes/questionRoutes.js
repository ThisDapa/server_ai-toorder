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
const { askSchema } = require('../schemas/askSchemas');

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
  registerProductsRoute(fastify);
}

/**
 * Register the main question asking endpoint
 * @param {FastifyInstance} fastify - Fastify instance
 */
function registerAskRoute(fastify) {
  fastify.post('/ask', { schema: { body: askSchema } }, async (request, reply) => {
    const questionId = uuidv4();
    const { name_store, question, whatsapp_number, product_data } = request.body;
    
    try {
      // Validate the question
      const validationResult = validateQuestion(request, reply, () => {});
      if (validationResult === false) return;

      // Log and initialize processing status
      logger.info(`Processing question ${questionId}: ${question}`);
      initializeProcessingStatus(questionId, question);
      
      // Process the question asynchronously
      processQuestionAsync(questionId, name_store, question, whatsapp_number, product_data);
      
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
 * @param {string} name_store
 * @param {string} question - The question text
 * @param {string} whatsapp_number - WhatsApp number for response delivery
 * @param {Object} product_data - Product data for context
 */
async function processQuestionAsync(questionId,name_store , question, whatsapp_number, product_data) {
  try {
    // Step 1: Process with AI model
    updateStatus(questionId, 'processing_ai', 'Processing with AI model');
    const response = await questionProcessor.processQuestion(name_store, question, whatsapp_number, product_data);
    
    // Step 2: Mark as completed
    const processingTime = Date.now() - processingStatus.get(questionId).startTime;
    updateStatus(questionId, 'completed', 'Processing completed', {
      response,
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

/**
 * Register product routes
 * @param {FastifyInstance} fastify - Fastify instance
 * 
 * API Endpoints:
 * - GET /products - Get all products
 * - GET /products/search?keyword=xyz - Search products by keyword
 * - GET /products/:identifier - Get product by code or name
 * - POST /products - Add a new product
 * - PUT /products/:productCode - Update product information
 * - PATCH /products/stock/:productCode - Update product stock
 * - DELETE /products/:identifier - Delete a product
 */
function registerProductsRoute(fastify) {
  // Get all products
  fastify.get('/products', async (request, reply) => {
    try {
      // Ensure OllamaService is initialized
      if (!ollamaService.initialized) {
        await ollamaService.init();
      }
      
      // Get product data from OllamaService
      const products = ollamaService.getProductData();
      
      // Return formatted product data
      return reply.send({
        success: true,
        data: products,
        count: Object.keys(products).length,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      logger.error(`Error retrieving products: ${error.message}`);
      return reply.status(500).send({ 
        success: false, 
        error: 'Failed to retrieve products'
      });
    }
  });
  

  
  // Get product by code or name
  fastify.get('/products/:identifier', async (request, reply) => {
    try {
      const { identifier } = request.params;
      
      // Ensure OllamaService is initialized
      if (!ollamaService.initialized) {
        await ollamaService.init();
      }
      
      // Get all products
      const allProducts = ollamaService.getProductData();
      
      // Find product by code (case insensitive)
      let product = null;
      let productName = null;
      
      // First try to find by product code
      for (const [name, details] of Object.entries(allProducts)) {
        if (details.code.toLowerCase() === identifier.toLowerCase()) {
          product = details;
          productName = name;
          break;
        }
      }
      
      // If not found by code, try to find by product name
      if (!product) {
        for (const [name, details] of Object.entries(allProducts)) {
          if (name.toLowerCase().includes(identifier.toLowerCase())) {
            product = details;
            productName = name;
            break;
          }
        }
      }
      
      // Return product if found
      if (product) {
        return reply.send({
          success: true,
          productName,
          product,
          timestamp: new Date().toISOString()
        });
      } else {
        return reply.status(404).send({
          success: false,
          error: `Product with identifier '${identifier}' not found`
        });
      }
    } catch (error) {
      logger.error(`Error retrieving product: ${error.message}`);
      return reply.status(500).send({ 
        success: false, 
        error: 'Failed to retrieve product'
      });
    }
  });
  
  // Update product stock
  fastify.patch('/products/stock/:productCode', async (request, reply) => {
    try {
      const { productCode } = request.params;
      const { stock } = request.body;
      
      // Validate request body
      if (stock === undefined) {
        return reply.status(400).send({
          success: false,
          error: 'Stock value is required in request body'
        });
      }
      
      // Ensure OllamaService is initialized
      if (!ollamaService.initialized) {
        await ollamaService.init();
      }
      
      // Update product stock
      const result = ollamaService.updateProductStock(productCode, stock);
      
      if (result.success) {
        return reply.send({
          ...result,
          timestamp: new Date().toISOString()
        });
      } else {
        return reply.status(404).send(result);
      }
    } catch (error) {
      logger.error(`Error updating product stock: ${error.message}`);
      return reply.status(500).send({ 
        success: false, 
        error: 'Failed to update product stock'
      });
    }
  });
  
  // Add new product
  fastify.post('/products', async (request, reply) => {
    try {
      const { name, code, price, stock, description } = request.body;
      
      // Validate request body
      if (!name || !code || !price) {
        return reply.status(400).send({
          success: false,
          error: 'Product name, code, and price are required in request body'
        });
      }
      
      // Ensure OllamaService is initialized
      if (!ollamaService.initialized) {
        await ollamaService.init();
      }
      
      // Add new product
      const result = ollamaService.addProduct(name, code, price, stock, description);
      
      if (result.success) {
        return reply.status(201).send({
          ...result,
          timestamp: new Date().toISOString()
        });
      } else {
        return reply.status(400).send(result);
      }
    } catch (error) {
      logger.error(`Error adding new product: ${error.message}`);
      return reply.status(500).send({ 
        success: false, 
        error: 'Failed to add new product'
      });
    }
  });
  
  // Delete product
  fastify.delete('/products/:identifier', async (request, reply) => {
    try {
      const { identifier } = request.params;
      
      if (!identifier) {
        return reply.status(400).send({
          success: false,
          error: 'Product identifier is required'
        });
      }
      
      // Ensure OllamaService is initialized
      if (!ollamaService.initialized) {
        await ollamaService.init();
      }
      
      // Remove product
      const result = ollamaService.removeProduct(identifier);
      
      if (result.success) {
        return reply.send({
          ...result,
          timestamp: new Date().toISOString()
        });
      } else {
        return reply.status(404).send(result);
      }
    } catch (error) {
      logger.error(`Error removing product: ${error.message}`);
      return reply.status(500).send({ 
        success: false, 
        error: 'Failed to remove product'
      });
    }
  });
  
  // Update product information
  fastify.put('/products/:productCode', async (request, reply) => {
    try {
      const { productCode } = request.params;
      const { price, description } = request.body;
      
      if (!productCode) {
        return reply.status(400).send({
          success: false,
          error: 'Product code is required'
        });
      }
      
      // Validate that at least one update field is provided
      if (price === undefined && description === undefined) {
        return reply.status(400).send({
          success: false,
          error: 'At least one field to update (price or description) is required'
        });
      }
      
      // Ensure OllamaService is initialized
      if (!ollamaService.initialized) {
        await ollamaService.init();
      }
      
      // Update product
      const updateData = {};
      if (price !== undefined) updateData.price = price;
      if (description !== undefined) updateData.description = description;
      
      const result = ollamaService.updateProduct(productCode, updateData);
      
      if (result.success) {
        return reply.send({
          ...result,
          timestamp: new Date().toISOString()
        });
      } else {
        return reply.status(404).send(result);
      }
    } catch (error) {
      logger.error(`Error updating product: ${error.message}`);
      return reply.status(500).send({ 
        success: false, 
        error: 'Failed to update product'
      });
    }
  });
  
  // Search products
  fastify.get('/products/search', async (request, reply) => {
    try {
      const { keyword } = request.query;
      
      if (!keyword || keyword.trim() === '') {
        return reply.status(400).send({
          success: false,
          error: 'Search keyword is required'
        });
      }
      
      // Ensure OllamaService is initialized
      if (!ollamaService.initialized) {
        await ollamaService.init();
      }
      
      // Search products
      const result = ollamaService.searchProducts(keyword);
      
      if (result.success) {
        return reply.send({
          ...result,
          timestamp: new Date().toISOString()
        });
      } else {
        // Return 200 status even for no results, but with success: false
        return reply.send({
          ...result,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      logger.error(`Error searching products: ${error.message}`);
      return reply.status(500).send({ 
        success: false, 
        error: 'Failed to search products'
      });
    }
  });
}

module.exports = questionRoutes;