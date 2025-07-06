/**
 * Global Error Handler Middleware
 * Centralizes error handling for the application
 */

'use strict';

const logger = require('../utils/logger');

// Error type constants
const ERROR_TYPES = {
  VALIDATION: 'ValidationError',
  CAST: 'CastError',
  CONNECTION_REFUSED: 'ECONNREFUSED',
  TIMEOUT: 'TimeoutError'
};

// Error codes
const ERROR_CODES = {
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
  TIMEOUT: 'TIMEOUT',
  AI_SERVICE_ERROR: 'AI_SERVICE_ERROR',
  NEURAL_NETWORK_ERROR: 'NEURAL_NETWORK_ERROR'
};

// HTTP Status codes
const HTTP_STATUS = {
  BAD_REQUEST: 400,
  NOT_FOUND: 404,
  REQUEST_TIMEOUT: 408,
  INTERNAL_SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503
};

/**
 * Global error handling middleware
 * @param {Error} err - The error object
 * @param {Object} req - Request object
 * @param {Object} res - Response object
 * @param {Function} next - Next middleware function
 */
const errorHandler = (err, req, res, next) => {
  // Log the error with context
  logError(err, req);

  // Create base error response
  const error = createBaseErrorResponse();
  
  // Handle specific error types
  if (isValidationError(err)) {
    return handleValidationError(err, res, error);
  }
  
  if (isCastError(err)) {
    return handleCastError(res, error);
  }
  
  if (isConnectionRefusedError(err)) {
    return handleConnectionRefusedError(res, error);
  }
  
  if (isTimeoutError(err)) {
    return handleTimeoutError(res, error);
  }
  
  if (isOllamaError(err)) {
    return handleOllamaError(res, error);
  }
  
  if (isBrainError(err)) {
    return handleBrainError(res, error);
  }

  // Add development details if in development mode
  addDevelopmentDetails(err, error);

  // Return default 500 error
  return res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json(error);
};

/**
 * Log the error with request context
 * @param {Error} err - The error object
 * @param {Object} req - Request object
 */
function logError(err, req) {
  logger.error(`Error: ${err.message}`, {
    stack: err.stack,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });
}

/**
 * Create the base error response object
 * @returns {Object} Base error response
 */
function createBaseErrorResponse() {
  return {
    success: false,
    message: 'Internal server error',
    timestamp: new Date().toISOString()
  };
}

/**
 * Check if error is a validation error
 * @param {Error} err - The error object
 * @returns {boolean} True if validation error
 */
function isValidationError(err) {
  return err.name === ERROR_TYPES.VALIDATION;
}

/**
 * Handle validation error
 * @param {Error} err - The error object
 * @param {Object} res - Response object
 * @param {Object} error - Error response object
 * @returns {Object} HTTP response
 */
function handleValidationError(err, res, error) {
  error.message = 'Validation error';
  error.details = err.message;
  return res.status(HTTP_STATUS.BAD_REQUEST).json(error);
}

/**
 * Check if error is a cast error
 * @param {Error} err - The error object
 * @returns {boolean} True if cast error
 */
function isCastError(err) {
  return err.name === ERROR_TYPES.CAST;
}

/**
 * Handle cast error
 * @param {Object} res - Response object
 * @param {Object} error - Error response object
 * @returns {Object} HTTP response
 */
function handleCastError(res, error) {
  error.message = 'Invalid data format';
  return res.status(HTTP_STATUS.BAD_REQUEST).json(error);
}

/**
 * Check if error is a connection refused error
 * @param {Error} err - The error object
 * @returns {boolean} True if connection refused error
 */
function isConnectionRefusedError(err) {
  return err.code === ERROR_TYPES.CONNECTION_REFUSED;
}

/**
 * Handle connection refused error
 * @param {Object} res - Response object
 * @param {Object} error - Error response object
 * @returns {Object} HTTP response
 */
function handleConnectionRefusedError(res, error) {
  error.message = 'Service temporarily unavailable';
  error.code = ERROR_CODES.SERVICE_UNAVAILABLE;
  return res.status(HTTP_STATUS.SERVICE_UNAVAILABLE).json(error);
}

/**
 * Check if error is a timeout error
 * @param {Error} err - The error object
 * @returns {boolean} True if timeout error
 */
function isTimeoutError(err) {
  return err.name === ERROR_TYPES.TIMEOUT;
}

/**
 * Handle timeout error
 * @param {Object} res - Response object
 * @param {Object} error - Error response object
 * @returns {Object} HTTP response
 */
function handleTimeoutError(res, error) {
  error.message = 'Request timeout';
  error.code = ERROR_CODES.TIMEOUT;
  return res.status(HTTP_STATUS.REQUEST_TIMEOUT).json(error);
}

/**
 * Check if error is an Ollama-related error
 * @param {Error} err - The error object
 * @returns {boolean} True if Ollama error
 */
function isOllamaError(err) {
  return err.message && err.message.includes('ollama');
}

/**
 * Handle Ollama-related error
 * @param {Object} res - Response object
 * @param {Object} error - Error response object
 * @returns {Object} HTTP response
 */
function handleOllamaError(res, error) {
  error.message = 'AI service temporarily unavailable';
  error.code = ERROR_CODES.AI_SERVICE_ERROR;
  return res.status(HTTP_STATUS.SERVICE_UNAVAILABLE).json(error);
}

/**
 * Check if error is a Brain.js-related error
 * @param {Error} err - The error object
 * @returns {boolean} True if Brain.js error
 */
function isBrainError(err) {
  return err.message && err.message.includes('brain');
}

/**
 * Handle Brain.js-related error
 * @param {Object} res - Response object
 * @param {Object} error - Error response object
 * @returns {Object} HTTP response
 */
function handleBrainError(res, error) {
  error.message = 'Neural network processing error';
  error.code = ERROR_CODES.NEURAL_NETWORK_ERROR;
  return res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json(error);
}

/**
 * Add development-specific details to error response
 * @param {Error} err - The error object
 * @param {Object} error - Error response object
 */
function addDevelopmentDetails(err, error) {
  if (process.env.NODE_ENV === 'development') {
    error.stack = err.stack;
    error.details = err.message;
  }
}

module.exports = errorHandler;