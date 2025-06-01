const logger = require('../utils/logger');

/**
 * Global error handling middleware
 */
const errorHandler = (err, req, res, next) => {
  logger.error(`Error: ${err.message}`, {
    stack: err.stack,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });

  // Default error response
  let error = {
    success: false,
    message: 'Internal server error',
    timestamp: new Date().toISOString()
  };

  // Handle specific error types
  if (err.name === 'ValidationError') {
    error.message = 'Validation error';
    error.details = err.message;
    return res.status(400).json(error);
  }

  if (err.name === 'CastError') {
    error.message = 'Invalid data format';
    return res.status(400).json(error);
  }

  if (err.code === 'ECONNREFUSED') {
    error.message = 'Service temporarily unavailable';
    error.code = 'SERVICE_UNAVAILABLE';
    return res.status(503).json(error);
  }

  if (err.name === 'TimeoutError') {
    error.message = 'Request timeout';
    error.code = 'TIMEOUT';
    return res.status(408).json(error);
  }

  // Handle Ollama specific errors
  if (err.message && err.message.includes('ollama')) {
    error.message = 'AI service temporarily unavailable';
    error.code = 'AI_SERVICE_ERROR';
    return res.status(503).json(error);
  }

  // Handle Brain.js errors
  if (err.message && err.message.includes('brain')) {
    error.message = 'Neural network processing error';
    error.code = 'NEURAL_NETWORK_ERROR';
    return res.status(500).json(error);
  }

  // Development vs Production error details
  if (process.env.NODE_ENV === 'development') {
    error.stack = err.stack;
    error.details = err.message;
  }

  // Default 500 error
  res.status(500).json(error);
};

module.exports = errorHandler;