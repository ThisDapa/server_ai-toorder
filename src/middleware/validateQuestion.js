/**
 * Question Validation Middleware
 * Validates and sanitizes incoming question requests
 */

'use strict';

const logger = require('../utils/logger');

// Constants for validation
const MAX_QUESTION_LENGTH = 1000;
const ERROR_CODES = {
  MISSING_QUESTION: 'MISSING_QUESTION',
  INVALID_QUESTION_TYPE: 'INVALID_QUESTION_TYPE',
  EMPTY_QUESTION: 'EMPTY_QUESTION',
  QUESTION_TOO_LONG: 'QUESTION_TOO_LONG',
  INVALID_CONTEXT_TYPE: 'INVALID_CONTEXT_TYPE'
};

/**
 * Middleware to validate incoming questions
 * @param {Object} req - Request object
 * @param {Object} res - Response object
 * @param {Function} next - Next middleware function
 * @returns {Boolean|undefined} - Returns false if validation fails, undefined otherwise
 */
const validateQuestion = (req, res, next) => {
  const { question, context } = req.body;
  
  // Validate question existence
  if (!question) {
    return sendValidationError(res, 
      'Question is required', 
      ERROR_CODES.MISSING_QUESTION, 
      'Missing question'
    );
  }
  
  // Validate question type
  if (typeof question !== 'string') {
    return sendValidationError(res, 
      'Question must be a string', 
      ERROR_CODES.INVALID_QUESTION_TYPE, 
      'Invalid question type'
    );
  }
  
  // Validate question is not empty
  if (question.trim().length === 0) {
    return sendValidationError(res, 
      'Question cannot be empty', 
      ERROR_CODES.EMPTY_QUESTION, 
      'Empty question'
    );
  }
  
  // Validate question length
  if (question.length > MAX_QUESTION_LENGTH) {
    return sendValidationError(res, 
      `Question is too long (max ${MAX_QUESTION_LENGTH} characters)`, 
      ERROR_CODES.QUESTION_TOO_LONG, 
      'Question too long'
    );
  }
  
  // Validate context if provided
  if (context && typeof context !== 'object') {
    return sendValidationError(res, 
      'Context must be an object', 
      ERROR_CODES.INVALID_CONTEXT_TYPE, 
      'Invalid context type'
    );
  }
  
  // Sanitize question
  req.body.question = sanitizeQuestion(question);
  
  // Log successful validation
  logSuccessfulValidation(question);
  
  // Continue to next middleware
  next();
};

/**
 * Send a validation error response
 * @param {Object} res - Response object
 * @param {string} errorMessage - User-facing error message
 * @param {string} errorCode - Error code for programmatic handling
 * @param {string} logMessage - Message to log
 * @returns {boolean} - Always returns false to indicate validation failure
 */
function sendValidationError(res, errorMessage, errorCode, logMessage) {
  logger.warn(`Question validation failed: ${logMessage}`);
  res.status(400).json({
    success: false,
    error: errorMessage,
    code: errorCode
  });
  return false;
}

/**
 * Sanitize the question string
 * @param {string} question - The question to sanitize
 * @returns {string} - The sanitized question
 */
function sanitizeQuestion(question) {
  return question.trim();
}

/**
 * Log successful validation
 * @param {string} question - The validated question
 */
function logSuccessfulValidation(question) {
  const previewLength = 50;
  const questionPreview = question.length > previewLength
    ? `${question.substring(0, previewLength)}...`
    : question;
  
  logger.info(`Question validated successfully: ${questionPreview}`);
}

module.exports = validateQuestion;