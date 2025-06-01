const logger = require('../utils/logger');

/**
 * Middleware to validate incoming questions
 */
const validateQuestion = (req, res, next) => {
  const { question, context } = req.body;
  
  // Check if question exists
  if (!question) {
    logger.warn('Question validation failed: Missing question');
    return res.status(400).json({
      success: false,
      error: 'Question is required',
      code: 'MISSING_QUESTION'
    });
  }
  
  // Check question type
  if (typeof question !== 'string') {
    logger.warn('Question validation failed: Invalid question type');
    return res.status(400).json({
      success: false,
      error: 'Question must be a string',
      code: 'INVALID_QUESTION_TYPE'
    });
  }
  
  // Check question length
  if (question.trim().length === 0) {
    logger.warn('Question validation failed: Empty question');
    return res.status(400).json({
      success: false,
      error: 'Question cannot be empty',
      code: 'EMPTY_QUESTION'
    });
  }
  
  if (question.length > 1000) {
    logger.warn('Question validation failed: Question too long');
    return res.status(400).json({
      success: false,
      error: 'Question is too long (max 1000 characters)',
      code: 'QUESTION_TOO_LONG'
    });
  }
  
  // Validate context if provided
  if (context && typeof context !== 'object') {
    logger.warn('Question validation failed: Invalid context type');
    return res.status(400).json({
      success: false,
      error: 'Context must be an object',
      code: 'INVALID_CONTEXT_TYPE'
    });
  }
  
  // Sanitize question
  req.body.question = question.trim();
  
  logger.info(`Question validated successfully: ${question.substring(0, 50)}...`);
  next();
};

module.exports = validateQuestion;