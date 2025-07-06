/**
 * Logger Module
 * Centralized logging configuration using Winston
 */

'use strict';

const winston = require('winston');
const path = require('path');
const fs = require('fs');

// Configuration constants
const CONFIG = {
  LOG_LEVEL: process.env.LOG_LEVEL || 'info',
  SERVICE_NAME: 'ai-server',
  MAX_FILE_SIZE: 5242880, // 5MB
  MAX_FILES: 5,
  TIMESTAMP_FORMAT: 'YYYY-MM-DD HH:mm:ss',
  CONSOLE_TIMESTAMP_FORMAT: 'HH:mm:ss'
};

// File paths
const PATHS = {
  LOGS_DIR: path.join(process.cwd(), 'logs'),
  ERROR_LOG: 'error.log',
  COMBINED_LOG: 'combined.log',
  EXCEPTIONS_LOG: 'exceptions.log',
  REJECTIONS_LOG: 'rejections.log'
};

/**
 * Initialize logging directory
 */
function initializeLogsDirectory() {
  if (!fs.existsSync(PATHS.LOGS_DIR)) {
    fs.mkdirSync(PATHS.LOGS_DIR, { recursive: true });
  }
}

/**
 * Create file format for logs
 * @returns {winston.Logform.Format} Configured Winston format
 */
function createFileFormat() {
  return winston.format.combine(
    winston.format.timestamp({
      format: CONFIG.TIMESTAMP_FORMAT
    }),
    winston.format.errors({ stack: true }),
    winston.format.json(),
    winston.format.prettyPrint()
  );
}

/**
 * Create console format for development
 * @returns {winston.Logform.Format} Configured Winston format for console
 */
function createConsoleFormat() {
  return winston.format.combine(
    winston.format.colorize(),
    winston.format.timestamp({
      format: CONFIG.CONSOLE_TIMESTAMP_FORMAT
    }),
    winston.format.printf(formatConsoleOutput)
  );
}

/**
 * Format console output
 * @param {Object} info - Log information
 * @returns {string} Formatted log message
 */
function formatConsoleOutput({ timestamp, level, message, ...meta }) {
  let msg = `${timestamp} [${level}]: ${message}`;
  if (Object.keys(meta).length > 0) {
    msg += ` ${JSON.stringify(meta)}`;
  }
  return msg;
}

/**
 * Create file transport for logs
 * @param {string} filename - Log file name
 * @param {string} [level] - Log level
 * @returns {winston.transport} Winston file transport
 */
function createFileTransport(filename, level) {
  const options = {
    filename: path.join(PATHS.LOGS_DIR, filename),
    maxsize: CONFIG.MAX_FILE_SIZE,
    maxFiles: CONFIG.MAX_FILES
  };
  
  if (level) {
    options.level = level;
  }
  
  return new winston.transports.File(options);
}

// Initialize logs directory
initializeLogsDirectory();

// Create logger instance
const logger = winston.createLogger({
  level: CONFIG.LOG_LEVEL,
  format: createFileFormat(),
  defaultMeta: { service: CONFIG.SERVICE_NAME },
  transports: [
    createFileTransport(PATHS.ERROR_LOG, 'error'),
    createFileTransport(PATHS.COMBINED_LOG)
  ],
  exceptionHandlers: [
    createFileTransport(PATHS.EXCEPTIONS_LOG)
  ],
  rejectionHandlers: [
    createFileTransport(PATHS.REJECTIONS_LOG)
  ]
});

// Add console transport for development
if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: createConsoleFormat()
  }));
}

/**
 * Log HTTP request details
 * @param {Object} req - Request object
 * @param {Object} res - Response object
 * @param {number} responseTime - Response time in milliseconds
 */
logger.logRequest = (req, res, responseTime) => {
  const logData = {
    method: req.method,
    url: req.url,
    statusCode: res.statusCode,
    responseTime: `${responseTime}ms`,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  };
  
  const logLevel = res.statusCode >= 400 ? 'warn' : 'info';
  logger[logLevel]('HTTP Request', logData);
};

/**
 * Log AI processing stages
 * @param {string} questionId - Question identifier
 * @param {string} stage - Processing stage
 * @param {Object} data - Additional data
 */
logger.logAIProcess = (questionId, stage, data = {}) => {
  logger.info(`AI Processing [${questionId}] - ${stage}`, data);
};

/**
 * Log application errors
 * @param {Error} error - Error object
 * @param {Object} context - Additional context
 */
logger.logError = (error, context = {}) => {
  logger.error('Application Error', {
    message: error.message,
    stack: error.stack,
    ...context
  });
};

module.exports = logger;