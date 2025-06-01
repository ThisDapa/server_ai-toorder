/**
 * Jest setup file
 * This file runs before all tests
 */

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.PORT = '3001'; // Use different port for testing
process.env.LOG_LEVEL = 'error'; // Reduce log noise during tests
process.env.OLLAMA_BASE_URL = 'http://localhost:11434';
process.env.OLLAMA_MODEL = 'llama2';
process.env.DATASET_PATH = './data/dataset.json';

// Increase timeout for async operations
jest.setTimeout(30000);

// Mock console methods to reduce noise during tests
const originalConsoleError = console.error;
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;

beforeAll(() => {
  // Mock console methods but allow important messages
  console.error = jest.fn((message) => {
    if (typeof message === 'string' && message.includes('ECONNREFUSED')) {
      // Allow connection error messages (expected in test environment)
      return;
    }
    originalConsoleError(message);
  });
  
  console.warn = jest.fn();
  
  console.log = jest.fn((message) => {
    // Allow specific log messages that are important for tests
    if (typeof message === 'string' && 
        (message.includes('Test') || message.includes('ERROR'))) {
      originalConsoleLog(message);
    }
  });
});

afterAll(() => {
  // Restore original console methods
  console.error = originalConsoleError;
  console.warn = originalConsoleWarn;
  console.log = originalConsoleLog;
});

// Global test helpers
global.testHelpers = {
  // Helper to create valid question data
  createValidQuestionData: (question = 'Test question', context = {}) => ({
    question,
    context: {
      user_id: 'test_user',
      session_id: 'test_session',
      timestamp: new Date().toISOString(),
      ...context
    }
  }),
  
  // Helper to wait for async operations
  wait: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
  
  // Helper to generate random string
  randomString: (length = 10) => {
    const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  }
};

// Handle unhandled promise rejections in tests
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Handle uncaught exceptions in tests
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});