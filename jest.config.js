module.exports = {
  testEnvironment: 'node',
  
  testMatch: [
    '**/test/**/*.test.js',
    '**/tests/**/*.test.js',
    '**/__tests__/**/*.js'
  ],
  
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/**/*.test.js',
    '!**/node_modules/**'
  ],
  
  setupFilesAfterEnv: ['<rootDir>/test/setup.js'],
  
  testTimeout: 30000,
  
  verbose: true,
  
  clearMocks: true,
  
  forceExit: true,
  
  detectOpenHandles: true
};