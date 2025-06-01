const request = require('supertest');
const app = require('../src/server');

describe('AI Server API Tests', () => {
  
  describe('Health Check', () => {
    test('GET /health should return OK status', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);
      
      expect(response.body.status).toBe('OK');
      expect(response.body.timestamp).toBeDefined();
      expect(response.body.uptime).toBeDefined();
    });
  });

  describe('Question Processing', () => {
    test('POST /api/questions/ask should accept valid question', async () => {
      const questionData = {
        question: 'What is artificial intelligence?',
        context: {
          user_id: 'test_user',
          session_id: 'test_session'
        }
      };

      const response = await request(app)
        .post('/api/questions/ask')
        .send(questionData)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.questionId).toBeDefined();
      expect(response.body.statusUrl).toBeDefined();
      expect(response.body.message).toContain('received and being processed');
    });

    test('POST /api/questions/ask should reject empty question', async () => {
      const questionData = {
        question: '',
        context: {}
      };

      const response = await request(app)
        .post('/api/questions/ask')
        .send(questionData)
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toContain('empty');
      expect(response.body.code).toBe('EMPTY_QUESTION');
    });

    test('POST /api/questions/ask should reject missing question', async () => {
      const questionData = {
        context: {}
      };

      const response = await request(app)
        .post('/api/questions/ask')
        .send(questionData)
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toContain('required');
      expect(response.body.code).toBe('MISSING_QUESTION');
    });

    test('POST /api/questions/ask should reject too long question', async () => {
      const longQuestion = 'a'.repeat(1001); // Exceed 1000 character limit
      const questionData = {
        question: longQuestion,
        context: {}
      };

      const response = await request(app)
        .post('/api/questions/ask')
        .send(questionData)
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toContain('too long');
      expect(response.body.code).toBe('QUESTION_TOO_LONG');
    });
  });

  describe('Status Checking', () => {
    test('GET /api/questions/status/:questionId should return 404 for invalid ID', async () => {
      const invalidId = 'invalid-question-id';
      
      const response = await request(app)
        .get(`/api/questions/status/${invalidId}`)
        .expect(404);

      expect(response.body.success).toBe(false);
      expect(response.body.error).toContain('not found');
    });

    test('Should be able to check status of submitted question', async () => {
      // First submit a question
      const questionData = {
        question: 'Test question for status check',
        context: { test: true }
      };

      const submitResponse = await request(app)
        .post('/api/questions/ask')
        .send(questionData)
        .expect(200);

      const questionId = submitResponse.body.questionId;

      // Then check its status
      const statusResponse = await request(app)
        .get(`/api/questions/status/${questionId}`)
        .expect(200);

      expect(statusResponse.body.success).toBe(true);
      expect(statusResponse.body.stage).toBeDefined();
      expect(statusResponse.body.startTime).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    test('Should handle 404 for unknown routes', async () => {
      const response = await request(app)
        .get('/api/unknown-route')
        .expect(404);

      expect(response.body.error).toContain('not found');
    });

    test('Should validate content-type for POST requests', async () => {
      const response = await request(app)
        .post('/api/questions/ask')
        .send('invalid json')
        .expect(400);

      // Should handle malformed JSON
    });
  });

  describe('Input Validation', () => {
    test('Should accept valid context object', async () => {
      const questionData = {
        question: 'Valid question',
        context: {
          user_id: '123',
          preferences: {
            language: 'en',
            detailed: true
          }
        }
      };

      const response = await request(app)
        .post('/api/questions/ask')
        .send(questionData)
        .expect(200);

      expect(response.body.success).toBe(true);
    });

    test('Should reject invalid context type', async () => {
      const questionData = {
        question: 'Valid question',
        context: 'invalid context type'
      };

      const response = await request(app)
        .post('/api/questions/ask')
        .send(questionData)
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.code).toBe('INVALID_CONTEXT_TYPE');
    });
  });
});