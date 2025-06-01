const QuestionProcessor = require('../src/services/QuestionProcessor');

// Mock dependencies
jest.mock('../src/services/BrainService', () => {
  return jest.fn().mockImplementation(() => ({
    init: jest.fn().mockResolvedValue(true),
    processContext: jest.fn().mockResolvedValue({
      brainRelevance: 0.85,
      predictedCategory: 'test',
      predictedTags: ['test', 'ai'],
      context: 'Ini adalah konteks pengujian',
      confidence: 0.9,
      relevantEntries: [
        { question: 'Apa itu AI?', answer: 'AI adalah kecerdasan buatan', tags: ['ai', 'technology'] }
      ],
      totalDatasetSize: 10,
      brainAnalysis: { test: 0.9 }
    }),
    tagQuestion: jest.fn().mockResolvedValue(['test', 'ai', 'question']),
    getServiceStats: jest.fn().mockReturnValue({
      datasetSize: 10,
      modelSize: '250KB',
      lastTrainingTime: '2023-01-01T00:00:00Z'
    }),
    retrain: jest.fn().mockResolvedValue(true),
    test: jest.fn().mockResolvedValue({
      success: true,
      accuracy: 0.92,
      testCases: 50
    })
  }));
});

jest.mock('../src/services/OllamaService', () => {
  return jest.fn().mockImplementation(() => ({
    init: jest.fn().mockResolvedValue(true),
    processWithAI: jest.fn().mockResolvedValue({
      answer: 'Ini adalah jawaban dari AI',
      confidence: 0.85,
      tags: ['test', 'ai', 'question'],
      sources: 1,
      processingTime: 250,
      tag: 'test'
    }),
    getServiceStats: jest.fn().mockReturnValue({
      model: 'llama3',
      temperature: 0.2,
      requestsProcessed: 25
    }),
    testConnection: jest.fn().mockResolvedValue({
      success: true,
      model: 'llama3',
      latency: 120
    })
  }));
});

jest.mock('../src/utils/logger', () => ({
  info: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn()
}));

describe('QuestionProcessor', () => {
  let questionProcessor;

  beforeEach(() => {
    jest.clearAllMocks();
    questionProcessor = new QuestionProcessor();
  });

  describe('init', () => {
    test('berhasil menginisialisasi layanan', async () => {
      await questionProcessor.init();
      
      expect(questionProcessor.brainService.init).toHaveBeenCalled();
      expect(questionProcessor.ollamaService.init).toHaveBeenCalled();
      expect(questionProcessor.isInitialized).toBe(true);
    });

    test('menangani kesalahan inisialisasi', async () => {
      questionProcessor.brainService.init.mockRejectedValueOnce(new Error('Kesalahan inisialisasi'));
      
      await expect(questionProcessor.init()).rejects.toThrow('Kesalahan inisialisasi');
      expect(questionProcessor.isInitialized).toBe(false);
    });
  });

  describe('processQuestion', () => {
    test('memproses pertanyaan dengan benar', async () => {
      const result = await questionProcessor.processQuestion('Apa itu kecerdasan buatan?');
      
      expect(questionProcessor.brainService.processContext).toHaveBeenCalledWith('Apa itu kecerdasan buatan?');
      expect(questionProcessor.brainService.tagQuestion).toHaveBeenCalled();
      expect(questionProcessor.ollamaService.processWithAI).toHaveBeenCalled();
      
      expect(result).toHaveProperty('answer', 'Ini adalah jawaban dari AI');
      expect(result).toHaveProperty('processingTime');
      expect(result).toHaveProperty('brainRelevance', 0.85);
      expect(result).toHaveProperty('relevantEntriesCount', 1);
    });

    test('menginisialisasi jika belum diinisialisasi', async () => {
      questionProcessor.isInitialized = false;
      
      await questionProcessor.processQuestion('Apa itu kecerdasan buatan?');
      
      expect(questionProcessor.brainService.init).toHaveBeenCalled();
      expect(questionProcessor.ollamaService.init).toHaveBeenCalled();
    });

    test('menangani kesalahan pemrosesan', async () => {
      questionProcessor.brainService.processContext.mockRejectedValueOnce(new Error('Kesalahan pemrosesan'));
      
      await expect(questionProcessor.processQuestion('Apa itu kecerdasan buatan?')).rejects.toThrow('Kesalahan pemrosesan');
    });
  });

  describe('getStats', () => {
    test('mengembalikan statistik yang benar', () => {
      const stats = questionProcessor.getStats();
      
      expect(stats).toHaveProperty('isInitialized');
      expect(stats).toHaveProperty('brainService');
      expect(stats).toHaveProperty('ollamaService');
      
      expect(stats.brainService).toHaveProperty('datasetSize', 10);
      expect(stats.ollamaService).toHaveProperty('model', 'llama3');
    });
  });

  describe('retrain', () => {
    test('melatih ulang jaringan dengan data baru', async () => {
      const newData = [{ question: 'Pertanyaan baru', answer: 'Jawaban baru', tags: ['new'] }];
      const result = await questionProcessor.retrain(newData);
      
      expect(questionProcessor.brainService.retrain).toHaveBeenCalledWith(newData);
      expect(result).toBe(true);
    });

    test('menangani kesalahan pelatihan ulang', async () => {
      questionProcessor.brainService.retrain.mockRejectedValueOnce(new Error('Kesalahan pelatihan'));
      
      const result = await questionProcessor.retrain();
      
      expect(result).toBe(false);
    });
  });

  describe('test', () => {
    test('menguji prosesor dengan pertanyaan sampel', async () => {
      const result = await questionProcessor.test();
      
      expect(questionProcessor.processQuestion).toHaveBeenCalledWith('What is artificial intelligence?');
      expect(result).toHaveProperty('success', true);
      expect(result).toHaveProperty('testQuestion');
      expect(result).toHaveProperty('result');
    });

    test('menangani kesalahan pengujian', async () => {
      questionProcessor.processQuestion = jest.fn().mockRejectedValueOnce(new Error('Kesalahan pengujian'));
      
      const result = await questionProcessor.test();
      
      expect(result).toHaveProperty('success', false);
      expect(result).toHaveProperty('error', 'Kesalahan pengujian');
    });
  });

  describe('testServices', () => {
    test('menguji layanan individual', async () => {
      const result = await questionProcessor.testServices();
      
      expect(questionProcessor.brainService.test).toHaveBeenCalled();
      expect(questionProcessor.ollamaService.testConnection).toHaveBeenCalled();
      
      expect(result).toHaveProperty('brain.success', true);
      expect(result).toHaveProperty('ollama.success', true);
      expect(result).toHaveProperty('overall', true);
    });

    test('menangani kesalahan pengujian layanan', async () => {
      questionProcessor.brainService.test.mockRejectedValueOnce(new Error('Kesalahan pengujian layanan'));
      
      const result = await questionProcessor.testServices();
      
      expect(result).toHaveProperty('brain.success', false);
      expect(result).toHaveProperty('ollama.success', false);
      expect(result).toHaveProperty('overall', false);
    });
  });
});