const logger = require('../utils/logger');
const BrainService = require('./BrainService');
const OllamaService = require('./OllamaService');

class QuestionProcessor {
  constructor() {
    this.brainService = new BrainService();
    this.ollamaService = new OllamaService();
    this.isInitialized = false;
    this.init();
  }

  /**
   * Inisialisasi layanan
   */
  async init() {
    try {
      // Layanan menginisialisasi diri mereka sendiri
      await Promise.all([
        this.brainService.init(),
        this.ollamaService.init()
      ]);

      this.isInitialized = true;
      logger.info('QuestionProcessor berhasil diinisialisasi');
    } catch (error) {
      logger.error(`Gagal menginisialisasi QuestionProcessor: ${error.message}`);
      throw error;
    }
  }

  /**
   * Memproses pertanyaan melalui pipeline lengkap
   */
  async processQuestion(question, number_whatsapp) {
    try {
      if (!this.isInitialized) {
        await this.init();
      }

      logger.info(`Memproses pertanyaan: ${question}`);
      const startTime = Date.now();

      // Langkah 1: Gunakan BrainService untuk mendapatkan konteks dan tag
      const brainResult = await this.brainService.processContext(question);
      const tags = await this.brainService.tagQuestion(question, brainResult.relevantEntries);

      // Langkah 2: Proses dengan OllamaService menggunakan template LangChain
      const aiResult = await this.ollamaService.processWithAI(
        question, 
        brainResult, 
        tags, 
        number_whatsapp
      );

      const processingTime = Date.now() - startTime;
      logger.info(`Pertanyaan diproses dalam ${processingTime}ms`);

      return {
        ...aiResult,
        processingTime,
        brainRelevance: brainResult.brainRelevance,
        relevantEntriesCount: brainResult.relevantEntries.length
      };
    } catch (error) {
      logger.error(`Error memproses pertanyaan: ${error.message}`);
      throw error;
    }
  }

  /**
   * Mendapatkan statistik prosesor
   */
  getStats() {
    return {
      isInitialized: this.isInitialized,
      brainService: this.brainService.getServiceStats(),
      ollamaService: this.ollamaService.getServiceStats()
    };
  }

  /**
   * Melatih ulang jaringan Brain.js dengan data baru
   */
  async retrain(newData = null) {
    try {
      const result = await this.brainService.retrain(newData);
      logger.info('Jaringan berhasil dilatih ulang');
      return result;
    } catch (error) {
      logger.error(`Error melatih ulang jaringan: ${error.message}`);
      return false;
    }
  }

  /**
   * Menguji prosesor dengan pertanyaan sampel
   */
  async test() {
    try {
      const testQuestion = "What is artificial intelligence?";
      const result = await this.processQuestion(testQuestion);
      logger.info('Pengujian prosesor berhasil diselesaikan');
      return {
        success: true,
        testQuestion,
        result
      };
    } catch (error) {
      logger.error(`Pengujian prosesor gagal: ${error.message}`);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Menguji layanan individual
   */
  async testServices() {
    try {
      const brainTest = await this.brainService.test();
      const ollamaTest = await this.ollamaService.testConnection();
      
      return {
        brain: brainTest,
        ollama: ollamaTest,
        overall: brainTest.success && ollamaTest.success
      };
    } catch (error) {
      logger.error(`Pengujian layanan gagal: ${error.message}`);
      return {
        brain: { success: false, error: error.message },
        ollama: { success: false, error: error.message },
        overall: false
      };
    }
  }
}

module.exports = QuestionProcessor;