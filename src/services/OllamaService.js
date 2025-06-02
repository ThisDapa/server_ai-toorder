const { ChatOllama } = require("@langchain/ollama");
const logger = require("../utils/logger.js");
const { ChatPromptTemplate, MessagesPlaceholder } = require("@langchain/core/prompts");
const redis = require('redis');
const BrainService = require('./BrainService');
const fuzz = require('fuzzball'); // Tambahkan library untuk fuzzy matching

const redisClient = redis.createClient();
let redisConnected = false;

class OllamaService {
  constructor() {
    this.initialized = false;
    this.chatModel = null;
    this.tagTemplates = {};
    this.brainService = new BrainService();
  }

  /**
   * Inisialisasi layanan Ollama.
   * @returns {Promise<boolean>} - Mengembalikan true jika inisialisasi berhasil, false jika gagal.
   */
  async init() {
    try {
      if (!redisConnected) {
        await redisClient.connect();
        redisConnected = true;
      }
      this.chatModel = new ChatOllama({
        model: process.env.OLLAMA_MODEL || 'llama3.2',
        temperature: process.env.OLLAMA_TEMPERATURE ? parseFloat(process.env.OLLAMA_TEMPERATURE) : 0.12, // lebih rendah, biar lebih confident
        topK: process.env.OLLAMA_TOP_K ? parseInt(process.env.OLLAMA_TOP_K) : 30, // lebih fokus
        topP: process.env.OLLAMA_TOP_P ? parseFloat(process.env.OLLAMA_TOP_P) : 0.8, // lebih fokus
        repeatPenalty: process.env.OLLAMA_REPEAT_PENALTY ? parseFloat(process.env.OLLAMA_REPEAT_PENALTY) : 1.18,
        maxTokens: process.env.OLLAMA_MAX_TOKENS ? parseInt(process.env.OLLAMA_MAX_TOKENS) : 400, // batasi agar jawaban padat
        stop: ['\nuser:', '\nassistant:', '\nAI:', '\nSystem:'],
        system: 'Kamu adalah CustoAI, customer service profesional. Fokuslah pada jawaban yang jelas, ringkas, dan langsung ke inti pertanyaan. Jangan terlalu fokus pada context, prioritaskan jawaban yang relevan dan tidak bertele-tele.',
        think: false,
      });
      await this.brainService.init();
      this.initializeTemplates();
      this.initialized = true;
      logger.info('OllamaService berhasil diinisialisasi');
      return true;
    } catch (error) {
      logger.error(`Error inisialisasi OllamaService: ${error.message}`);
      this.initialized = false;
      this.chatModel = null;
      return false;
    }
  }

  /**
   * Inisialisasi template prompt
   */
  initializeTemplates() {
    // Template untuk tag greeting
    this.tagTemplates.greeting = `
      Kamu adalah CustoAI, customer service profesional dan ramah. Jawablah seperti manusia, bukan robot.
      - Jika pelanggan menyapa, balas dengan ramah dan gunakan bahasa sehari-hari.
      - Tawarkan bantuan untuk mencari produk premium atau layanan streaming.
      - Tambahkan emoji yang sesuai agar percakapan lebih hidup.
      - Jangan pernah memberikan informasi produk yang tidak ada di daftar produk.
      - Jika produk tidak ditemukan, jawab: "Maaf kak, produk tersebut tidak tersedia di toko kami 🙏"
      - Jika produk ada tapi stok 0, jawab: "Maaf kak, untuk saat ini [nama produk] sedang kosong. Mau dicek produk lainnya? 😊"

      context:
      {context}


      Data Produk:
      {products}
    `;

    // Template untuk tag price_inquiry
    this.tagTemplates.price_inquiry = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      - Jika pelanggan meminta pricelist/daftar produk berikan semua produk yang tersedia beserta harga dan stock nya.
      - Jika pelanggan menanyakan harga, cek ketersediaan produk di daftar produk.
      - Jika produk tidak ada di daftar, jawab: "Maaf kak, produk tersebut tidak tersedia di toko kami 🙏"
      - Jika produk ditemukan:
        * Sebutkan harga dan stok dengan jelas dan benar sesuai data.
        * Jelaskan fitur/keunggulan produk dari field desc.
        * Jika stok tersedia, tunjukkan antusiasme dan tawarkan proses pembelian.
        * Jika stok 0, beritahu dengan sopan bahwa sedang kosong dan tawarkan produk alternatif yang tersedia (jika ada).
      - Jangan pernah memberikan informasi produk yang tidak ada di daftar produk.
      - Jangan pernah mengarang harga, stok, atau nama produk.
      - Pilih kata yang sopan, jelas, dan mudah dipahami.
      - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

      context:
      {context}


      Data Produk:
      {products}
    `;

    // Template untuk tag availability
    this.tagTemplates.availability = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.


      - Jika pelanggan menanyakan ketersediaan produk:
        * Jika produk tidak ada di daftar, jawab: "Maaf kak, produk tersebut tidak tersedia di toko kami 🙏"
        * Jika produk ada tapi stok 0: "Maaf kak, untuk saat ini [nama produk] sedang kosong. Mau dicek produk lainnya? 😊"
        * Jika produk tersedia: Beritahu stok dan harga dengan benar, lalu tawarkan proses pembelian.
      - Jangan pernah memberikan informasi produk yang tidak ada di daftar produk.
      - Jangan pernah mengarang harga, stok, atau nama produk.
      - Pilih kata yang sopan, jelas, dan mudah dipahami.
      - Tambahkan emoji yang sesuai agar percakapan lebih hidup.
      - Jika relevan, tawarkan alternatif produk serupa yang tersedia.

      context:
      {context}


      Data Produk:
      {products}
    `;

    // Template untuk tag unknown (default)
    this.tagTemplates.unknown = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      - Berikan jawaban yang relevan, jelas, dan membantu sesuai pertanyaan pelanggan.
      - Jangan pernah memberikan informasi produk, harga, stok, metode pembayaran, atau layanan yang tidak ada di daftar produk.
      - Untuk pertanyaan produk:
        * Jika produk tidak ditemukan: "Maaf kak, produk tersebut tidak tersedia di toko kami 🙏"
        * Jika produk ada tapi stok 0: "Maaf kak, untuk saat ini [nama produk] sedang kosong. Mau dicek produk lainnya? 😊"
      - Untuk pertanyaan metode pembayaran: "Maaf kak, saat ini pembayaran hanya bisa melalui QRIS 🙏"
      - Jika pertanyaan tentang layanan AI atau produk digital, jawab hanya jika ada di daftar produk/layanan.
      - Jika tidak yakin dengan jawaban, tawarkan untuk menghubungi customer service manusia.
      - Pilih kata yang sopan, jelas, dan mudah dipahami.
      - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

      context:
      {context}

      Data Produk:
      {products}
    `;

    // Template untuk tag technical_details
    this.tagTemplates.technical_details = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      - Jika pelanggan menanyakan detail teknis produk (misal: bisa di HP, smart TV, jumlah user, kualitas, dsb), jawab hanya berdasarkan data produk yang tersedia.
      - Jika data teknis tidak ada di daftar produk, jawab: "Maaf kak, informasi teknis tersebut tidak tersedia untuk produk ini 🙏"
      - Jangan pernah mengarang fitur, spesifikasi, atau keunggulan yang tidak ada di data produk.
      - Pilih kata yang sopan, jelas, dan mudah dipahami.
      - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

      context:
      {context}

      Data Produk:
      {products}
    `;

    // Template untuk tag payment_method
    this.tagTemplates.payment_method = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      - Jika pelanggan menanyakan metode pembayaran, jawab hanya QRIS sebagai satu-satunya metode pembayaran yang tersedia.
      - Jika pelanggan bertanya tentang metode lain, jawab: "Maaf kak, saat ini pembayaran hanya bisa melalui QRIS 🙏"
      - Jangan pernah mengarang atau menambah metode pembayaran lain.
      - Pilih kata yang sopan, jelas, dan mudah dipahami.
      - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

      context:
      {context}

      Data Produk:
      {products}
    `;

    // Template untuk tag refund_policy
    this.tagTemplates.refund_policy = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      - Jika pelanggan menanyakan kebijakan refund/garansi, jawab hanya sesuai kebijakan yang berlaku di toko dan data yang tersedia.
      - Jika tidak ada kebijakan refund untuk produk tersebut, jawab: "Maaf kak, untuk produk ini belum ada kebijakan refund khusus 🙏"
      - Jangan pernah mengarang atau menjanjikan refund/garansi di luar kebijakan yang ada.
      - Pilih kata yang sopan, jelas, dan mudah dipahami.
      - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

      context:
      {context}

      Data Produk:
      {products}
    `;

    // Template untuk tag emerging_services
    this.tagTemplates.emerging_services = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      - Jika pelanggan menanyakan layanan baru, AI tools, atau fitur digital lain, jawab hanya berdasarkan layanan yang benar-benar tersedia di daftar produk/layanan.
      - Jika layanan tidak tersedia, jawab: "Maaf kak, layanan tersebut belum tersedia di toko kami 🙏"
      - Jangan pernah mengarang atau menambah layanan yang tidak ada.
      - Pilih kata yang sopan, jelas, dan mudah dipahami.
      - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

      context:
      {context}

      Data Produk:
      {products}
    `;

    // Template untuk tag referral_loyalty
    this.tagTemplates.referral_loyalty = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      - Jika pelanggan menanyakan program referral, loyalti, atau bonus, jawab hanya sesuai program yang benar-benar berlaku di toko.
      - Jika tidak ada program referral/loyalti, jawab: "Maaf kak, saat ini belum ada program referral atau loyalti di toko kami 🙏"
      - Jangan pernah mengarang atau menjanjikan bonus/loyalti di luar program yang ada.
      - Pilih kata yang sopan, jelas, dan mudah dipahami.
      - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

      context:
      {context}

      Data Produk:
      {products}
    `;
  }

  /**
   * Menghitung kesamaan antara dua string menggunakan Jaccard similarity.
   * @param {string} str1 - String pertama.
   * @param {string} str2 - String kedua.
   * @returns {number} - Skor kesamaan (0-1).
   */
  calculateSimilarity(str1, str2) {
    if (!str1 || !str2) return 0;

    const tokens1 = new Set(str1.toLowerCase().split(/\s+/));
    const tokens2 = new Set(str2.toLowerCase().split(/\s+/));

    const intersection = new Set([...tokens1].filter(x => tokens2.has(x)));
    const union = new Set([...tokens1, ...tokens2]);

    return intersection.size / union.size;
  }

  /**
   * Memproses pertanyaan pelanggan menggunakan AI.
   * @param {string} question - Pertanyaan pelanggan.
   * @param {Object} context - Konteks dari BrainService.
   * @param {string} nomorWhatsapp - Nomor WhatsApp pelanggan untuk mendapatkan riwayat chat.
   * @returns {Promise<Object>} - Hasil pemrosesan AI, termasuk respons dan informasi produk.
   */
  async processWithAI(question, context, nomorWhatsapp) {
    try {
      if (!this.initialized) {
        await this.init();
      }

      logger.info(`Sedang memproses pertanyaan: '${question}'`);

      const historyKey = `lel:${nomorWhatsapp}`;
      let history = await redisClient.lRange(historyKey, -6, -1);
      history = history.map(JSON.parse);

      const formattedHistory = history.map(msg => {
        return msg.role === 'user' ? 
          {type: 'user', content: msg.content} : 
          {type: 'ai', content: msg.content};
      });

      const tags = await this.getQuestionTag(question);
      let systemTemplate = this.tagTemplates[tags] || this.tagTemplates.unknown;

      const relevantInfo = this.formatRelevantInfo(context.relevantEntries);

      const products = this.getProductData();

      const words = question.toLowerCase().split(/\s+/);
      let productInfo = null;

      for (let i = 0; i < words.length; i++) {
        for (let j = i + 1; j <= Math.min(i + 5, words.length); j++) {
          const potentialProduct = words.slice(i, j).join(' ');
          const availability = this.checkProductAvailability(potentialProduct);
          if (availability.exists) {
            productInfo = availability;
            break;
          }
        }
        if (productInfo) break;
      }

      const productString = Object.entries(products)
        .map(([name, p]) => {
          const stockStatus = parseInt(p.stock) > 0 ? 'Tersedia' : 'Kosong';
          return `${name}: Harga ${p.price}, Status: ${stockStatus}, Stock: ${p.stock}`;
        })
        .join('\n');

      const chatPrompt = ChatPromptTemplate.fromMessages([
        ["system", systemTemplate],
        ...formattedHistory.map(msg => [msg.type === 'user' ? 'user' : 'assistant', msg.content]),
        ["user", "{query}"]
      ]);

      const formattedMessages = await chatPrompt.formatMessages({
        context: relevantInfo,
        products: productString,
        chat_history: formattedHistory,
        query: question
      });

      const startTime = Date.now();
      const response = await this.chatModel.invoke(formattedMessages);
      const processingTime = Date.now() - startTime;

      const chatEntry = JSON.stringify({
        role: "user",
        content: question
      });
      await redisClient.rPush(historyKey, chatEntry);
      await redisClient.expire(historyKey, 2 * 24 * 60 * 60);

      const aiEntry = JSON.stringify({
        role: "assistant",
        content: response.content
      });
      await redisClient.rPush(historyKey, aiEntry);
      await redisClient.expire(historyKey, 2 * 24 * 60 * 60); 

      const historyLength = await redisClient.lLen(historyKey);
      if (historyLength > 20) {
        await redisClient.lTrim(historyKey, historyLength - 20, -1);
      }
      logger.info(`Pertanyaan '${question}' diproses dalam ${processingTime}ms dengan jawaban '${response.content}'`);

      return {
        response: response.content,
        processingTime,
        tags,
        productInfo
      };

    } catch (error) {
      logger.error(`Error processing with AI: ${error.message}`);
      throw error;
    }
  }

  /**
   * Format informasi relevan untuk prompt
   * @param {Array} relevantEntries - Entri relevan dari dataset
   * @returns {string} - Informasi relevan yang diformat
   */
  formatRelevantInfo(relevantEntries) {
    if (!relevantEntries || relevantEntries.length === 0) {
      return 'Tidak ada informasi relevan yang ditemukan dalam dataset';
    }

    return relevantEntries
      .map((entry, index) => {
        return `${index + 1}. Pertanyaan: "${entry.question}"\n   Konteks: ${JSON.stringify(entry.answer, null, 2)}`;
      })
      .join('\n\n');
  }

  /**
   * Dapatkan data produk
   * @returns {Object} - Data produk
   */
  getProductData() {
    return {
      "Akun Gmail Fresh": {
        price: "Rp 5.000",
        stock: "2",
        desc: "Akun Gmail baru dengan garansi 7 hari. Cocok untuk keperluan registrasi atau akun utama."
      },
      "Akun Gmail Aged": {
        price: "Rp 15.000/akun",
        stock: "1",
        desc: "Akun Gmail berumur lebih dari 1 tahun dengan garansi 14 hari. Cocok untuk bisnis atau akun verifikasi."
      },
      "Netflix 1P1U": {
        price: "Rp 24.000",
        stock: "0",
        desc: "Akun Netflix sharing dengan 1 profile dan 1 user. Bebas gangguan, bisa digunakan kapan saja."
      },
      "Netflix 1P2U": {
        price: "Rp 13.000",
        stock: "5",
        desc: "Akun Netflix sharing dengan 1 profile untuk 2 user. Ada kemungkinan kendala jika dipakai bersamaan."
      },
      "Netflix Premium 4K UHD": {
        price: "Rp 50.000",
        stock: "3",
        desc: "Akun Netflix sharing dengan kualitas 4K UHD. Bisa digunakan di 4 perangkat bersamaan."
      },
      "Disney+ Hotstar": {
        price: "Rp 30.000",
        stock: "8",
        desc: "Akun Disney+ Hotstar premium untuk menonton film dan serial eksklusif Disney, Marvel, dan lainnya."
      }
    };
  }

  /**
   * Get the tag classification for a given question
   * @param {string} question - The question to classify
   * @returns {Promise<string>} - Returns the classified tag
   */
  async getQuestionTag(question) {
    try {
      if (!this.initialized) {
        await this.init();
      }

      const messages = [
        {
          role: "user",
          content: `Classify the following question into one of the tags: price_inquiry, availability, greeting, technical_details, payment_method, refund_policy, emerging_services, warranty_refund, referral_loyalty, order, unknown. Return only the tag for this question: "${question}"`
        }
      ];

      const response = await this.chatModel.invoke(messages, {
        format: {
          type: "object",
          properties: {
            tag: {
              type: "string",
              enum: ["price_inquiry", "availability", "greeting", "technical_details", 
                "payment_method", "refund_policy", "emerging_services", 
                "warranty_refund", "referral_loyalty", "order", "unknown"
              ]         
            }
          },
          required: ["tag"]
        }
      });

      // Extract tag from response
      let tag;
      try {
        // Try to parse if response is JSON string
        const parsed = JSON.parse(response.content);
        tag = parsed.tag;
      } catch {
        // If not JSON, try to extract tag directly from content
        tag = response.content.trim().toLowerCase();
      }

      // Validate tag
      const validTags = [
        "price_inquiry",
        "availability",
        "greeting",
        "technical_details",
        "payment_method",
        "refund_policy",
        "emerging_services",
        "warranty_refund",
        "referral_loyalty",
        "order",
        "unknown",
      ];
      if (!validTags.includes(tag)) {
        tag = "unknown";
      }

      logger.info(`Question "${question}" classified as tag: ${tag}`);
      return tag;
    } catch (error) {
      logger.error(`Error getting question tag: ${error.message}`);
      return "unknown";
    }
  }

  async testConnection() {
    try {
      if (!this.initialized) {
        await this.init();
      }
      if (!this.chatModel) {
        throw new Error('OllamaService not initialized: chatModel is null');
      }
      const startTime = Date.now();
      const response = await this.chatModel.invoke([
        {
          role: 'user',
          content: 'Hello, this is a test message. Please respond with "OK"'
        }
      ]);
      const latency = Date.now() - startTime;
      return {
        success: true,
        model: process.env.OLLAMA_MODEL || 'llama3.2',
        latency,
        response: response.content.substring(0, 50) + '...'
      };
    } catch (error) {
      logger.error(`Error testing Ollama connection: ${error.message}`);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Dapatkan statistik layanan
   * @returns {Object} - Statistik layanan
   */
  getServiceStats() {
    return {
      initialized: this.initialized,
      model: process.env.OLLAMA_MODEL || 'llama3.2',
      temperature: parseFloat(process.env.OLLAMA_TEMPERATURE) || 0.2,
      templateCount: Object.keys(this.tagTemplates).length
    };
  }

  /**
   * Checks if a product exists and is in stock
   * @param {string} productName - Name of the product to check
   * @returns {Object} - Product availability info {exists: boolean, inStock: boolean, product: Object}
   */
  checkProductAvailability(productName) {
    const products = this.getProductData();
    const normalizedSearch = productName.toLowerCase().trim();

    // Enhanced product matching using Fuzzy Matching
    let bestMatch = null;
    let highestScore = 0;

    Object.entries(products).forEach(([name, product]) => {
      const score = fuzz.ratio(normalizedSearch, name.toLowerCase());
      if (score > highestScore) {
        highestScore = score;
        bestMatch = { name, product };
      }
    });

    if (!bestMatch || highestScore < 70) { // Threshold for fuzzy matching
      return { exists: false, inStock: false, product: null };
    }

    const { name, product } = bestMatch;
    const inStock = parseInt(product.stock) > 0;

    return {
      exists: true,
      inStock,
      product: { ...product, name }
    };
  }
}

module.exports = OllamaService;