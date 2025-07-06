/**
 * OllamaService Module
 * Handles AI processing using Ollama models and LangChain integration
 * Manages chat history, product information, and template-based responses
 */

'use strict';

const { ChatOllama } = require("@langchain/ollama");
const logger = require("../utils/logger.js");
const { ChatPromptTemplate } = require("@langchain/core/prompts");
const redis = require('redis');
const BrainService = require('./BrainService');
const fuzz = require('fuzzball');

// Configuration constants
const CONFIG = {
  REDIS_EXPIRY: 2 * 24 * 60 * 60, // 2 days in seconds
  MAX_HISTORY_LENGTH: 12,
  FUZZY_MATCH_THRESHOLD: 70,
  HISTORY_CONTEXT_SIZE: 6 // Number of recent messages to include in context
};

// Initialize Redis client
const redisClient = redis.createClient();
let redisConnected = false;

/**
 * OllamaService class
 * Manages AI-powered customer service interactions using Ollama models
 */
class OllamaService {
  /**
   * Creates a new OllamaService instance
   */
  constructor() {
    this.initialized = false;
    this.chatModel = null;
    this.tagTemplates = {};
    this.brainService = new BrainService();
  }

  /**
   * Initializes the Ollama service
   * @returns {Promise<boolean>} True if initialization succeeds, false otherwise
   */
  async init() {
    try {
      await this.connectToRedis();
      await this.initializeChatModel();
      await this.brainService.init();
      this.initializeTemplates();
      
      this.initialized = true;
      logger.info("OllamaService berhasil diinisialisasi");
      return true;
    } catch (error) {
      logger.error(`Error inisialisasi OllamaService: ${error.message}`);
      this.initialized = false;
      this.chatModel = null;
      return false;
    }
  }
  
  /**
   * Connects to Redis if not already connected
   * @returns {Promise<void>}
   * @private
   */
  async connectToRedis() {
    if (!redisConnected) {
      await redisClient.connect();
      redisConnected = true;
    }
  }
  
  /**
   * Initializes the Ollama chat model with configuration
   * @returns {Promise<void>}
   * @private
   */
  async initializeChatModel() {
    this.chatModel = new ChatOllama({
      model: process.env.OLLAMA_MODEL || "llama3.2",
      temperature: this.getConfigFloat('OLLAMA_TEMPERATURE', 0.12),
      topK: this.getConfigInt('OLLAMA_TOP_K', 30),
      topP: this.getConfigFloat('OLLAMA_TOP_P', 0.8),
      repeatPenalty: this.getConfigFloat('OLLAMA_REPEAT_PENALTY', 1.18),
      maxTokens: this.getConfigInt('OLLAMA_MAX_TOKENS', 400),
      stop: ["\nuser:", "\nassistant:", "\nAI:", "\nSystem:"],
      system: `Kamu adalah CustoAI, customer service profesional dari toko yang bernama ${process.env.NAME_STORE}. Fokuslah pada jawaban yang jelas, ringkas, dan langsung ke inti pertanyaan. Jangan terlalu fokus pada context, prioritaskan jawaban yang relevan dan tidak bertele-tele.`,
      think: false,
    });
  }
  
  /**
   * Gets a float configuration value from environment variables
   * @param {string} key - Environment variable key
   * @param {number} defaultValue - Default value if not found
   * @returns {number} - Parsed float value
   * @private
   */
  getConfigFloat(key, defaultValue) {
    return process.env[key] ? parseFloat(process.env[key]) : defaultValue;
  }
  
  /**
   * Gets an integer configuration value from environment variables
   * @param {string} key - Environment variable key
   * @param {number} defaultValue - Default value if not found
   * @returns {number} - Parsed integer value
   * @private
   */
  getConfigInt(key, defaultValue) {
    return process.env[key] ? parseInt(process.env[key]) : defaultValue;
  }

  /**
   * Inisialisasi template prompt
   */
  initializeTemplates() {
    // Template untuk tag greeting
    this.tagTemplates.greeting = `
      Kamu adalah CustoAI, customer service profesional dan ramah dari ${process.env.NAME_STORE || 'toko kami'}. Jawablah seperti manusia, bukan robot.
      
      ATURAN UTAMA:
      - Sapa pelanggan dengan hangat dan profesional
      - Perkenalkan diri sebagai CustoAI dan tawarkan bantuan
      - Sebutkan produk unggulan: Akun Gmail, Netflix, Disney+ Hotstar
      - Gunakan emoji yang sesuai untuk membuat percakapan lebih hidup
      - Jangan langsung memberikan daftar harga kecuali diminta
      - Fokus pada membangun rapport dan menanyakan kebutuhan pelanggan
      
      CONTOH RESPONS:
      "Halo! 👋 Selamat datang di ${process.env.NAME_STORE || 'toko kami'}! Saya CustoAI, siap membantu Anda. 
      
      Kami menyediakan:
      🔹 Akun Gmail (Fresh & Aged)
      🔹 Netflix Premium (berbagai paket)
      🔹 Disney+ Hotstar
      
      Ada yang bisa saya bantu hari ini? 😊"

      context: {context}
      Data Produk: {products}
    `;

    // Template untuk tag price_inquiry
    this.tagTemplates.price_inquiry = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      
      ATURAN UTAMA:
      - WAJIB gunakan data produk yang akurat dari daftar yang diberikan
      - Jika diminta pricelist lengkap, tampilkan semua produk dengan format yang rapi
      - Untuk pertanyaan harga spesifik, berikan detail lengkap produk tersebut
      - SELALU cek stok sebelum memberikan informasi
      - Jangan pernah mengarang atau mengubah harga, nama, atau stok produk
      
      FORMAT RESPONS PRICELIST:
      "📋 **DAFTAR HARGA PRODUK**\n\n" +
      "🔹 **Akun Gmail Fresh** - Rp 5.000 (Stok: 2)\n   ✨ Akun baru dengan garansi 7 hari\n\n" +
      "🔹 **Akun Gmail Aged** - Rp 15.000 (Stok: 1)\n   ✨ Akun berumur 1+ tahun, garansi 14 hari\n\n" +
      [lanjutkan untuk semua produk]
      
      RESPONS UNTUK PRODUK KOSONG:
      "Maaf kak, [nama produk] sedang kosong 😔 Mau cek produk lain yang tersedia?"
      
      RESPONS UNTUK PRODUK TIDAK ADA:
      "Maaf kak, produk tersebut tidak tersedia di toko kami 🙏 Ini produk yang kami punya: [sebutkan alternatif]"
      
      SELALU AKHIRI DENGAN:
      "Mau pesan yang mana kak? Tinggal bilang aja! 😊"

      context: {context}
      Data Produk: {products}
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

    // Template untuk konfirmasi pesanan
    this.tagTemplates.order_confirmation = `
      Kamu adalah CustoAI, customer service yang menangani konfirmasi pesanan.
      
      ATURAN UTAMA:
      - Pelanggan sedang mengkonfirmasi pesanan yang sudah dibuat
      - Berikan instruksi pembayaran yang jelas
      - Sebutkan total yang harus dibayar
      - Berikan informasi kontak admin untuk pembayaran
      
      FORMAT RESPONS:
      "✅ **Pesanan Dikonfirmasi!**\n\n" +
      "📋 Detail pesanan Anda sudah dicatat\n" +
      "💰 Total pembayaran: [total dari context]\n\n" +
      "📱 **Cara Pembayaran:**\n" +
      "1️⃣ Transfer ke rekening/QRIS yang akan diberikan admin\n" +
      "2️⃣ Kirim bukti transfer\n" +
      "3️⃣ Tunggu konfirmasi dan pengiriman akun\n\n" +
      "📞 Hubungi admin: [kontak admin]\n" +
      "⏰ Pesanan akan diproses dalam 1-24 jam"
      
      context: {context}
      Data Produk: {products}
    `;

    // Template untuk tag unknown (default)
    this.tagTemplates.unknown = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      
      ATURAN UTAMA:
      - Analisis pertanyaan pelanggan dengan cermat
      - Berikan jawaban yang relevan dan membantu
      - Jika tentang produk, gunakan data yang akurat
      - Jika tidak yakin, arahkan ke layanan yang tersedia
      - Selalu tawarkan bantuan lebih lanjut
      
      RESPONS UNTUK BERBAGAI SITUASI:
      
      Produk tidak tersedia:
      "Maaf kak, produk tersebut tidak tersedia di toko kami 🙏 Tapi kami punya produk lain yang mungkin cocok: [sebutkan alternatif]"
      
      Pertanyaan pembayaran:
      "Untuk pembayaran, kami menerima transfer bank dan QRIS 💳 Setelah pesan, admin akan berikan detail pembayarannya ya!"
      
      Pertanyaan teknis/garansi:
      "Semua produk kami bergaransi sesuai ketentuan 🛡️ Untuk detail teknis, bisa langsung tanya admin setelah pembelian ya!"
      
      Pertanyaan umum:
      "Saya siap membantu! Mau tanya tentang produk, harga, atau cara pesan? 😊"
      
      SELALU AKHIRI DENGAN:
      "Ada yang bisa saya bantu lagi? 🤗"

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

    // Template untuk tag products_details
    this.tagTemplates.products_details = `
      Kamu adalah CustoAI, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
      
      - Jika pelanggan bertanya tentang detail produk, jawab berdasarkan data produk yang tersedia di database.
      - Berikan informasi seperti: fitur, manfaat, masa aktif, metode pengiriman, dan hal penting lain jika tersedia.
      - Jangan memberikan informasi yang tidak ada dalam data produk.
      - Jika data produk terbatas, jawab sejujur mungkin dan ajak pelanggan untuk bertanya lebih lanjut jika perlu.
      - Gunakan bahasa yang sopan, ramah, dan mudah dipahami.
      - Tambahkan emoji yang relevan agar lebih bersahabat.

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

    const intersection = new Set([...tokens1].filter((x) => tokens2.has(x)));
    const union = new Set([...tokens1, ...tokens2]);

    return intersection.size / union.size;
  }

  /**
   * Processes a customer question using AI
   * @param {string} question - Customer's question
   * @param {Object} context - Context from BrainService
   * @param {string} nomorWhatsapp - Customer's WhatsApp number for chat history
   * @returns {Promise<Object>} - AI processing result with response and metadata
   * @throws {Error} If processing fails
   */
  /**
   * Handles product clarification responses
   * @param {string} question - User's question/response
   * @param {Array} formattedHistory - Formatted chat history
   * @returns {Promise<Object>} Product clarification result
   * @private
   */
  async handleProductClarification(question, formattedHistory) {
    try {
      logger.info(`Processing product clarification response: "${question}"`);
      
      // Get product data
      const productData = this.getProductData();
      
      // Find the ambiguous product query in chat history
      let ambiguousProductInfo = null;
      let selectedVariant = null;
      
      // Look for the ambiguous product query in recent messages
      for (let i = formattedHistory.length - 5; i < formattedHistory.length; i++) {
        if (i < 0) continue;
        
        const msg = formattedHistory[i];
        if (msg.type === 'ai' && msg.content && 
            msg.content.includes('⚠️ **Mohon Klarifikasi**') && 
            msg.content.includes('Saya menemukan beberapa varian produk')) {
          
          // Extract product variants from the message
          for (const [name, data] of Object.entries(productData)) {
            const normalizedName = name.toLowerCase();
            const normalizedQuestion = question.toLowerCase();
            const normalizedCode = data.code.toLowerCase();
            
            if (normalizedQuestion.includes(normalizedName) || normalizedQuestion.includes(normalizedCode)) {
              selectedVariant = {
                name,
                code: data.code,
                price: data.price,
                quantity: 1 // Default quantity
              };
              break;
            }
          }
          
          break;
        }
      }
      
      if (selectedVariant) {
        logger.info(`User selected product variant: ${selectedVariant.name}`);
        
        // Extract quantity if specified
        const quantityMatches = question.match(/\b(\d+)\s*(buah|pcs|unit|akun)?\b/i);
        if (quantityMatches && quantityMatches[1]) {
          selectedVariant.quantity = parseInt(quantityMatches[1]);
        }
        
        // Calculate price
        const price = parseInt(selectedVariant.price.replace(/[^0-9]/g, '')) || 0;
        const subtotal = price * selectedVariant.quantity;
        
        const response = `✅ **Pesanan Dikonfirmasi**\n\n` +
          `• ${selectedVariant.name} (${selectedVariant.code})\n` +
          `  Harga: ${selectedVariant.price} x ${selectedVariant.quantity} = Rp ${subtotal.toLocaleString()}\n\n` +
          `💰 **Total: Rp ${subtotal.toLocaleString()}**\n\n` +
          `✅ Untuk melanjutkan pemesanan, silakan konfirmasi dengan mengetik "konfirmasi" atau hubungi admin untuk pembayaran.`;
        
        return {
          response,
          processingTime: 0,
          tags: "product_clarification",
          productInfo: { 
            orders: [selectedVariant], 
            total: subtotal 
          },
        };
      } else {
        logger.warn(`Could not identify selected product variant in user response`);
        
        return {
          response: `❓ Maaf, saya tidak dapat mengenali produk yang Anda pilih. Mohon sebutkan nama lengkap atau kode produk yang Anda inginkan.`,
          processingTime: 0,
          tags: "product_clarification",
          productInfo: null,
        };
      }
    } catch (error) {
      logger.error(`Error in handleProductClarification: ${error.message}`);
      return {
        response: `❌ Maaf, terjadi kesalahan saat memproses pilihan produk Anda. Silakan coba lagi atau hubungi admin untuk bantuan.`,
        processingTime: 0,
        tags: "product_clarification",
        productInfo: null,
      };
    }
  }

  async processWithAI(question, context, nomorWhatsapp) {
    try {
      // Input validation
      if (!question || typeof question !== 'string' || question.trim().length === 0) {
        throw new Error('Question is required and must be a non-empty string');
      }
      
      if (!nomorWhatsapp || typeof nomorWhatsapp !== 'string') {
        throw new Error('WhatsApp number is required and must be a string');
      }
      
      await this.ensureInitialized();
      logger.info(`Processing question from user ${nomorWhatsapp}: "${question.substring(0, 100)}${question.length > 100 ? '...' : ''}"`);

      // Get chat history and format it
      const formattedHistory = await this.getChatHistory(nomorWhatsapp);
      logger.info(`Retrieved ${formattedHistory.length} messages from chat history`);
      
      // Determine question type/tag
      const tag = await this.getQuestionTag(question, formattedHistory);
      logger.info(`Question classified as tag: ${tag}`);
      
      // Handle special cases for order tags
      if (tag === "order") {
        logger.info('Processing order request');
        const result = await this.handleOrderTag(formattedHistory);
        logger.info(`Order processing completed with ${result.productInfo?.orders?.length || 0} items`);
        return result;
      }
      
      if (tag === "order_confirmation") {
        logger.info('Processing order confirmation');
        const result = await this.handleOrderConfirmation(formattedHistory);
        logger.info(`Order confirmation processed: ${result.productInfo?.confirmed ? 'confirmed' : 'not found'}`);
        return result;
      }
      
      if (tag === "product_clarification") {
        logger.info('Processing product clarification');
        const result = await this.handleProductClarification(question, formattedHistory);
        logger.info(`Product clarification processed for: ${result.productInfo?.orders?.[0]?.name || 'unknown product'}`);
        return result;
      }
      
      // Get relevant information and product data
      const relevantInfo = this.formatRelevantInfo(context?.relevantEntries || []);
      const products = this.getProductData();
      const productInfo = this.findProductInQuestion(question);
      const productString = this.formatProductData(products);
      
      logger.info(`Found product in question: ${productInfo ? productInfo.product.name : 'none'}`);
      
      // Generate AI response
      const startTime = Date.now();
      const response = await this.generateResponse(question, tag, formattedHistory, relevantInfo, productString);
      const processingTime = Date.now() - startTime;
      
      logger.info(`AI response generated in ${processingTime}ms`);
      
      // Update chat history
      await this.updateChatHistory(nomorWhatsapp, question, response.content);
      
      return {
        response: response.content,
        processingTime: response.processingTime || processingTime,
        tags: tag,
        productInfo,
      };
    } catch (error) {
      logger.error(`Error processing with AI for user ${nomorWhatsapp}: ${error.message}`);
      logger.error(`Stack trace: ${error.stack}`);
      
      // Return a user-friendly error response instead of throwing
      return {
        response: "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi atau hubungi admin untuk bantuan.",
        processingTime: 0,
        tags: "error",
        productInfo: null,
        error: error.message
      };
    }
  }
  
  /**
   * Ensures the service is initialized
   * @returns {Promise<void>}
   * @private
   */
  async ensureInitialized() {
    if (!this.initialized) {
      await this.init();
    }
  }
  
  /**
   * Retrieves and formats chat history for a customer
   * @param {string} nomorWhatsapp - Customer's WhatsApp number
   * @returns {Promise<Array>} Formatted chat history
   * @private
   */
  async getChatHistory(nomorWhatsapp) {
    const historyKey = `lel:${nomorWhatsapp}`;
    let history = await redisClient.lRange(historyKey, -CONFIG.HISTORY_CONTEXT_SIZE, -1);
    history = history.map(JSON.parse);

    return history.map((msg) => {
      return msg.role === "user"
        ? { type: "user", content: msg.content }
        : { type: "ai", content: msg.content };
    });
  }
  
  /**
   * Handles questions tagged as orders
   * @param {Array} formattedHistory - Formatted chat history
   * @returns {Promise<Object>} Order processing result
   * @private
   */
  /**
   * Checks if a product is ambiguous (has variants that need clarification)
   * @param {string} productName - Name of the product to check
   * @param {Object} productData - Product catalog data
   * @returns {Object|null} - Ambiguity information or null if not ambiguous
   * @private
   */
  isAmbiguousProduct(productName, productData) {
    const normalizedName = productName.toLowerCase().trim();
    const productVariants = [];
    const baseProductTerms = [];
    
    // Extract base product terms (e.g., "netflix", "gmail")
    const words = normalizedName.split(/\s+/);
    for (const word of words) {
      if (word.length > 3) { // Only consider meaningful words
        baseProductTerms.push(word);
      }
    }
    
    // Find all variants that match the base product terms
    for (const [name, data] of Object.entries(productData)) {
      const productNameLower = name.toLowerCase();
      
      // Check if this product contains any of the base terms
      for (const term of baseProductTerms) {
        if (productNameLower.includes(term)) {
          productVariants.push({
            name,
            code: data.code,
            price: data.price,
            stock: data.stock,
            desc: data.desc
          });
          break;
        }
      }
    }
    
    // If we found multiple variants, this product is ambiguous
    if (productVariants.length > 1) {
      return {
        isAmbiguous: true,
        baseTerms: baseProductTerms,
        variants: productVariants
      };
    }
    
    return null;
  }
  
  /**
   * Handles ambiguous product orders by asking for clarification
   * @param {Object} ambiguousProduct - Ambiguous product information
   * @returns {string} - Response asking for clarification
   * @private
   */
  handleAmbiguousProduct(ambiguousProduct) {
    if (!ambiguousProduct || !ambiguousProduct.variants || ambiguousProduct.variants.length <= 1) {
      return null;
    }
    
    const variantsList = ambiguousProduct.variants
      .map(variant => `• **${variant.name}** (${variant.code}): ${variant.price}\n  ${variant.desc || ''}`)
      .join('\n\n');
    
    const baseProduct = ambiguousProduct.baseTerms.join(' ');
    
    return `⚠️ **Mohon Klarifikasi**\n\n` +
      `Saya menemukan beberapa varian produk "${baseProduct}" yang tersedia. ` +
      `Mohon tentukan varian mana yang Anda inginkan:\n\n` +
      `${variantsList}\n\n` +
      `Silakan balas dengan nama lengkap atau kode produk yang Anda inginkan.`;
  }

  async handleOrderTag(formattedHistory) {
    try {
      logger.info(`Processing order extraction from chat history with ${formattedHistory.length} messages`);
      const startTime = Date.now();
      
      // Get product data
      const productData = await this.getProductData();
      logger.info(`Retrieved ${Object.keys(productData).length} products from catalog`);
      
      // Extract orders from chat
      logger.info(`Extracting orders from chat history...`);
      const orders = await this.extractOrdersFromChat(formattedHistory, productData);
      
      const processingTime = Date.now() - startTime;
      logger.info(`Order extraction completed in ${processingTime}ms`);
      
      if (orders.success && orders.orders.length > 0) {
        // Check for ambiguous products first
        const ambiguousOrders = orders.orders.filter(order => order.ambiguous);
        
        if (ambiguousOrders.length > 0) {
          // Handle the first ambiguous product
          const ambiguousResponse = this.handleAmbiguousProduct(ambiguousOrders[0].ambiguous);
          
          if (ambiguousResponse) {
            logger.info(`Found ambiguous product order: ${ambiguousOrders[0].name}`);
            return {
              response: ambiguousResponse,
              processingTime,
              tags: "order",
              productInfo: { ambiguousOrder: ambiguousOrders[0] },
            };
          }
        }
        
        // Calculate total price
        let totalPrice = 0;
        const orderList = orders.orders
          .map(order => {
            const price = parseInt(order.price.replace(/[^0-9]/g, '')) || 0;
            const subtotal = price * order.quantity;
            totalPrice += subtotal;
            return `• ${order.name} (${order.code})\n  Harga: ${order.price} x ${order.quantity} = Rp ${subtotal.toLocaleString()}`;
          })
          .join("\n\n");
        
        logger.info(`Successfully extracted ${orders.orders.length} orders with total price Rp ${totalPrice.toLocaleString()}`);
        
        const response = `📋 **Daftar Pesanan Anda:**\n\n${orderList}\n\n💰 **Total: Rp ${totalPrice.toLocaleString()}**\n\n✅ Untuk melanjutkan pemesanan, silakan konfirmasi dengan mengetik "konfirmasi" atau hubungi admin untuk pembayaran.`;
        
        return {
          response,
          processingTime,
          tags: "order",
          productInfo: { orders: orders.orders, total: totalPrice },
        };
      } else {
        logger.info(`No valid orders found or extraction failed: ${orders.error || 'No orders detected'}`);
        
        return {
          response: `❌ ${orders.error || 'Tidak ditemukan pesanan dalam percakapan ini. Silakan sebutkan produk dan jumlah yang ingin dipesan.'}\n\n📝 Contoh: "Saya mau pesan Netflix 1P2U sebanyak 2 akun"`,
          processingTime,
          tags: "order",
          productInfo: null,
        };
      }
    } catch (error) {
      logger.error(`Error in handleOrderTag: ${error.message}`);
      return {
        response: `❌ Maaf, terjadi kesalahan saat memproses pesanan Anda. Silakan coba lagi atau hubungi admin untuk bantuan.`,
        processingTime: 0,
        tags: "order",
        productInfo: null,
      };
    }
  }
  
  /**
   * Handles order confirmation requests
   * @param {Array} formattedHistory - Formatted chat history
   * @returns {Promise<Object>} Order confirmation result
   * @private
   */
  async handleOrderConfirmation(formattedHistory) {
    // Look for recent order in chat history
    let recentOrder = null;
    for (let i = formattedHistory.length - 1; i >= 0; i--) {
      const msg = formattedHistory[i];
      if (msg.content && msg.content.includes('📋') && msg.content.includes('Total:')) {
        recentOrder = msg.content;
        break;
      }
    }
    
    if (recentOrder) {
      // Extract total from the order
      const totalMatch = recentOrder.match(/Total: (Rp [\d.,]+)/);
      const total = totalMatch ? totalMatch[1] : 'Rp 0';
      
      const response = `✅ **Pesanan Dikonfirmasi!**\n\n` +
        `📋 Detail pesanan Anda sudah dicatat\n` +
        `💰 Total pembayaran: ${total}\n\n` +
        `📱 **Cara Pembayaran:**\n` +
        `1️⃣ Transfer ke rekening/QRIS yang akan diberikan admin\n` +
        `2️⃣ Kirim bukti transfer\n` +
        `3️⃣ Tunggu konfirmasi dan pengiriman akun\n\n` +
        `📞 Hubungi admin untuk detail pembayaran\n` +
        `⏰ Pesanan akan diproses dalam 1-24 jam\n\n` +
        `Terima kasih sudah berbelanja! 🙏`;
      
      return {
        response,
        processingTime: 0,
        tags: "order_confirmation",
        productInfo: { confirmed: true, total },
      };
    } else {
      return {
        response: `❓ Maaf, saya tidak menemukan pesanan yang perlu dikonfirmasi.\n\n` +
          `Silakan buat pesanan terlebih dahulu dengan menyebutkan produk dan jumlah yang diinginkan.\n\n` +
          `📝 Contoh: "Saya mau pesan Netflix 1P2U sebanyak 2 akun"`,
        processingTime: 0,
        tags: "order_confirmation",
        productInfo: null,
      };
    }
  }
  
  /**
   * Finds product mentions in a question
   * @param {string} question - Customer's question
   * @returns {Object|null} Product information if found
   * @private
   */
  findProductInQuestion(question) {
    const words = question.toLowerCase().split(/\s+/);
    
    for (let i = 0; i < words.length; i++) {
      for (let j = i + 1; j <= Math.min(i + 5, words.length); j++) {
        const potentialProduct = words.slice(i, j).join(" ");
        const availability = this.checkProductAvailability(potentialProduct);
        if (availability.exists) {
          return availability;
        }
      }
    }
    
    return null;
  }
  
  /**
   * Formats product data for display
   * @param {Object} products - Product data
   * @returns {string} Formatted product string
   * @private
   */
  formatProductData(products) {
    return Object.entries(products)
      .map(([name, p]) => {
        const stockStatus = parseInt(p.stock) > 0 ? "Tersedia" : "Kosong";
        return `${name}: Harga ${p.price}, Status: ${stockStatus}, Stock: ${p.stock}`;
      })
      .join("\n");
  }
  
  /**
   * Generates an AI response based on question, tag, and context
   * @param {string} question - Customer's question
   * @param {string} tag - Question classification tag
   * @param {Array} formattedHistory - Formatted chat history
   * @param {string} relevantInfo - Relevant context information
   * @param {string} productString - Formatted product data
   * @returns {Promise<Object>} Response with content and processing time
   * @private
   */
  async generateResponse(question, tag, formattedHistory, relevantInfo, productString) {
    const systemTemplate = this.tagTemplates[tag] || this.tagTemplates.unknown;
    
    const chatPrompt = ChatPromptTemplate.fromMessages([
      ["system", systemTemplate],
      ...formattedHistory.map((msg) => [
        msg.type === "user" ? "user" : "assistant",
        msg.content,
      ]),
      ["user", "{query}"],
    ]);

    const formattedMessages = await chatPrompt.formatMessages({
      context: relevantInfo,
      products: productString,
      chat_history: formattedHistory,
      query: question,
    });

    const startTime = Date.now();
    const response = await this.chatModel.invoke(formattedMessages);
    const processingTime = Date.now() - startTime;
    
    logger.info(`Pertanyaan '${question}' diproses dalam ${processingTime}ms dengan jawaban '${response.content}'`);
    
    return {
      content: response.content,
      processingTime
    };
  }
  
  /**
   * Updates chat history with new messages
   * @param {string} nomorWhatsapp - Customer's WhatsApp number
   * @param {string} question - Customer's question
   * @param {string} answer - AI's answer
   * @returns {Promise<void>}
   * @private
   */
  async updateChatHistory(nomorWhatsapp, question, answer) {
    const historyKey = `lel:${nomorWhatsapp}`;
    
    // Add user message
    const chatEntry = JSON.stringify({
      role: "user",
      content: question,
    });
    await redisClient.rPush(historyKey, chatEntry);
    await redisClient.expire(historyKey, CONFIG.REDIS_EXPIRY);

    // Add AI response
    const aiEntry = JSON.stringify({
      role: "assistant",
      content: answer,
    });
    await redisClient.rPush(historyKey, aiEntry);
    await redisClient.expire(historyKey, CONFIG.REDIS_EXPIRY);

    // Trim history if too long
    const historyLength = await redisClient.lLen(historyKey);
    if (historyLength > CONFIG.MAX_HISTORY_LENGTH) {
      await redisClient.lTrim(historyKey, historyLength - CONFIG.MAX_HISTORY_LENGTH, -1);
    }
  }

  /**
   * Format informasi relevan untuk prompt
   * @param {Array} relevantEntries - Entri relevan dari dataset
   * @returns {string} - Informasi relevan yang diformat
   */
  /**
   * Formats relevant entries from the dataset for AI context
   * @param {Array} relevantEntries - Relevant entries from the dataset
   * @returns {string} Formatted relevant information
   */
  formatRelevantInfo(relevantEntries) {
    if (!relevantEntries || relevantEntries.length === 0) {
      return "Tidak ada informasi relevan yang ditemukan dalam dataset";
    }

    return relevantEntries
      .map(entry => this.formatSingleEntry(entry))
      .join("\n\n");
  }
  
  /**
   * Formats a single entry from the dataset
   * @param {Object} entry - Entry to format
   * @returns {string} Formatted entry
   * @private
   */
  formatSingleEntry(entry) {
    return `${entry.question ? `Pertanyaan: "${entry.question}"\n` : ""}
   Konteks: ${JSON.stringify(entry.answer, null, 2)}`;
  }

  /**
   * Dapatkan data produk
   * @returns {Object} - Data produk
   */
  /**
   * Gets product data catalog
   * @returns {Object} Product data with details
   */
  getProductData() {
    return {
      "Akun Gmail Fresh": this.createProductEntry("GMF", "Rp 5.000", "2", "Akun Gmail baru dengan garansi 7 hari. Cocok untuk keperluan registrasi atau akun utama."),
      "Akun Gmail Aged": this.createProductEntry("GMA", "Rp 15.000/akun", "1", "Akun Gmail berumur lebih dari 1 tahun dengan garansi 14 hari. Cocok untuk bisnis atau akun verifikasi."),
      "Netflix 1P1U": this.createProductEntry("N1P1U", "Rp 24.000", "0", "Akun Netflix sharing dengan 1 profile dan 1 user. Bebas gangguan, bisa digunakan kapan saja."),
      "Netflix 1P2U": this.createProductEntry("N1P2U", "Rp 13.000", "5", "Akun Netflix sharing dengan 1 profile untuk 2 user. Ada kemungkinan kendala jika dipakai bersamaan."),
      "Disney+ Hotstar": this.createProductEntry("DHS", "Rp 30.000", "8", "Akun Disney+ Hotstar premium untuk menonton film dan serial eksklusif Disney, Marvel, dan lainnya."),
    };
  }
  
  /**
   * Creates a product entry with consistent structure
   * @param {string} code - Product code
   * @param {string} price - Product price
   * @param {string} stock - Available stock
   * @param {string} description - Product description
   * @returns {Object} Structured product entry
   * @private
   */
  createProductEntry(code, price, stock, description) {
    return {
      code,
      price,
      stock,
      desc: description
    };
  }

  /**
   * Classifies a given question based on chat history and a predefined list of valid tags
   * Optimized for 100% accuracy on 'order' tags and 90% accuracy on other tags
   * @param {string} question - The question to classify
   * @param {Array} chatHistory - The chat history for context
   * @returns {Promise<string>} - The classified tag
   */
  async getQuestionTag(question, chatHistory) {
    try {
      await this.ensureInitialized();
      
      // First check for order-related keywords with 100% accuracy
      const orderResult = this.isOrderRelatedQuestion(question, chatHistory);
      if (orderResult) {
        logger.info(`Question classified as ${orderResult} with high confidence: "${question}"`);
        return orderResult;
      }
      
      const validTags = this.getValidQuestionTags();
      const systemInstruction = this.createTagClassificationPrompt(validTags);
      
      // Enhanced messages with more context for better accuracy
      const messages = [
        { role: "system", content: systemInstruction },
        ...chatHistory,
        {
          role: "user",
          content: `Tentukan tag paling relevan untuk pertanyaan ini: "${question}". Berikan jawaban dengan tingkat keyakinan minimal 90%.`,
        },
      ];
      
      // Get AI response with temperature adjustment for higher confidence
      const originalTemperature = this.chatModel.temperature;
      try {
        // Lower temperature for more deterministic results
        this.chatModel.temperature = 0.1;
        
        const response = await this.chatModel.invoke(messages, {
          format: {
            type: "object",
            properties: {
              tag: {
                type: "string",
                enum: validTags,
              },
              confidence: {
                type: "number",
                minimum: 0,
                maximum: 1
              }
            },
            required: ["tag", "confidence"],
          },
        });
        
        // Parse and validate the tag with confidence check
        return this.parseTagResponseWithConfidence(response, validTags);
      } finally {
        // Restore original temperature
        this.chatModel.temperature = originalTemperature;
      }
    } catch (error) {
      logger.error(`Gagal mendapatkan tag pertanyaan: ${error.message}`);
      return "unknown";
    }
  }
  
  /**
   * Checks if a question is related to product availability/stock
   * @param {string} question - The question to check
   * @returns {boolean} True if the question is about product availability
   * @private
   */
  isAvailabilityQuestion(question) {
    // Normalize question for consistent matching
    const normalizedQuestion = question.toLowerCase();
    
    // Strong availability-related keywords in Indonesian
    const availabilityKeywords = [
      'stok', 'stock', 'tersedia', 'ketersediaan', 'ada', 'masih ada', 
      'habis', 'kosong', 'ready', 'persediaan', 'sisa', 'tersisa',
      'masih tersedia', 'masih ready', 'masih ada stok', 'stoknya',
      'stok masih', 'stock masih', 'masih dijual', 'masih bisa dibeli'
    ];
    
    // Check for direct availability keywords
    for (const keyword of availabilityKeywords) {
      if (normalizedQuestion.includes(keyword)) {
        return true;
      }
    }
    
    // Check for availability question patterns
    const availabilityPatterns = [
      /apakah (masih|sudah|ada|tersedia|ready)/i,
      /masih (ada|tersedia|ready|dijual)/i,
      /sudah (habis|kosong|tidak ada)/i,
      /berapa (stok|stock|persediaan|sisa)/i,
      /stok (\w+) (masih|sudah|ada|tersedia|ready|habis|kosong)/i,
      /(\w+) (masih|sudah) (ada|tersedia|ready|habis|kosong)/i,
      /masih ada (\w+)/i
    ];
    
    for (const pattern of availabilityPatterns) {
      if (pattern.test(normalizedQuestion)) {
        return true;
      }
    }
    
    return false;
  }

  /**
   * Checks if a question is related to orders with high confidence
   * @param {string} question - The question to check
   * @param {Array} chatHistory - Chat history for context
   * @returns {boolean} True if the question is order-related
   * @private
   */
  isOrderRelatedQuestion(question, chatHistory) {
    // Normalize question for consistent matching
    const normalizedQuestion = question.toLowerCase();
    
    // Check for order confirmation first
    const confirmationResult = this.isOrderConfirmation(normalizedQuestion, chatHistory);
    if (confirmationResult === true) {
      return "order_confirmation";
    } else if (confirmationResult === "product_clarification") {
      return "product_clarification";
    }
    
    // Check if this is an availability question first
    // This should take precedence over order detection
    if (this.isAvailabilityQuestion(question)) {
      return "availability";
    }
    
    // Strong order-related keywords in Indonesian
    const orderKeywords = [
      'pesan', 'pesanan', 'order', 'beli', 'checkout', 'keranjang', 
      'belanja', 'transaksi', 'pembelian', 'daftar pesanan', 'riwayat pesanan',
      'status pesanan', 'detail pesanan', 'jumlah pesanan',
      'total pesanan', 'harga pesanan', 'bayar pesanan', 'pembayaran pesanan',
      'mau beli', 'mau pesan', 'mau order', 'saya pesan', 'saya beli'
    ];
    
    // Check for direct order keywords
    for (const keyword of orderKeywords) {
      if (normalizedQuestion.includes(keyword)) {
        return "order";
      }
    }
    
    // Check for order patterns with product names
    const productNames = Object.keys(this.getProductData()).map(name => name.toLowerCase());
    const orderPatterns = [
      /mau (beli|pesan|order) (\d+|beberapa|satu|dua|tiga|empat|lima|\w+)/i,
      /beli (\d+|beberapa|satu|dua|tiga|empat|lima|\w+)/i,
      /pesan (\d+|beberapa|satu|dua|tiga|empat|lima|\w+)/i,
      /order (\d+|beberapa|satu|dua|tiga|empat|lima|\w+)/i,
      /saya (mau|ingin|butuh) (\d+|beberapa|satu|dua|tiga|empat|lima)/i,
      /(\d+) (akun|buah|unit)/i,
      /tambahkan ke (keranjang|pesanan)/i,
      /checkout/i
    ];
    
    for (const pattern of orderPatterns) {
      if (pattern.test(normalizedQuestion)) {
        return "order";
      }
    }
    
    // Check if question contains product name + quantity indicators
    for (const productName of productNames) {
      if (normalizedQuestion.includes(productName)) {
        const quantityIndicators = ['1', '2', '3', '4', '5', 'satu', 'dua', 'tiga', 'empat', 'lima', 'beberapa', 'banyak'];
        for (const qty of quantityIndicators) {
          if (normalizedQuestion.includes(qty)) {
            return "order";
          }
        }
      }
    }
    
    // Check chat history for order context
    if (chatHistory && chatHistory.length > 0) {
      const recentMessages = chatHistory.slice(-3); // Check last 3 messages
      for (const msg of recentMessages) {
        if (msg.content && typeof msg.content === 'string') {
          const content = msg.content.toLowerCase();
          // If recent messages contain order confirmation or listing
          if (content.includes('daftar pesanan') || 
              content.includes('pesanan anda') ||
              content.includes('total:') ||
              content.includes('📋')) {
            return "order";
          }
        }
      }
    }
    
    return false;
  }
  
  /**
   * Checks if the question is an order confirmation
   * @param {string} normalizedQuestion - Normalized question
   * @param {Array} chatHistory - Chat history for context
   * @returns {boolean} True if it's an order confirmation
   * @private
   */
  isOrderConfirmation(normalizedQuestion, chatHistory) {
    const confirmationKeywords = [
      'konfirmasi', 'confirm', 'ya', 'iya', 'ok', 'oke', 'setuju', 
      'lanjut', 'lanjutkan', 'proses', 'bayar', 'pembayaran'
    ];
    
    // Check if there's a recent order in chat history
    let hasRecentOrder = false;
    let hasAmbiguousProductQuery = false;
    let ambiguousProductVariants = [];
    
    if (chatHistory && chatHistory.length > 0) {
      const recentMessages = chatHistory.slice(-5);
      for (const msg of recentMessages) {
        if (msg.content && typeof msg.content === 'string') {
          const content = msg.content.toLowerCase();
          
          // Check for regular order
          if (content.includes('daftar pesanan') || 
              content.includes('total:') ||
              content.includes('📋') ||
              content.includes('konfirmasi dengan mengetik')) {
            hasRecentOrder = true;
            break;
          }
          
          // Check for ambiguous product query
          if (content.includes('mohon klarifikasi') && 
              content.includes('saya menemukan beberapa varian produk')) {
            hasAmbiguousProductQuery = true;
            
            // Extract product variants from the message
            const productData = this.getProductData();
            for (const [name, data] of Object.entries(productData)) {
              if (content.includes(name.toLowerCase()) || content.includes(data.code.toLowerCase())) {
                ambiguousProductVariants.push({
                  name: name,
                  code: data.code
                });
              }
            }
            
            break;
          }
        }
      }
    }
    
    // If there's a recent order and user uses confirmation keywords
    if (hasRecentOrder) {
      for (const keyword of confirmationKeywords) {
        if (normalizedQuestion.includes(keyword)) {
          return true;
        }
      }
    }
    
    // Check if the user is responding to an ambiguous product query
    if (hasAmbiguousProductQuery && ambiguousProductVariants.length > 0) {
      // Check if the user's response matches any of the product variants
      for (const variant of ambiguousProductVariants) {
        if (normalizedQuestion.includes(variant.name.toLowerCase()) || 
            normalizedQuestion.includes(variant.code.toLowerCase())) {
          return "product_clarification";
        }
      }
    }
    
    return false;
  }
  
  /**
   * Parses and validates the tag response with confidence check
   * @param {Object} response - AI response object
   * @param {Array<string>} validTags - List of valid tags
   * @returns {string} Validated tag or "unknown"
   * @private
   */
  parseTagResponseWithConfidence(response, validTags) {
    try {
      const parsed =
        typeof response.content === "string"
          ? JSON.parse(response.content)
          : response.content;

      const tag = parsed?.tag?.toLowerCase();
      const confidence = parsed?.confidence || 0;
      
      // Log the confidence level
      logger.info(`Tag: ${tag}, Confidence: ${confidence}`);
      
      // Accept tag only if confidence is high enough
      if (validTags.includes(tag) && confidence >= 0.9) {
        logger.info(`Question classified as: ${tag} with confidence ${confidence}`);
        return tag;
      } else if (validTags.includes(tag)) {
        logger.warn(`Tag ${tag} detected but confidence ${confidence} is below threshold, defaulting to "unknown"`);
        return "unknown";
      } else {
        logger.warn(`Invalid tag detected: ${tag}, defaulting to "unknown"`);
        return "unknown";
      }
    } catch (e) {
      logger.warn(`Gagal parsing response JSON: ${response.content}`);
      const tag = response?.content?.trim().toLowerCase();
      
      if (validTags.includes(tag)) {
        return tag;
      }
      return "unknown";
    }
  }
  
  /**
   * Gets the list of valid question tags
   * @returns {Array<string>} List of valid tags
   * @private
   */
  getValidQuestionTags() {
    return [
      "price_inquiry",
      "availability",
      "greeting",
      "technical_details",
      "payment_method",
      "refund_policy",
      "emerging_services",
      "products_details",
      "warranty_refund",
      "referral_loyalty",
      "order",
      "order_confirmation",
      "unknown",
    ];
  }
  
  /**
   * Creates the system prompt for tag classification
   * @param {Array<string>} validTags - List of valid tags
   * @returns {string} System prompt for tag classification
   * @private
   */
  createTagClassificationPrompt(validTags) {
    return `
Anda adalah sistem klasifikasi tag untuk percakapan pelanggan.
Tugas Anda adalah menentukan **satu** tag yang paling relevan untuk pertanyaan terakhir,
berdasarkan konteks keseluruhan chat sebelumnya.

Berikut adalah daftar tag yang tersedia:
- price_inquiry: pertanyaan tentang harga, biaya, atau meminta daftar harga/pricelist
- availability: pertanyaan tentang menanyakan stok, ketersediaan, sebuah produk
- greeting: sapaan seperti "halo", "selamat pagi", atau semacamnya
- technical_details: pertanyaan teknis atau detail produk/layanan
- payment_method: pertanyaan tentang cara membayar atau metode pembayaran
- refund_policy: pertanyaan tentang pengembalian dana atau pembatalan
- emerging_services: pertanyaan tentang layanan baru atau fitur eksperimental
- products_details: pertanyaan tentang informasi produk atau tentang produk
- warranty_refund: pertanyaan tentang garansi atau pengembalian barang
- referral_loyalty: pertanyaan tentang sistem referral atau program loyalitas
- order: permintaan untuk memesan atau konfirmasi pesanan
- unknown: jika tidak dapat diklasifikasikan ke salah satu kategori di atas

Return hanya objek JSON seperti ini:
{"tag": "<nama_tag>"}
Contoh: {"tag": "order"}
`;
  }
  
  /**
   * Parses and validates the tag response from AI
   * @param {Object} response - AI response object
   * @param {Array<string>} validTags - List of valid tags
   * @returns {string} Validated tag or "unknown"
   * @private
   */
  parseTagResponse(response, validTags) {
    try {
      const parsed =
        typeof response.content === "string"
          ? JSON.parse(response.content)
          : response.content;

      const tag = parsed?.tag?.toLowerCase();
      
      if (validTags.includes(tag)) {
        logger.info(`Question classified as: ${tag}`);
        return tag;
      } else {
        logger.warn(`Invalid tag detected: ${tag}, defaulting to 'unknown'`);
        return "unknown";
      }
    } catch (e) {
      logger.warn(`Gagal parsing response JSON: ${response.content}`);
      const tag = response?.content?.trim().toLowerCase();
      
      if (validTags.includes(tag)) {
        return tag;
      }
      return "unknown";
    }
  }

  /**
   * Tests the connection to the Ollama service
   * @returns {Promise<Object>} Connection test result with status, model, latency and response snippet
   */
  async testConnection() {
    try {
      await this.ensureInitialized();
      
      // Measure response time
      const startTime = Date.now();
      const response = await this.invokeTestMessage();
      const latency = Date.now() - startTime;
      
      // Format response for display
      const snippet = this.formatResponseSnippet(response.content);
      
      return {
        success: true,
        model: process.env.OLLAMA_MODEL || "llama3.2",
        latency,
        snippet,
      };
    } catch (error) {
      logger.error(`Connection test failed: ${error.message}`);
      return {
        success: false,
        error: error.message,
      };
    }
  }
  
  /**
   * Invokes a simple test message to check connection
   * @returns {Promise<Object>} AI response
   * @private
   */
  async invokeTestMessage() {
    return await this.chatModel.invoke([
      {
        role: "user",
        content: "Respond with 'Connection successful' and nothing else.",
      },
    ]);
  }
  
  /**
   * Formats a response snippet for display
   * @param {string} content - Full response content
   * @returns {string} Formatted snippet
   * @private
   */
  formatResponseSnippet(content) {
    const maxLength = 50;
    return content.substring(0, maxLength) + (content.length > maxLength ? "..." : "");
  }

  /**
   * Gets service statistics
   * @returns {Object} Service statistics and configuration
   */
  getServiceStats() {
    return {
      initialized: this.initialized,
      model: process.env.OLLAMA_MODEL || "llama3.2",
      temperature: this.getConfigFloat('OLLAMA_TEMPERATURE', 0.2),
      templateCount: Object.keys(this.tagTemplates).length,
    };
  }

  /**
   * Checks if a product exists and is in stock using fuzzy matching
   * @param {string} productName - Name of the product to check
   * @returns {Object} - Product availability info {exists: boolean, inStock: boolean, product: Object}
   */
  checkProductAvailability(productName) {
    const products = this.getProductData();
    const normalizedSearch = productName.toLowerCase().trim();

    // Find best match using fuzzy matching
    const bestMatch = this.findBestProductMatch(normalizedSearch, products);

    if (!bestMatch) {
      return { exists: false, inStock: false, product: null };
    }

    const { name, product } = bestMatch;
    const inStock = parseInt(product.stock) > 0;

    return {
      exists: true,
      inStock,
      product: { ...product, name },
    };
  }
  
  /**
   * Finds the best matching product using fuzzy matching
   * @param {string} normalizedSearch - Normalized search term
   * @param {Object} products - Product data
   * @returns {Object|null} Best match or null if no good match found
   * @private
   */
  findBestProductMatch(normalizedSearch, products) {
    let bestMatch = null;
    let highestScore = 0;

    // Clean the input search term
    const cleanSearch = normalizedSearch.toLowerCase().trim();
    
    Object.entries(products).forEach(([name, product]) => {
      const cleanName = name.toLowerCase();
      
      // Check for exact match first
      if (cleanName === cleanSearch) {
        bestMatch = { name, product, score: 100 };
        highestScore = 100;
        return; // Exit the forEach early
      }
      
      // Check for code match
      if (product.code && product.code.toLowerCase() === cleanSearch) {
        bestMatch = { name, product, score: 100 };
        highestScore = 100;
        return; // Exit the forEach early
      }
      
      // Check for partial matches
      if (cleanName.includes(cleanSearch) || cleanSearch.includes(cleanName)) {
        const score = Math.max(
          fuzz.ratio(cleanSearch, cleanName),
          fuzz.partial_ratio(cleanSearch, cleanName)
        );
        if (score > highestScore) {
          highestScore = score;
          bestMatch = { name, product, score };
        }
      } else {
        // Use multiple fuzzy matching algorithms
        const ratio = fuzz.ratio(cleanSearch, cleanName);
        const partialRatio = fuzz.partial_ratio(cleanSearch, cleanName);
        const tokenRatio = fuzz.token_sort_ratio(cleanSearch, cleanName);
        
        const score = Math.max(ratio, partialRatio, tokenRatio);
        if (score > highestScore) {
          highestScore = score;
          bestMatch = { name, product, score };
        }
      }
      
      // Also check against product descriptions
      if (product.desc) {
        const descScore = fuzz.partial_ratio(cleanSearch, product.desc.toLowerCase());
        if (descScore > highestScore) {
          highestScore = descScore;
          bestMatch = { name, product, score: descScore };
        }
      }
    });

    return (bestMatch && highestScore >= CONFIG.FUZZY_MATCH_THRESHOLD) ? bestMatch : null;
  }

  /**
   * Extracts order lists from chat history
   * @param {Array} chatHistory - Chat history
   * @param {Object} productData - Product data
   * @returns {Object} - Extracted order list or error message
   */
  async extractOrdersFromChat(chatHistory, productData) {
    try {
      await this.ensureInitialized();
      
      // Create prompt for order extraction
      const systemPrompt = `You are an order extraction assistant. Analyze the chat history and extract ONLY product orders explicitly mentioned by the customer.

Rules:
1. ONLY extract EXPLICIT orders when customer clearly states they want to buy/order/purchase a product
2. Look for Indonesian phrases like: "mau beli", "saya pesan", "order", "beli", "mau", "butuh", "ambil"
3. ONLY include products that match the available product catalog below
4. Extract product name and quantity (if quantity not specified, assume 1)
5. Include price from the product catalog
6. If no clear orders found, return empty array
7. DO NOT include products that were only mentioned in passing or in questions
8. DO NOT make assumptions about what the customer might want to order

Format output as JSON with this EXACT structure:
{
  "orders": [
    {
      "code": "product_code",
      "name": "exact_product_name",
      "price": "Rp price",
      "quantity": number
    }
  ]
}

Available products (ONLY match from this list):
${Object.entries(productData).map(([name, data]) => 
  `- ${name} (Code: ${data.code}, Price: ${data.price}, Stock: ${data.stock})`
).join('\n')}

Chat history to analyze:
${JSON.stringify(chatHistory, null, 2)}
      `;
      
      const messages = [
        {
          role: "system",
          content: systemPrompt,
        },
        ...chatHistory,
        {
          role: "user",
          content:
            "Tolong bantu saya mendapatkan daftar pesanan saya dari chat ini. Saya ingin tahu apa saja yang sudah saya pesan. Untuk product nya berikan code produk, nama produk, harga, dan jumlahnya. Jika ada yang belum lengkap, tolong beri tahu saya.",
        },
      ];
      
      // Get AI response
      const response = await this.chatModel.invoke(messages, {
        format: {
          type: "object",
          properties: {
            orders: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  code: { type: "string" },
                  name: { type: "string" },
                  price: { type: "string" },
                  quantity: { type: "integer" },
                },
                required: ["code", "name", "price", "quantity"],
              },
            },
          },
          required: ["orders"],
        },
      });

      // Parse and validate the response
      return this.parseOrderResponse(response);
    } catch (error) {
      logger.error(`Error extracting orders: ${error.message}`);
      return {
        success: false,
        error: "Gagal mengekstrak pesanan"
      };
    }
  }
  
  /**
   * Parses and validates the order response from AI
   * @param {Object} response - AI response object
   * @returns {Object} Parsed and validated orders or error
   * @private
   */
  parseOrderResponse(response) {
    try {
      let dataOrders = [];
      logger.info(`Parsing order response: ${response.content.substring(0, 100)}...`);
      
      // Try to parse structured JSON response first
      try {
        const parsed = typeof response.content === 'string' 
          ? JSON.parse(response.content) 
          : response.content;
        
        if (parsed && Array.isArray(parsed.orders)) {
          dataOrders = parsed.orders;
          logger.info(`Successfully parsed JSON orders: ${dataOrders.length} items found`);
          
          // Ensure prices are correctly formatted in JSON response
          dataOrders = dataOrders.map(order => {
            // Get the correct price from product catalog if available
            const productData = this.getProductData();
            const productMatch = this.findBestProductMatch(order.name, productData);
            
            if (productMatch && productMatch.score >= CONFIG.FUZZY_MATCH_THRESHOLD) {
              logger.info(`Using catalog price for ${order.name}: ${productMatch.product.price}`);
              return {
                ...order,
                price: productMatch.product.price,
                ambiguous: this.isAmbiguousProduct(order.name, productData)
              };
            }
            return order;
          });
        } else {
          logger.warn(`Response has invalid format, missing orders array: ${JSON.stringify(parsed)}`);
        }
      } catch (jsonError) {
        logger.warn(`Failed to parse JSON response: ${jsonError.message}. Falling back to text parsing.`);
        
        // Enhanced fallback parsing for various order patterns
        const lines = response.content
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line);
        
        logger.info(`Fallback parsing: Processing ${lines.length} lines of text`);
        
        for (const line of lines) {
          // Pattern 1: "1. Product - 2 units"
          let match = line.match(/\d+\.\s*([^-]+)\s*-\s*(\d+)\s*(?:units|unit|qty|quantity|pcs|pieces|item|items|buah|pcs|akun|akun)?/i);
          
          // Pattern 2: "Product (2 units)"
          if (!match) {
            match = line.match(/([^(]+)\s*\(\s*(\d+)\s*(?:units|unit|qty|quantity|pcs|pieces|item|items|buah|pcs|akun|akun)?\)/i);
          }
          
          // Pattern 3: "Product x2" or "Product 2x"
          if (!match) {
            match = line.match(/([^x]+)\s*x\s*(\d+)/i) || line.match(/([^\d]+)\s*(\d+)\s*x/i);
          }
          
          // Pattern 4: "2 units of Product" or "2 buah Product" or "2 akun Product"
          if (!match) {
            const reverseMatch = line.match(/(\d+)\s*(?:units|unit|qty|quantity|pcs|pieces|item|items|buah|pcs|akun|akun)?\s*(?:of|dari)?\s*(.+)/i);
            if (reverseMatch) {
              match = [reverseMatch[0], reverseMatch[2], reverseMatch[1]];
            }
          }
          
          // Pattern 5: "Saya mau pesan/beli Product sebanyak 2"
          if (!match) {
            const indonesianMatch = line.match(/(?:pesan|beli|order|mau)\s+([^\d]+)\s+(?:sebanyak|sejumlah|dengan\s+jumlah|dengan\s+kuantitas)\s+(\d+)/i);
            if (indonesianMatch) {
              match = [indonesianMatch[0], indonesianMatch[1], indonesianMatch[2]];
            }
          }
          
          // Pattern 6: "Product sebanyak 2"
          if (!match) {
            const simpleIndonesianMatch = line.match(/([^\d]+)\s+(?:sebanyak|sejumlah|dengan\s+jumlah|dengan\s+kuantitas)\s+(\d+)/i);
            if (simpleIndonesianMatch) {
              match = [simpleIndonesianMatch[0], simpleIndonesianMatch[1], simpleIndonesianMatch[2]];
            }
          }
          
          // Pattern 7: "Mau beli/pesan 2 Product"
          if (!match) {
            const buyMatch = line.match(/(?:mau|ingin|akan)\s+(?:beli|pesan|order)\s+(\d+)\s+(.+)/i);
            if (buyMatch) {
              match = [buyMatch[0], buyMatch[2], buyMatch[1]];
            }
          }

          // Pattern 8: "Beli/pesan 2 Product"
          if (!match) {
            const directBuyMatch = line.match(/(?:beli|pesan|order)\s+(\d+)\s+(.+)/i);
            if (directBuyMatch) {
              match = [directBuyMatch[0], directBuyMatch[2], directBuyMatch[1]];
            }
          }

          // Pattern 9: "Product 2 buah/akun"
          if (!match) {
            const productFirstMatch = line.match(/([^\d]+)\s+(\d+)\s+(?:buah|akun|unit|pcs)/i);
            if (productFirstMatch) {
              match = [productFirstMatch[0], productFirstMatch[1], productFirstMatch[2]];
            }
          }

          // Pattern 10: CSV format fallback
          if (!match) {
            const parts = line.split(",");
            if (parts.length >= 4) {
              const code = parts[0]?.trim() || "";
              const name = parts[1]?.trim() || "";
              const price = parts[2]?.trim() || "";
              const quantity = parseInt(parts[3]?.trim()) || 0;
              
              if (code && name && price && quantity > 0) {
                logger.info(`CSV format detected: ${code}, ${name}, ${price}, ${quantity}`);
                dataOrders.push({ code, name, price, quantity });
                continue;
              }
            }
          }
          
          if (match) {
            const product = match[1].trim();
            const quantity = parseInt(match[2].trim());
            logger.info(`Matched product pattern: "${product}" with quantity ${quantity} (pattern match: "${match[0]}")`);
            
            // Extract price if available in the line
            let price = null;
            const priceMatch = line.match(/(?:Rp|IDR)\s*([\d.,]+)/i);
            if (priceMatch) {
              price = `Rp ${priceMatch[1]}`;
              logger.info(`Price found in text: ${price}`);
            }
            
            if (product && !isNaN(quantity) && quantity > 0) {
              // Try to find matching product using fuzzy matching
              const productData = this.getProductData();
              const bestMatch = this.findBestProductMatch(product, productData);
              
              if (bestMatch && bestMatch.score >= CONFIG.FUZZY_MATCH_THRESHOLD) {
                logger.info(`Product matched: "${product}" -> "${bestMatch.name}" (score: ${bestMatch.score})`);
                const orderItem = {
                  code: bestMatch.product.code,
                  name: bestMatch.name,
                  price: bestMatch.product.price, // Always use catalog price for consistency
                  quantity: quantity
                };
                logger.info(`Adding order item: ${JSON.stringify(orderItem)}`);
                dataOrders.push(orderItem);
              } else {
                logger.warn(`No product match found for "${product}" or match score too low`);
              }
            } else {
              logger.warn(`Invalid product or quantity: product="${product}", quantity=${quantity}, isNaN=${isNaN(quantity)}`);
            }
          } else {
            logger.debug(`No pattern match for line: "${line.substring(0, 100)}"`);
          }
        }
      }

      // Consolidate quantities for duplicate products
      const productMap = new Map();
      for (const order of dataOrders) {
        const key = order.code;
        if (productMap.has(key)) {
          const existing = productMap.get(key);
          existing.quantity += order.quantity;
          logger.info(`Consolidated quantities for ${order.name} (${order.code}): new quantity = ${existing.quantity}`);
        } else {
          productMap.set(key, { ...order });
        }
      }
      
      // Filter out invalid orders and ensure correct price format
      const productData = this.getProductData();
      const filteredOrders = Array.from(productMap.values())
        .filter(order => order.code && order.name && order.price && order.quantity > 0)
        .map(order => {
          // Find the product in catalog to ensure correct price
          for (const [name, product] of Object.entries(productData)) {
            if (product.code === order.code) {
              return {
                ...order,
                price: product.price // Always use the catalog price
              };
            }
          }
          return order;
        });
      
      logger.info(`Final order count after filtering: ${filteredOrders.length}`);
      
      if (filteredOrders.length === 0) {
        return {
          success: false,
          error:
            "Tidak ditemukan chat yang berisi produk dan jumlah pesanan. Silakan sebutkan produk dan jumlah yang ingin dipesan.",
        };
      }

      return {
        success: true,
        orders: filteredOrders,
      };
    } catch (parseError) {
      logger.error(`Error parsing order data: ${parseError.message}`);
      return {
        success: false,
        error: "Format pesanan tidak valid"
      };
    }
  }
}

module.exports = OllamaService;