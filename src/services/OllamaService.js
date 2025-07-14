/**
 * OllamaService Module
 * Handles AI processing using Ollama models and LangChain integration
 * Manages chat history, product information, and template-based responses
 * Integrates with LangChain tools for order processing and product inquiries
 */

'use strict';

const { ChatOllama } = require("@langchain/ollama");
const logger = require("../utils/logger.js");
const { ChatPromptTemplate } = require("@langchain/core/prompts");
const redis = require('redis');
const BrainService = require('./BrainService');
const fuzz = require('fuzzball');
const { OrderProcessingTool, ProductInquiryTool } = require('../utils/OllamaTools');
const { initializeAgentExecutorWithOptions } = require('langchain/agents');
const { AgentExecutor } = require('langchain/agents');

// Configuration constants
const CONFIG = {
  REDIS_EXPIRY: 2 * 24 * 60 * 60, // 2 days in seconds
  MAX_HISTORY_LENGTH: 12,
  FUZZY_MATCH_THRESHOLD: 70,
  HISTORY_CONTEXT_SIZE: 12 // Number of recent messages to include in context
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
    this.orderProcessingTool = null;
    this.productInquiryTool = null;
    this.agentExecutor = null;
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
      await this.initializeTools();

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
      model: process.env.OLLAMA_MODEL || "llama3.2", // Sticking with your default, but consider trying others
      temperature: this.getConfigFloat("OLLAMA_TEMPERATURE", 0.2), // Slightly increased for more natural variation, but still controlled
      topK: this.getConfigInt("OLLAMA_TOP_K", 20), // Reduced to focus on more probable and relevant tokens
      topP: this.getConfigFloat("OLLAMA_TOP_P", 0.7), // Reduced for more focused and less random responses
      repeatPenalty: this.getConfigFloat("OLLAMA_REPEAT_PENALTY", 1.1), // Slightly lowered to allow for some repetition of key terms if necessary
      maxTokens: this.getConfigInt("OLLAMA_MAX_TOKENS", 450), // Reduced for conciseness in customer service responses
      keepAlive: "30m",
      stop: [
        "\nuser:",
        "\nassistant:",
        "\nAI:",
        "\nSystem:",
        "Customer:",
        "Agent:",
      ],
      system: `Kamu adalah CustoAI yang berasal dari Indonesia, asisten AI customer service yang ramah, profesional, dan sangat membantu dari toko "${
        process.env.NAME_STORE || "Toko Kami"
      }". Gaya bicaramu natural dan bersahabat seperti manusia, bukan robot kaku. Selalu gunakan sapaan "kak". Gunakan emoji secara wajar untuk menambah kehangatan dalam percakapan. JAWAB HANYA BERDASARKAN INFORMASI DARI "Conversation Context" DAN "Available Products Data". Jangan mengarang informasi.`,
    });
  }

  /**
   * Initializes LangChain tools for order processing and product inquiries
   * @returns {Promise<void>}
   * @private
   */
  async initializeTools() {
    try {
      // Pastikan chat model sudah diinisialisasi
      if (!this.chatModel) {
        await this.initializeChatModel();
      }
      
      // Initialize tools with the chat model
      this.orderProcessingTool = new OrderProcessingTool({ chatModel: this.chatModel });
      this.productInquiryTool = new ProductInquiryTool({ chatModel: this.chatModel });
      
      // Initialize agent executor with tools
      this.agentExecutor = await initializeAgentExecutorWithOptions(
        [this.orderProcessingTool, this.productInquiryTool],
        this.chatModel,
        {
          agentType: "structured-chat-zero-shot-react-description",
          verbose: process.env.NODE_ENV === 'development',
          handleParsingErrors: true,
          maxIterations: 3,
        }
      );
      
      logger.info("LangChain tools berhasil diinisialisasi");
    } catch (error) {
      logger.error(`Error inisialisasi LangChain tools: ${error.message}`);
      throw error;
    }
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

    this.tagTemplates.greeting = `Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan ramah dari {name_store}. Jawablah seperti manusia, bukan robot.

   ATURAN UTAMA:
   - Sapa pelanggan dengan hangat dan profesional
   - Perkenalkan diri sebagai CustoAI dan tawarkan bantuan pastikan juga pertannya nyambung dan fokus
   - Gunakan emoji yang sesuai untuk membuat percakapan lebih hidup tetapi jangan terlalu sering gunakan sebutuh nya saja
   - Jangan langsung memberikan daftar harga kecuali diminta
   - Fokus pada membangun rapport dan menanyakan kebutuhan pelanggan
   - Tetap lah fokus ke pertanyaan nya dan jangan melenceng atau ngawur
   - Memberikan respon yang kreatif tapi fokus ke pertanyaan juga

   context: 
   {context}

   Data Produk: 
   {products}
  `; // Template untuk tag price_inquiry

    this.tagTemplates.price_inquiry = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.

   ATURAN UTAMA:
   - WAJIB gunakan data produk yang akurat dari daftar yang diberikan
   - Jika diminta pricelist lengkap, tampilkan semua produk dengan format yang rapi dengan emoji yang akan menampilkan stock,status produk,harga produk, deskripsi produk dan ada kata-kata penutup juga.
   - Untuk pertanyaan harga spesifik, berikan detail lengkap produk tersebut
   - SELALU cek stok sebelum memberikan informasi
   - Jangan pernah mengarang atau mengubah harga, nama, atau stok produk harus sesuai dengan data produk yang diberikan.

   FORMAT RESPONS PRICELIST: "Berikan format response pricelist yang kreatif dengan emoji yang akan menampilkan stock,status produk,harga produk, deskripsi produk dan ada kata-kata penutup juga."

   RESPONS UNTUK PRODUK KOSONG:
   "Maaf kak, [nama produk] sedang kosong 😔 Mau cek produk lain yang tersedia?" (JIKA PRODUK KOSONG!)

   RESPONS UNTUK PRODUK TIDAK ADA:
   "Maaf kak, produk tersebut tidak tersedia di toko kami 🙏 Ini produk yang kami punya: [sebutkan alternatif]" (JIKA PRODUK TIDAK ADA!)

   SELALU AKHIRI DENGAN KATA-KATA YANG KREATIF TAPI HARUS FOKUS JUGA KE PERTANYAAN

   context:
   {context}

   Data Produk:
    {products}
  `; // Template untuk tag availability

    this.tagTemplates.availability = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.

   - Jika pelanggan menanyakan ketersediaan produk:
    * Jika produk tidak ada di daftar: Berikan kata-kata yang memberitahu jika produk yang di cari tidak ada!
    * Jika produk ada tapi stok 0: Berikan kata-kata seperti customer service yang lainnya jika stock produk kosong
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

  `; // Template untuk menghandle customer yang ingin order

    this.tagTemplates.order = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service yang menangani pesanan customer.

   ATURAN UTAMA:
   - Pelanggan sedang mengorder produk
   - Kamu akan memantau pesanan customer
   - Kamu akan memberikan tanggapan jika ada pesanan yang tidak sesuai
   - Jika ada product yang di beli lalu product tersebut tidak tersedia, maka kamu akan memberikan tanggapan ke customer jika produk yang di cari tidak tersedia
   - Jika jumlah produk yang di order mines maka berikan kata-kata jika ingin mengorder jumlah produk tidak boleh minus/mines

   FORMAT RESPONS: "Berikan format respons yang jelas menampilkan format produk yang akan dikirimkan,total pembayaran, kata-kata ingin di lanjutkan atau tidak"

   context: 
   {context}

   Data Produk: 
    {products}
  `; // Template untuk mengkonfirmasi pesanan

    this.tagTemplates.order_confirmation = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service yang menangani konfirmasi pesanan.

   ATURAN UTAMA:
   - Pelanggan sedang mengkonfirmasi pesanan yang sudah dibuat
   - Berikan instruksi pembayaran yang jelas
   - Sebutkan total yang harus dibayar
   - Berikan informasi kontak admin untuk pembayaran

   FORMAT RESPONS: "Berikan format respons yang jelas menampilkan format produk yang akan dikirimkan,total pembayaran, kata-kata ingin di lanjutkan atau tidak"

   context: 
   {context}

   Data Produk: 
    {products}
  `; // Template untuk tag unknown (default)

    this.tagTemplates.unknown = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.

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
  `; // Template untuk tag technical_details

    this.tagTemplates.technical_details = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
   - Jika pelanggan menanyakan detail teknis produk (misal: bisa di HP, smart TV, jumlah user, kualitas, dsb), jawab hanya berdasarkan data produk yang tersedia.
   - Jika data teknis tidak ada di daftar produk, jawab: "Maaf kak, informasi teknis tersebut tidak tersedia untuk produk ini 🙏"
   - Jangan pernah mengarang fitur, spesifikasi, atau keunggulan yang tidak ada di data produk.
   - Pilih kata yang sopan, jelas, dan mudah dipahami.
   - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

   context:
   {context}

   Data Produk:
   {products}

  `; // Template untuk tag payment_method

    this.tagTemplates.payment_method = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
   - Jika pelanggan menanyakan metode pembayaran, jawab hanya QRIS sebagai satu-satunya metode pembayaran yang tersedia.
   - Jika pelanggan bertanya tentang metode lain, jawab: "Maaf kak, saat ini pembayaran hanya bisa melalui QRIS 🙏"
   - Jangan pernah mengarang atau menambah metode pembayaran lain.
   - Pilih kata yang sopan, jelas, dan mudah dipahami.
   - Tambahkan emoji yang sesuai agar percakapan lebih hidup.
   - Jangan pernah mengirimkan informasi mengenai metode pembayaran lain selain QRIS.


   context:
   {context}

   Data Produk:
   {products}
  `; // Template untuk tag refund_policy

    this.tagTemplates.refund_policy = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
   - Jika pelanggan menanyakan kebijakan refund/garansi, jawab hanya sesuai kebijakan yang berlaku di toko dan data yang tersedia.
   - Jika tidak ada kebijakan refund untuk produk tersebut, jawab: "Maaf kak, untuk produk ini belum ada kebijakan refund khusus 🙏"
   - Jangan pernah mengarang atau menjanjikan refund/garansi di luar kebijakan yang ada.
   - Pilih kata yang sopan, jelas, dan mudah dipahami.
   - Tambahkan emoji yang sesuai agar percakapan lebih hidup.


   context:
   {context}

   Data Produk:
   {products}

  `; // Template untuk tag products_details

    this.tagTemplates.products_details = `

   Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
   - Jika pelanggan bertanya tentang detail produk, jawab berdasarkan data produk yang tersedia di database.
   - Berikan informasi seperti: fitur, manfaat, masa aktif, metode pengiriman, dan hal penting lain jika tersedia.
   - Jangan memberikan informasi yang tidak ada dalam data produk.
   - Jika data produk terbatas, jawab sejujur mungkin dan ajak pelanggan untuk bertanya lebih lanjut jika perlu.
   - Gunakan bahasa yang sopan, ramah, dan mudah dipahami.
   - Tambahkan emoji yang relevan agar lebih bersahabat.
   - Jangan pernah mengarang atau menambah layanan yang tidak ada.
   - Pilih kata yang sopan, jelas, dan mudah dipahami.
   - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

   context:
   {context}

   Data Produk:
   {products}

  `; // Template untuk tag emerging_services

    this.tagTemplates.emerging_services = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
   - Jika pelanggan menanyakan layanan baru, AI tools, atau fitur digital lain, jawab hanya berdasarkan layanan yang benar-benar tersedia di daftar produk/layanan.
   - Jika layanan tidak tersedia, jawab: "Maaf kak, layanan tersebut belum tersedia di toko kami 🙏"
   - Jangan pernah mengarang atau menambah layanan yang tidak ada.
   - Pilih kata yang sopan, jelas, dan mudah dipahami.
   - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

   context:
   {context}


   Data Produk:
   {products}

  `; // Template untuk tag referral_loyalty

    this.tagTemplates.referral_loyalty = `
   Kamu adalah CustoAI dibuat oleh custoai.id, customer service profesional dan informatif. Jawablah seperti manusia, bukan robot.
   - Jika pelanggan menanyakan program referral, loyalti, atau bonus, jawab hanya sesuai program yang benar-benar berlaku di toko.
   - Jika tidak ada program referral/loyalti, jawab: "Maaf kak, saat ini belum ada program referral atau loyalti di toko kami 🙏"
   - Jangan pernah mengarang atau menjanjikan bonus/loyalti di luar program yang ada.
   - Pilih kata yang sopan, jelas, dan mudah dipahami.
   - Tambahkan emoji yang sesuai agar percakapan lebih hidup.

    context:
    {context}


    Data Produk:
    {products}`;
  }

  // Fungsi calculateSimilarity dihapus karena tidak digunakan dan digantikan oleh fuzzball

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
   * @param {Object} productData - Data product
   * @returns {Promise<Object>} Product clarification result
   * @private
   */
  async handleProductClarification(question, formattedHistory, productData) {
    try {
      logger.info(`Processing product clarification response: "${question}"`);

      // Find the ambiguous product query in chat history
      let ambiguousProductInfo = null;
      let selectedVariant = null;

      // Look for the ambiguous product query in recent messages
      for (
        let i = formattedHistory.length - 5;
        i < formattedHistory.length;
        i++
      ) {
        if (i < 0) continue;

        const msg = formattedHistory[i];
        if (
          msg.type === "ai" &&
          msg.content &&
          msg.content.includes("⚠️ **Mohon Klarifikasi**") &&
          msg.content.includes("Saya menemukan beberapa varian produk")
        ) {
          // Extract product variants from the message
          for (const [name, data] of Object.entries(productData)) {
            const normalizedName = name.toLowerCase();
            const normalizedQuestion = question.toLowerCase();
            const normalizedCode = data.code.toLowerCase();

            if (
              normalizedQuestion.includes(normalizedName) ||
              normalizedQuestion.includes(normalizedCode)
            ) {
              selectedVariant = {
                name,
                code: data.code,
                price: data.price,
                quantity: 1, // Default quantity
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
        const quantityMatches = question.match(
          /\b(\d+)\s*(buah|pcs|unit|akun)?\b/i
        );
        if (quantityMatches && quantityMatches[1]) {
          selectedVariant.quantity = parseInt(quantityMatches[1]);
        }

        // Calculate price
        const price =
          parseInt(selectedVariant.price.replace(/[^0-9]/g, "")) || 0;
        const subtotal = price * selectedVariant.quantity;

        const response =
          `✅ **Pesanan Dikonfirmasi**\n\n` +
          `• ${selectedVariant.name} (${selectedVariant.code})\n` +
          `  Harga: ${selectedVariant.price} x ${
            selectedVariant.quantity
          } = Rp ${subtotal.toLocaleString()}\n\n` +
          `💰 **Total: Rp ${subtotal.toLocaleString()}**\n\n` +
          `✅ Untuk melanjutkan pemesanan, silakan konfirmasi dengan mengetik "konfirmasi" atau hubungi admin untuk pembayaran.`;

        return {
          response,
          processingTime: 0,
          tags: "product_clarification",
          productInfo: {
            orders: [selectedVariant],
            total: subtotal,
          },
        };
      } else {
        logger.warn(
          `Could not identify selected product variant in user response`
        );

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

  async processWithAI(name_store, question, context, nomorWhatsapp, productData) {
    try {
      // Input validation
      if (
        !question ||
        typeof question !== "string" ||
        question.trim().length === 0
      ) {
        throw new Error("Question is required and must be a non-empty string");
      }

      if (!nomorWhatsapp || typeof nomorWhatsapp !== "string") {
        throw new Error("WhatsApp number is required and must be a string");
      }

      // Pastikan service sudah diinisialisasi
      if (!this.initialized) {
        await this.init();
      } else {
        // Pastikan chatModel tersedia
        if (!this.chatModel) {
          await this.initializeChatModel();
        }
        
        // Pastikan tools tersedia
        if (!this.productInquiryTool || !this.orderProcessingTool) {
          await this.initializeTools();
        }
      }
      
      logger.info(
        `Processing question from user ${nomorWhatsapp}: "${question.substring(
          0,
          100
        )}${question.length > 100 ? "..." : ""}"`
      );

      // Get chat history and format it
      const formattedHistory = await this.getChatHistory(nomorWhatsapp);
      logger.info(
        `Retrieved ${formattedHistory.length} messages from chat history`
      );

      // Get relevant information and product data
      const relevantInfo = this.formatRelevantInfo(
        context?.relevantEntries || []
      );
      const products = productData;
      const productString = this.formatProductData(products);

      // Determine question type/tag
      const tag = await this.getQuestionTag(
        question,
        formattedHistory,
        relevantInfo
      );
      logger.info(`Question classified as tag: ${tag}`);

      // Handle special cases for order tags
      if (tag === "order") {
        logger.info("Processing order request");
        const result = await this.handleOrderTag(question, formattedHistory, productData);
        logger.info(
          `Order processing completed with ${
            result.productInfo?.orders?.length || 0
          } items`
        );
        return result;
      }

      if (tag === "order_confirmation") {
        logger.info("Processing order confirmation");
        const result = await this.handleOrderConfirmation(formattedHistory);
        logger.info(
          `Order confirmation processed: ${
            result.productInfo?.confirmed ? "confirmed" : "not found"
          }`
        );
        return result;
      }

      if (tag === "product_clarification") {
        logger.info("Processing product clarification");
        const result = await this.handleProductClarification(
          question,
          formattedHistory,
          productData
        );
        logger.info(
          `Product clarification processed for: ${
            result.productInfo?.orders?.[0]?.name || "unknown product"
          }`
        );
        return result;
      }
      
      // Handle product inquiries using ProductInquiryTool
      if (tag === "price_inquiry") {
        logger.info("Processing price inquiry");
        const result = await this.handleProductInquiry(question, 'price', products);
        logger.info(`Price inquiry processed`);
        return result;
      }
      
      if (tag === "products_details" || tag === "availability" || tag === "technical_details") {
        const inquiryType = tag === "availability" ? 'availability' : 
                           tag === "technical_details" ? 'features' : 'general';
        logger.info(`Processing ${tag} inquiry`);
        const result = await this.handleProductInquiry(question, inquiryType, products);
        logger.info(`${tag} inquiry processed`);
        return result;
      }
1
      // Hanya gunakan agent executor untuk pertanyaan terkait order/pemesanan
      if (tag === "order" || tag === "order_confirmation" || 
          (tag === "unknown" && (
            question.toLowerCase().includes("pesan") || 
            question.toLowerCase().includes("order") || 
            question.toLowerCase().includes("beli") ||
            question.toLowerCase().includes("mau") ||
            question.toLowerCase().includes("ambil") ||
            question.toLowerCase().includes("jual") ||
            question.toLowerCase().includes("checkout") ||
            /\b\d+\s*(buah|pcs|unit|akun)?\b/i.test(question) || // Deteksi angka yang mungkin menunjukkan jumlah pesanan
            this.containsProductName(question) // Deteksi nama produk dalam pertanyaan
          ))) {
        logger.info("Attempting to process with agent executor for order-related query");
        try {
          const agentResult = await this.processWithAgent(question);
          
          // If agent produced a meaningful response, use it
          if (agentResult.response && !agentResult.response.includes("❌")) {
            logger.info(`Successfully processed with agent executor, identified as ${agentResult.tags} tag`);
            
            // Update chat history
            await this.updateChatHistory(nomorWhatsapp, question, agentResult.response);
            
            return agentResult;
          } else {
            logger.info("Agent executor failed to process order, falling back to standard response generation");
          }
        } catch (agentError) {
          logger.error(`Error using agent executor: ${agentError.message}, falling back to standard response generation`);
        }
      }
      
      // Generate AI response using standard method as fallback
      const startTime = Date.now();
      const response = await this.generateResponse(
        name_store,
        question,
        tag,
        formattedHistory,
        relevantInfo,
        productString
      );
      const processingTime = Date.now() - startTime;

      logger.info(`AI response generated in ${processingTime / 1000}ms`);

      // Update chat history
      await this.updateChatHistory(nomorWhatsapp, question, response.content);

      return {
        response: response.content,
        processingTime: response.processingTime || processingTime,
        tags: tag,
      };
    } catch (error) {
      logger.error(
        `Error processing with AI for user ${nomorWhatsapp}: ${error.message}`
      );
      logger.error(`Stack trace: ${error.stack}`);

      // Return a user-friendly error response instead of throwing
      return {
        response:
          "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi atau hubungi admin untuk bantuan.",
        processingTime: 0,
        tags: "error",
        productInfo: null,
        error: error.message,
      };
    }
  }

  // Fungsi ensureInitialized dihapus karena sudah diimplementasikan langsung di processWithAI

  /**
   * Retrieves and formats chat history for a customer
   * @param {string} nomorWhatsapp - Customer's WhatsApp number
   * @returns {Promise<Array>} Formatted chat history
   * @private
   */
  async getChatHistory(nomorWhatsapp) {
    const historyKey = `lel:${nomorWhatsapp}`;
    let history = await redisClient.lRange(
      historyKey,
      -CONFIG.HISTORY_CONTEXT_SIZE,
      -1
    );
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
      if (word.length > 3) {
        // Only consider meaningful words
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
            desc: data.desc,
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
        variants: productVariants,
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
    if (
      !ambiguousProduct ||
      !ambiguousProduct.variants ||
      ambiguousProduct.variants.length <= 1
    ) {
      return null;
    }

    const variantsList = ambiguousProduct.variants
      .map(
        (variant) =>
          `• **${variant.name}** (${variant.code}): ${variant.price}\n  ${
            variant.desc || ""
          }`
      )
      .join("\n\n");

    const baseProduct = ambiguousProduct.baseTerms.join(" ");

    return (
      `⚠️ **Mohon Klarifikasi**\n\n` +
      `Saya menemukan beberapa varian produk "${baseProduct}" yang tersedia. ` +
      `Mohon tentukan varian mana yang Anda inginkan:\n\n` +
      `${variantsList}\n\n` +
      `Silakan balas dengan nama lengkap atau kode produk yang Anda inginkan.`
    );
  }

  /**
   * Handles order tag using OrderProcessingTool
   * @param {string} question - Customer's question
   * @param {Array} formattedHistory - Formatted chat history
   * @param {Object} productData - Product data
   * @returns {Promise<Object>} Response object
   */
  async handleOrderTag(question, formattedHistory, productData) {
    try {
      logger.info(
        `Processing order extraction with OrderProcessingTool for: "${question.substring(0, 100)}${question.length > 100 ? '...' : ''}"`
      );
      const startTime = Date.now();

      // Use OrderProcessingTool to process the order
      const orderResult = await this.orderProcessingTool.call(question);
      const processingTime = Date.now() - startTime;
      
      logger.info(`Order processing completed in ${processingTime}ms`);
      
      // If the tool returns a formatted order summary, use it directly
      if (orderResult && !orderResult.includes('❌')) {
        // Extract order information for productInfo
        const orderItems = [];
        let totalPrice = 0;
        
        // Parse the order information from the formatted response
        const orderLines = orderResult.split('\n');
        const orderItemRegex = /• (.+?) \((.+?)\)[\s\S]*?Harga: (.+?) x (\d+) = Rp ([\d,]+)/g;
        let match;
        
        const orderText = orderResult;
        while ((match = orderItemRegex.exec(orderText)) !== null) {
          const [, name, code, price, quantity, subtotal] = match;
          const priceValue = parseInt(price.replace(/[^0-9]/g, "")) || 0;
          const quantityValue = parseInt(quantity) || 0;
          const subtotalValue = parseInt(subtotal.replace(/[^0-9]/g, "")) || 0;
          
          orderItems.push({
            name,
            code,
            price,
            quantity: quantityValue,
          });
          
          totalPrice += subtotalValue;
        }
        
        return {
          response: orderResult,
          processingTime,
          tags: "order",
          productInfo: { orders: orderItems, total: totalPrice },
        };
      } else {
        // If there was an error or no valid orders found
        logger.info(`No valid orders found or processing failed`);
        
        return {
          response: orderResult || `❌ Tidak ditemukan pesanan dalam percakapan ini. Silakan sebutkan produk dan jumlah yang ingin dipesan.\n\n📝 Contoh: "Saya mau pesan Netflix 1P2U sebanyak 2 akun"`,
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
   * Handles product inquiry using ProductInquiryTool
   * @param {string} question - Customer's question
   * @param {string} inquiryType - Type of inquiry (price, availability, features, general)
   * @param {Object} productData - Product data (optional)
   * @returns {Promise<Object>} Response object
   */
  async handleProductInquiry(question, inquiryType = 'general', productData = null) {
    try {
      logger.info(
        `Processing product inquiry with ProductInquiryTool: "${question.substring(0, 100)}${question.length > 100 ? '...' : ''}" [${inquiryType}]`
      );
      const startTime = Date.now();

      // Pastikan chat model sudah diinisialisasi
      if (!this.chatModel) {
        await this.initializeChatModel();
      }
      
      // Buat instance baru dari ProductInquiryTool untuk setiap permintaan
      // untuk memastikan kita menggunakan model chat yang benar dan data produk terbaru
      console.log(productData ? Object.values(productData) : Object.values(this.getProductData() || {}))
      const validatedProductData = Array.isArray(productData) 
        ? productData 
        : Object.values(productData || this.getProductData() || {});

      if (!Array.isArray(validatedProductData)) {
        logger.warn('Invalid productData format - forcing to array');
        validatedProductData = [];
      }

      const productInquiryTool = new ProductInquiryTool({
        chatModel: this.chatModel,
        productData: validatedProductData
      });
      
      // Periksa apakah productInquiryTool sudah diinisialisasi dengan benar
      if (!productInquiryTool.chatModel) {
        throw new Error('Chat model tidak tersedia di ProductInquiryTool');
      }

      // Prepare input with inquiry type hint
      const input = `${question} [${inquiryType}]`;
      
      // Use ProductInquiryTool to process the inquiry
      const inquiryResult = await productInquiryTool.call(input);
      const processingTime = Date.now() - startTime;
      
      logger.info(`Product inquiry processing completed in ${processingTime}ms`);
      
      return {
        response: inquiryResult || "Maaf, saya tidak dapat memproses pertanyaan Anda saat ini.",
        processingTime,
        tags: inquiryType === 'price' ? 'price_inquiry' : 'products_details',
        productInfo: null,
      };
    } catch (error) {
      logger.error(`Error in handleProductInquiry: ${error.message}`);
      return {
        response: `❌ Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi atau hubungi admin untuk bantuan.`,
        processingTime: 0,
        tags: inquiryType === 'price' ? 'price_inquiry' : 'products_details',
        productInfo: null,
      };
    }
  }
  
  /**
   * Processes a complex query using the agent executor with both tools
   * @param {string} question - Customer's question
   * @returns {Promise<Object>} Response object
   */
  async processWithAgent(question) {
    try {
      logger.info(
        `Processing order-related query with agent executor for: "${question.substring(0, 100)}${question.length > 100 ? '...' : ''}"`
      );
      const startTime = Date.now();

      // Pastikan chat model tersedia
      if (!this.chatModel) {
        await this.initializeChatModel();
      }
      
      // Pastikan tools tersedia
      if (!this.productInquiryTool || !this.orderProcessingTool) {
        await this.initializeTools();
      }
      
      // Pastikan agent executor tersedia
      if (!this.agentExecutor) {
        this.agentExecutor = await initializeAgentExecutorWithOptions(
          [this.orderProcessingTool, this.productInquiryTool],
          this.chatModel,
          {
            agentType: "structured-chat-zero-shot-react-description",
            verbose: process.env.NODE_ENV === 'development',
            handleParsingErrors: true,
            maxIterations: 3,
          }
        );
      }

      // Use agent executor to determine the best tool and process the query
      const agentResult = await this.agentExecutor.invoke({
        input: question,
      });
      
      const processingTime = Date.now() - startTime;
      logger.info(`Agent processing completed in ${processingTime}ms`);
      
      // Hanya fokus pada tag order, abaikan tag lainnya
      let tag = "order";
      
      // Jika tidak menggunakan order_processing_tool, tetap kembalikan sebagai order
      // untuk memastikan agent hanya menangani pertanyaan terkait pemesanan
      if (!agentResult.log.includes("order_processing_tool")) {
        logger.info("Agent tidak menggunakan order_processing_tool, tetapi tetap dikategorikan sebagai order");
      }
      
      return {
        response: agentResult.output,
        processingTime,
        tags: tag,
        productInfo: null,
      };
    } catch (error) {
      logger.error(`Error in processWithAgent: ${error.message}`);
      return {
        response: `❌ Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi atau hubungi admin untuk bantuan.`,
        processingTime: 0,
        tags: "unknown",
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
      if (
        msg.content &&
        msg.content.includes("📋") &&
        msg.content.includes("Total:")
      ) {
        recentOrder = msg.content;
        break;
      }
    }

    if (recentOrder) {
      // Extract total from the order
      const totalMatch = recentOrder.match(/Total: (Rp [\d.,]+)/);
      const total = totalMatch ? totalMatch[1] : "Rp 0";

      const response =
        `✅ **Pesanan Dikonfirmasi!**\n\n` +
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
        response:
          `❓ Maaf, saya tidak menemukan pesanan yang perlu dikonfirmasi.\n\n` +
          `Silakan buat pesanan terlebih dahulu dengan menyebutkan produk dan jumlah yang diinginkan.\n\n` +
          `📝 Contoh: "Saya mau pesan Netflix 1P2U sebanyak 2 akun"`,
        processingTime: 0,
        tags: "order_confirmation",
        productInfo: null,
      };
    }
  }

  /**
   * Formats product data for display
   * @param {Object} products - Product data
   * @returns {string} Formatted product string
   * @private
   */
  formatProductData(products) {
    return Object.entries(products)
      .map(([index, p]) => {
        const stockStatus = parseInt(p.stock) > 0 ? "Tersedia" : "Kosong";
        return `${index}. ${p.name}: Harga ${p.price}, Status: ${stockStatus}, Stock: ${p.stock}`;
      })
      .join("\n");
  }

  /**
   * Checks if the question contains any product name
   * @param {string} question - Customer's question
   * @returns {boolean} True if question contains a product name
   * @private
   */
  containsProductName(question) {
    const productEntities = [
      'netflix', 'spotify', 'youtube', 'disney', 'canva', 'vidio',
      'amazon', 'hbo', 'game pass', 'chatgpt', 'loklok', 'prime',
      'viu', 'wetv', 'iqiyi', 'mola tv', 'apple music', 'deezer',
      'tidal', 'crunchyroll', 'hulu', 'paramount', 'peacock'
    ];
    
    const lowerText = question.toLowerCase();
    return productEntities.some(entity => lowerText.includes(entity));
  }

  /**
   * Generates an AI response based on question, tag, and context
   * @param {string} name_store - Name store
   * @param {string} question - Customer's question
   * @param {string} tag - Question classification tag
   * @param {Array} formattedHistory - Formatted chat history
   * @param {string} relevantInfo - Relevant context information
   * @param {string} productString - Formatted product data
   * @returns {Promise<Object>} Response with content and processing time
   * @private
   */
  async generateResponse(
    name_store,
    question,
    tag,
    formattedHistory,
    relevantInfo,
    productString
  ) {
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
      name_store,
      products: productString,
      chat_history: formattedHistory,
      query: question,
    });

    const startTime = Date.now();
    const response = await this.chatModel.invoke(formattedMessages);
    const processingTime = Date.now() - startTime;

    const cleanOutput = response.content
      .replace(/<think>.*?<\/think>/gs, "");

    logger.info(
      `Pertanyaan '${question}' diproses dalam ${processingTime / 1000}S dengan jawaban '${cleanOutput}'`
    );

    return {
      content: cleanOutput,
      processingTime,
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
    const historyKey = `data:${nomorWhatsapp}`;

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
      await redisClient.lTrim(
        historyKey,
        historyLength - CONFIG.MAX_HISTORY_LENGTH,
        -1
      );
    }
  }

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
      .map((entry) => {
        return `${entry.question ? `Pertanyaan: "${entry.question}"\n` : ""}
   Konteks: ${JSON.stringify(entry.answer, null, 2)}`;
      })
      .join("\n\n");
  }

  /**
   * Classifies a given question based on chat history and a predefined list of valid tags
   * Optimized for 100% accuracy on all tags
   * @param {string} question - The question to classify
   * @param {Array} chatHistory - The chat history for context
   * @param {Object} context - Context
   * @returns {Promise<string>} - The classified tag
   */
  async getQuestionTag(question, chatHistory, context) {
    try {
      if (!this.initialized) {
        await this.init();
      } else {
        // Pastikan chatModel tersedia
        if (!this.chatModel) {
          await this.initializeChatModel();
        }
        
        // Pastikan tools tersedia
        if (!this.productInquiryTool || !this.orderProcessingTool) {
          await this.initializeTools();
        }
      }

      const validTags = this.getValidQuestionTags();
      const systemInstruction = this.createTagClassificationPrompt(context);

      // Enhanced messages with more context for better accuracy
      const messages = [
        { role: "system", content: systemInstruction },
        ...chatHistory,
        {
          role: "user",
          content: `Tentukan tag paling relevan untuk pertanyaan ini: "${question}". Analisis dengan sangat teliti dan berikan jawaban dengan tingkat keyakinan 100%.`,
        },
      ];

      // Get AI response with zero temperature for deterministic output
      const originalModel = this.chatModel.model;
      const originalTemperature = this.chatModel.temperature;
      const originalTopP = this.chatModel.topP;
      try {
        this.chatModel.model = process.env.OLLAMA_MODEL_PREDICTION;
        this.chatModel.temperature = 0;
        this.chatModel.topP = 1;

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
                maximum: 1,
              },
            },
            required: ["tag", "confidence"],
          },
        });

        const responseParse = JSON.parse(response.content);

        logger.info(`Tag: ${responseParse.tag}, Confidence: ${responseParse.confidence}`);
        return responseParse.tag;
      } finally {
        // Restore original parameters
        this.chatModel.model = originalModel;
        this.chatModel.temperature = originalTemperature;
        this.chatModel.topP = originalTopP;
      }
    } catch (error) {
      logger.error(`Gagal mendapatkan tag pertanyaan: ${error.message}`);
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
   * @returns {string} System prompt for tag classification
   * @param context - Context
   * @private
   */
  createTagClassificationPrompt(context) {
    return `
PERAN ANDA:
Anda adalah sistem klasifikasi otomatis yang bertugas mengidentifikasi **satu** tag yang paling relevan untuk **pertanyaan terakhir** pelanggan, berdasarkan **konteks keseluruhan percakapan** sebelumnya. Anda HARUS memberikan klasifikasi dengan akurasi 100% dan tidak boleh salah Baca context agar akurasi nya benar.

PRIORITAS UTAMA:
Jika pertanyaan mengandung kata-kata terkait pemesanan seperti "pesan", "beli", "order", "checkout", "mau", "ambil", "jual", atau menyebutkan nama produk (seperti Netflix, Spotify, Disney, YouTube, dll) dengan atau tanpa jumlah, PRIORITASKAN tag "order" di atas tag lainnya. Jika pertanyaan menyebutkan nama produk dan mengandung angka, hampir pasti itu adalah pesanan.

DAFTAR TAG YANG TERSEDIA:
- **order**: Permintaan untuk memesan, membeli, checkout, atau konfirmasi pemesanan. Contoh: "Saya mau beli Netflix", "Pesan 2 akun Gmail", "Mau order Disney+", "Netflix 1", "Saya ambil Spotify 2 bulan", "Mau yang Netflix", "Jual Netflix ga?", "Netflix", "Spotify premium", "Disney+ berapa", "YouTube premium ada?", "Canva pro masih tersedia?", "Mau Netflix", "Berapa Netflix?", "Netflix ready?"
- **order_confirmation**: Konfirmasi atau persetujuan terhadap pesanan. Contoh: "Ya, saya jadi pesan", "Oke lanjut", "Setuju dengan pesanannya", "Jadi", "Lanjut", "Ok", "Siap", "Baik", "Benar", "Betul", "Ya", "Iya", "Gas", "Mantap", "Sip", "Oke", "Bisa", "Boleh", "Mau", "Jadi ambil", "Jadi beli"
- **price_inquiry**: Pertanyaan tentang harga, biaya, atau permintaan daftar harga (pricelist). Contoh: "Berapa harga Netflix?", "Minta pricelist", "Biaya untuk akun Gmail berapa?"
- **availability**: Pertanyaan tentang stok atau ketersediaan produk/layanan. Contoh: "Apakah masih ada stok?", "Netflix masih ready?", "Stoknya masih ada?"
- **greeting**: Sapaan seperti "halo", "selamat pagi", atau bentuk salam lainnya. Contoh: "Hai", "Selamat siang", "Halo admin"
- **technical_details**: Pertanyaan teknis terkait fitur, penggunaan, atau spesifikasi produk. Contoh: "Bagaimana cara menggunakan?", "Apa fitur Netflix Premium?", "Spesifikasi akun seperti apa?"
- **payment_method**: Pertanyaan tentang cara pembayaran atau metode transfer. Contoh: "Cara bayarnya gimana?", "Bisa bayar pakai QRIS?", "Transfer ke rekening mana?"
- **refund_policy**: Pertanyaan tentang pembatalan, pengembalian dana, atau refund. Contoh: "Kalau tidak bisa dipakai bisa refund?", "Kebijakan pengembalian dana?", "Bisa dibatalkan pesanannya?"
- **emerging_services**: Pertanyaan tentang layanan baru, fitur eksperimental, atau produk yang belum umum. Contoh: "Ada layanan baru apa?", "Fitur terbaru apa saja?", "Produk yang akan datang?"
- **products_details**: Pertanyaan tentang informasi atau rincian spesifik produk tertentu. Contoh: "Detail produk Netflix?", "Informasi lengkap tentang Disney+?", "Apa saja yang didapat dari akun premium?"
- **warranty_refund**: Pertanyaan tentang garansi, penukaran, atau pengembalian barang. Contoh: "Ada garansi berapa lama?", "Kalau tidak bisa login bisa ditukar?", "Jaminan akun berapa lama?"
- **referral_loyalty**: Pertanyaan tentang program referral, poin loyalitas, atau reward pelanggan. Contoh: "Ada program referral?", "Sistem poin member?", "Reward untuk pelanggan setia?"
- **unknown**: Jika pertanyaan tidak dapat diklasifikasikan ke kategori mana pun di atas.

PETUNJUK OUTPUT:
- Analisis pertanyaan dengan sangat teliti dan berikan tag yang PALING tepat.
- Kembalikan **hanya satu** tag yang paling relevan dengan tingkat keyakinan 100%.
- Berikan alasan yang jelas mengapa tag tersebut dipilih.
- Format output **WAJIB** dalam format JSON berikut:

\`\`\`json
{
  "tag": "<nama_tag>",
  "confidence": 1.0,
  "reasoning": "<alasan detail pemilihan tag>"
}
\`\`\`

Contoh:
\`\`\`json
{
  "tag": "order",
  "confidence": 1.0,
  "reasoning": "Pertanyaan mengandung kata 'beli' dan menyebutkan produk spesifik, yang jelas menunjukkan niat untuk melakukan pemesanan."
}
\`\`\`

Context :
${context}
`;
  }

  /**
   * Tests the connection to the Ollama service
   * @returns {Promise<Object>} Connection test result with status, model, latency and response snippet
   */
  async testConnection() {
    try {
      if (!this.initialized) {
        await this.init();
      } else {
        // Pastikan chatModel tersedia
        if (!this.chatModel) {
          await this.initializeChatModel();
        }
        
        // Pastikan tools tersedia
        if (!this.productInquiryTool || !this.orderProcessingTool) {
          await this.initializeTools();
        }
      }

      // Measure response time
      const startTime = Date.now();
      const response = await this.chatModel.invoke([
        {
          role: "user",
          content: "Respond with 'Connection successful' and nothing else.",
        },
      ]);
      const latency = Date.now() - startTime;

      // Format response for display
      const maxLength = 50;
      const snippet = response.content.substring(0, maxLength) + 
                     (response.content.length > maxLength ? "..." : "");

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

  // Fungsi findBestProductMatch dihapus karena sudah diintegrasikan langsung ke dalam checkProductAvailability

  /**
   * Extracts order lists from chat history
   * @param {String} question - Question from customer
   * @param {Array} chatHistory - Chat history
   * @param {Object} productData - Product data
   * @returns {Object} - Extracted order list or error message
   */
  async extractOrdersFromChat(question, chatHistory, productData) {
    try {
      if (!this.initialized) {
        await this.init();
      } else {
        // Pastikan chatModel tersedia
        if (!this.chatModel) {
          await this.initializeChatModel();
        }
        
        // Pastikan tools tersedia
        if (!this.productInquiryTool || !this.orderProcessingTool) {
          await this.initializeTools();
        }
      }

      // Validasi productData untuk mencegah error "Cannot convert undefined or null to object"
      if (!productData || typeof productData !== 'object') {
        logger.error(`Error: productData is ${productData === null ? 'null' : typeof productData}`);
        return {
          success: false,
          error: "Data produk tidak tersedia"
        };
      }

      // Membuat prompt untuk ekstraksi dan validasi pesanan
      const systemPrompt = `
PERAN ANDA:
Anda adalah AI canggih yang bertugas sebagai **Sistem Ekstraksi dan Validasi Pesanan Otomatis**. Tugas Anda adalah menganalisis chat, memvalidasi setiap item pesanan terhadap katalog, dan melaporkan hasilnya secara akurat.

PROSES BERPIKIR (LANGKAH-DEMI-LANGKAH):
Anda HARUS mengikuti proses ini untuk setiap permintaan:

1.  **Langkah 1: Identifikasi Niat Pesan**
    - Baca seluruh riwayat chat untuk menemukan semua item yang secara eksplisit ingin dibeli oleh pelanggan menggunakan kata kunci pemesanan.

2.  **Langkah 2: Validasi Setiap Item Satu per Satu**
    - Untuk setiap item yang diidentifikasi:
      a. **Cari di Katalog**: Coba temukan nama produk yang disebutkan pelanggan di dalam "KATALOG PRODUK YANG TERSEDIA".
      b. **Jika Produk TIDAK DITEMUKAN**: Catat item ini sebagai "productNotFound".
      c. **Jika Produk DITEMUKAN**: Lanjutkan ke pengecekan stok.
      d. **Cek Stok**: Lihat nilai \`stock\` pada data produk tersebut.
      e. **Jika Stok Habis (\`stock: 0\`)**: Catat item ini sebagai "outOfStock".
      f. **Jika Stok TERSEDIA (\`stock > 0\`)**: Catat item ini sebagai pesanan yang valid. Tentukan kuantitasnya (jika tidak disebut, WAJIB gunakan \`1\`).
      g. **Validasi Varian**:
      - Jika nama produk umum ditemukan (contoh: "Netflix") tetapi terdapat **beberapa varian** di katalog dengan nama dasar yang sama, maka pelanggan **WAJIB** menyebutkan varian (misal: 1P1U, 1P2U).
      - Jika tidak menyebutkan varian, maka catat item ini sebagai **"productNotFound"** dengan \`reason\`: **"Varian tidak disebutkan"**.
      h. **Validasi Kuantitas**:
      - Jika pelanggan ingin membeli produk terus **tidak menyebutkan jumlah order**, maka catat sebagai **"quantityNotFound"** dengan \`reason\`: **"Jumlah pesanan tidak di sebutkan"**
      - Jika pelanggan menyebutkan produk dan variannya tetapi **tidak menyebutkan jumlah**, maka catat sebagai **"productNotFound"** dengan \`reason\`: **"Jumlah pesanan tidak disebutkan"**.
      - Jika pelanggan membeli sebuah produk melebihi stock yang ada di daftar produk maka catat sebagai **"quantityOutOfStock"** dengan \`reason\`: **"Jumlah pesanan melebihi stok daftar produk"**

      3.  **Langkah 3: Tentukan Status Akhir**
    - Setelah semua item divalidasi, tentukan \`status\` output berdasarkan prioritas berikut:
      - Jika ada **satu atau lebih** item yang "productNotFound", maka status akhir adalah **\`productNotFound\`**.
      - Jika tidak ada yang "productNotFound", tetapi ada **satu atau lebih** item yang "outOfStock", maka status akhir adalah **\`outOfStock\`**.
      - Jika semua item yang dipesan valid (ditemukan dan stok tersedia), maka status akhir adalah **\`success\`**.
      - Jika tidak ada niat pemesanan sama sekali, status akhir adalah **\`noOrderFound\`**.

4.  **Langkah 4: Bentuk JSON Output Sesuai Format**
    - Buat JSON output sesuai dengan \`status\` akhir yang telah ditentukan. Sertakan pesanan yang valid di \`validOrders\` dan item bermasalah di \`problematicItems\`.

---

FORMAT OUTPUT:
Format output WAJIB berupa satu objek JSON tunggal dengan struktur di bawah ini. Isi dari \`status\`, \`validOrders\`, dan \`problematicItems\` bergantung pada hasil validasi.

**Struktur Dasar:**
\`\`\`json
{
  "status": "success" | "productNotFound" | "outOfStock" | "noOrderFound" | "quantityNotFound" | "quantityOutOfStock", "unknownProductVariant",
  "validOrders": [
    {
      "code": "kode_produk",
      "name": "nama_produk",
      "price": "harga_produk",
      "quantity": 1
    }
  ],
  "problematicItems": [
    {
      "userInput": "apa_yang_diketik_pelanggan",
      "reason": "Produk tidak ditemukan" | "Stok habis"
    }
  ]
}
\`\`\`

**Contoh 1: Output \`success\` (Semua pesanan valid)**
\`\`\`json
{
  "status": "success",
  "validOrders": [
    {
      "code": "GMAIL-FRESH",
      "name": "Akun Gmail Fresh",
      "price": "Rp 5.000",
      "quantity": 2
    }
  ],
  "problematicItems": []
}
\`\`\`

**Contoh 2: Output \`productNotFound\` (Ada produk yang tidak ada di katalog)**
\`\`\`json
{
  "status": "productNotFound",
  "validOrders": [
    {
      "code": "NFLX-PREMIUM",
      "name": "Netflix Premium 1 Bulan",
      "price": "Rp 150.000",
      "quantity": 1
    }
  ],
  "problematicItems": [
    {
      "userInput": "akun spotify",
      "reason": "Produk tidak ditemukan"
    }
  ]
}
\`\`\`

**Contoh 3: Output \`outOfStock\` (Produk ditemukan tapi stok habis)**
\`\`\`json
{
  "status": "outOfStock",
  "validOrders": [],
  "problematicItems": [
    {
      "userInput": "akun gmail aged",
      "reason": "Stok habis"
    }
  ]
}
\`\`\`

**Contoh 4: Output \`noOrderFound\` (Tidak ada niat membeli)**
\`\`\`json
{
  "status": "noOrderFound",
  "validOrders": [],
  "problematicItems": []
}
\`\`\`

**Contoh 4: Output \`quantityNotFound\` (Jumlah pesanan tidak di temukan)**
\`\`\`json
{
  "status": "quantityNotFound",
  "validOrders": [],
  "problematicItems": [
    {
      "userInput": "mau paket netflix 1p1u nya",
      "reason": "Jumlah pesanan tidak disebutkan"
    }
  ]
}
\`\`\`

**Contoh 5: Output \`quantityOutOfStock\` (Jumlah pesanan melebihi stok)**
\`\`\`json
{
  "status": "quantityOutOfStock",
  "validOrders": [
    {
      "code": "NFLX-PREMIUM",
      "name": "Netflix Premium 1 Bulan",
      "price": "Rp 150.000",
      "quantity": 2
    }
  ],
  "problematicItems": [
    {
      "userInput": "mau 2 paket netflix 1p1u nya",
      "reason": "Jumlah pesanan melebihi stok"
    }
  ]
}
\`\`\`

**Contoh 6: Output \`unknowProductVariant\` (Varian produk tidak disebutkan karena terdapat beberapa produk serupa)**
\`\`\`json
{
  "status": "unknowProductVariant",
  "validOrders": [],
  "problematicItems": [
    {
      "userInput": "mau 2 paket netflix nya",
      "reason": "Varian produk tidak disebutkan, karena terdapat beberapa produk serupa dengan nama yang mirip. Mohon sebutkan varian secara spesifik."
    }
  ]
}
\`\`\`

---

ATURAN PENTING & LARANGAN:
- **Kata Kunci Pemesanan**: Perhatikan kata-kata: "mau beli", 'saya pesan', "order", "pesen", "mau yg itu", "deal yg ini", "saya ambil", "jadi", "oke fix", "butuh", "mau dong".
- **Kuantitas Default**: Jika jumlah (kuantitas) pesanan tidak disebutkan dengan jelas, **WAJIB** gunakan angka **1**.
- **JANGAN BERASUMSI**: Jangan menebak. Jika pelanggan hanya bertanya ("harga netflix berapa?"), itu BUKAN pesanan.
- **DATA AKURAT**: Nama, kode, dan harga pada \`validOrders\` harus sama persis seperti di katalog.
- **Varian Wajib**: Jika sebuah produk memiliki beberapa varian di katalog, pelanggan WAJIB menyebutkan varian yang dimaksud (misalnya: 1P1U, 1P2U, dll). Jika tidak disebutkan, produk dianggap tidak lengkap.
- **Kuantitas Wajib untuk Varian**: Jika varian disebutkan tapi tidak ada kuantitas, itu dianggap belum lengkap dan tidak bisa diproses.

---

KATALOG PRODUK YANG TERSEDIA (SUMBER KEBENARAN TUNGGAL):
${Object.entries(productData)
  .map(
    ([_, data]) =>
      `- ${data.name} (Code: ${data.code}, Price: ${data.price}, Stock: ${data.stock})`
  )
  .join("\n")}

RIWAYAT CHAT UNTUK DIANALISIS:
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
          content: `${question}\n\nTolong bantu saya mendapatkan daftar pesanan saya baru-baru ini dari chat ini. Saya ingin tahu apa saja yang sudah saya pesan. Untuk produknya, berikan code produk, nama produk, harga, dan jumlahnya. Jika ada yang belum lengkap, tolong beri tahu saya.`,
        },
      ];

      const originalModel = this.chatModel.model;
      const originalTemperature = this.chatModel.temperature
      const originalTopP = this.chatModel.topP;
      try {
        this.chatModel.model = process.env.OLLAMA_MODEL_PREDICTION
        this.chatModel.temperature = 0;
        this.chatModel.topP = 1;

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
      console.log(response)
      return this.parseOrderResponse(response, productData);
      } finally {
        this.chatModel.model = originalModel;
        this.chatModel.temperature = originalTemperature;
        this.chatModel.topP = originalTopP;
      }
    } catch (error) {
      logger.error(`Error extracting orders: ${error}`);
      return {
        success: false,
        error: "Gagal mengekstrak pesanan",
      };
    }
  }

  /**
   * Parses and validates the order response from AI
   * @param {Object} response - AI response object
   * @param {Object} productData - Product data
   * @returns {Object} Parsed and validated orders or error
   * @private
   */
  parseOrderResponse(response, productData) {
    try {
      let dataOrders = [];
      logger.info(
        `Parsing order response: ${response.content.substring(0, 100)}...`
      );
      
      // Validasi productData untuk mencegah error "Cannot convert undefined or null to object"
      if (!productData || typeof productData !== 'object') {
        logger.error(`Error: productData is ${productData === null ? 'null' : typeof productData}`);
        return {
          success: false,
          error: "Data produk tidak tersedia"
        };
      }

      // Try to parse structured JSON response first
      try {
        const parsed =
          typeof response.content === "string"
            ? JSON.parse(response.content)
            : response.content;

        if (parsed && Array.isArray(parsed.orders)) {
          dataOrders = parsed.orders;
          logger.info(
            `Successfully parsed JSON orders: ${dataOrders.length} items found`
          );

          // Ensure prices are correctly formatted in JSON response
          dataOrders = dataOrders.map((order) => {
            // Get the correct price from product catalog if available
            const productMatch = this.checkProductAvailability(
              order.name,
              productData
            );

            if (
              productMatch &&
              productMatch.score >= CONFIG.FUZZY_MATCH_THRESHOLD
            ) {
              logger.info(
                `Using catalog price for ${order.name}: ${productMatch.product.price}`
              );
              return {
                ...order,
                price: productMatch.product.price,
                ambiguous: this.isAmbiguousProduct(order.name, productData),
              };
            }
            return order;
          });
        } else {
          logger.warn(
            `Response has invalid format, missing orders array: ${JSON.stringify(
              parsed
            )}`
          );
        }
      } catch (jsonError) {
        logger.warn(
          `Failed to parse JSON response: ${jsonError.message}. Falling back to text parsing.`
        );

        // Enhanced fallback parsing for various order patterns
        const lines = response.content
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line);

        logger.info(
          `Fallback parsing: Processing ${lines.length} lines of text`
        );

        for (const line of lines) {
          // Pattern 1: "1. Product - 2 units"
          let match = line.match(
            /\d+\.\s*([^-]+)\s*-\s*(\d+)\s*(?:units|unit|qty|quantity|pcs|pieces|item|items|buah|pcs|akun|akun)?/i
          );

          // Pattern 2: "Product (2 units)"
          if (!match) {
            match = line.match(
              /([^(]+)\s*\(\s*(\d+)\s*(?:units|unit|qty|quantity|pcs|pieces|item|items|buah|pcs|akun|akun)?\)/i
            );
          }

          // Pattern 3: "Product x2" or "Product 2x"
          if (!match) {
            match =
              line.match(/([^x]+)\s*x\s*(\d+)/i) ||
              line.match(/([^\d]+)\s*(\d+)\s*x/i);
          }

          // Pattern 4: "2 units of Product" or "2 buah Product" or "2 akun Product"
          if (!match) {
            const reverseMatch = line.match(
              /(\d+)\s*(?:units|unit|qty|quantity|pcs|pieces|item|items|buah|pcs|akun|akun)?\s*(?:of|dari)?\s*(.+)/i
            );
            if (reverseMatch) {
              match = [reverseMatch[0], reverseMatch[2], reverseMatch[1]];
            }
          }

          // Pattern 5: "Saya mau pesan/beli Product sebanyak 2"
          if (!match) {
            const indonesianMatch = line.match(
              /(?:pesan|beli|order|mau)\s+([^\d]+)\s+(?:sebanyak|sejumlah|dengan\s+jumlah|dengan\s+kuantitas)\s+(\d+)/i
            );
            if (indonesianMatch) {
              match = [
                indonesianMatch[0],
                indonesianMatch[1],
                indonesianMatch[2],
              ];
            }
          }

          // Pattern 6: "Product sebanyak 2"
          if (!match) {
            const simpleIndonesianMatch = line.match(
              /([^\d]+)\s+(?:sebanyak|sejumlah|dengan\s+jumlah|dengan\s+kuantitas)\s+(\d+)/i
            );
            if (simpleIndonesianMatch) {
              match = [
                simpleIndonesianMatch[0],
                simpleIndonesianMatch[1],
                simpleIndonesianMatch[2],
              ];
            }
          }

          // Pattern 7: "Mau beli/pesan 2 Product"
          if (!match) {
            const buyMatch = line.match(
              /(?:mau|ingin|akan)\s+(?:beli|pesan|order)\s+(\d+)\s+(.+)/i
            );
            if (buyMatch) {
              match = [buyMatch[0], buyMatch[2], buyMatch[1]];
            }
          }

          // Pattern 8: "Beli/pesan 2 Product"
          if (!match) {
            const directBuyMatch = line.match(
              /(?:beli|pesan|order)\s+(\d+)\s+(.+)/i
            );
            if (directBuyMatch) {
              match = [directBuyMatch[0], directBuyMatch[2], directBuyMatch[1]];
            }
          }

          // Pattern 9: "Product 2 buah/akun"
          if (!match) {
            const productFirstMatch = line.match(
              /([^\d]+)\s+(\d+)\s+(?:buah|akun|unit|pcs)/i
            );
            if (productFirstMatch) {
              match = [
                productFirstMatch[0],
                productFirstMatch[1],
                productFirstMatch[2],
              ];
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
                logger.info(
                  `CSV format detected: ${code}, ${name}, ${price}, ${quantity}`
                );
                dataOrders.push({ code, name, price, quantity });
                continue;
              }
            }
          }

          if (match) {
            const product = match[1].trim();
            const quantity = parseInt(match[2].trim());
            logger.info(
              `Matched product pattern: "${product}" with quantity ${quantity} (pattern match: "${match[0]}")`
            );

            // Extract price if available in the line
            let price = null;
            const priceMatch = line.match(/(?:Rp|IDR)\s*([\d.,]+)/i);
            if (priceMatch) {
              price = `Rp ${priceMatch[1]}`;
              logger.info(`Price found in text: ${price}`);
            }

            if (product && !isNaN(quantity) && quantity > 0) {
              // Try to find matching product using fuzzy matching
              const bestMatch = this.checkProductAvailability(product, productData);

              if (
                bestMatch &&
                bestMatch.score >= CONFIG.FUZZY_MATCH_THRESHOLD
              ) {
                logger.info(
                  `Product matched: "${product}" -> "${bestMatch.name}" (score: ${bestMatch.score})`
                );
                const orderItem = {
                  code: bestMatch.product.code,
                  name: bestMatch.name,
                  price: bestMatch.product.price, // Always use catalog price for consistency
                  quantity: quantity,
                };
                logger.info(`Adding order item: ${JSON.stringify(orderItem)}`);
                dataOrders.push(orderItem);
              } else {
                logger.warn(
                  `No product match found for "${product}" or match score too low`
                );
              }
            } else {
              logger.warn(
                `Invalid product or quantity: product="${product}", quantity=${quantity}, isNaN=${isNaN(
                  quantity
                )}`
              );
            }
          } else {
            logger.debug(
              `No pattern match for line: "${line.substring(0, 100)}"`
            );
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
          logger.info(
            `Consolidated quantities for ${order.name} (${order.code}): new quantity = ${existing.quantity}`
          );
        } else {
          productMap.set(key, { ...order });
        }
      }

      // Filter out invalid orders and ensure correct price format
      const filteredOrders = Array.from(productMap.values())
        .filter(
          (order) =>
            order.code && order.name && order.price && order.quantity > 0
        )
        .map((order) => {
          // Find the product in catalog to ensure correct price
          for (const [name, product] of Object.entries(productData)) {
            if (product.code === order.code) {
              return {
                ...order,
                price: product.price, // Always use the catalog price
              };
            }
          }
          return order;
        });

      logger.info(
        `Final order count after filtering: ${filteredOrders.length}`
      );

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
        error: "Format pesanan tidak valid",
      };
    }
  }
}

module.exports = OllamaService;