/**
 * OllamaTools Module
 * Provides LangChain tools for handling purchase and order inquiries
 * Integrates with Ollama models for natural language processing
 */

'use strict';

const { Tool } = require('langchain/tools');
const { ChatOllama } = require('@langchain/ollama');
const { PromptTemplate } = require('@langchain/core/prompts');
const logger = require('./logger.js');

/**
 * OrderProcessingTool class
 * A LangChain tool for processing customer orders and purchase inquiries
 */
class OrderProcessingTool extends Tool {
  constructor(options = {}) {
    super();
    this.name = 'order_processing_tool';
    this.description = 'Use this tool to process customer orders and purchase inquiries';
    this.productData = options.productData || {};
    this.chatModel = options.chatModel || null;
    this.returnDirect = true;
  }

  /**
   * Processes a customer order or purchase inquiry
   * @param {string} input - Customer's order or purchase inquiry
   * @returns {Promise<string>} - Processing result with order details or clarification request
   */
  async _call(input) {
    try {
      logger.info(`Processing order with OrderProcessingTool: "${input.substring(0, 100)}${input.length > 100 ? '...' : ''}"`);
      
      if (!this.chatModel) {
        throw new Error('Chat model not initialized');
      }

      // Extract order details from input
      const orderDetails = await this.extractOrderDetails(input);
      
      // Check if products exist and are in stock
      const validationResult = await this.validateOrder(orderDetails);
      
      if (!validationResult.valid) {
        return validationResult.message;
      }
      
      // Format order summary
      return this.formatOrderSummary(validationResult.orders);
    } catch (error) {
      logger.error(`Error in OrderProcessingTool: ${error.message}`);
      return `Maaf, terjadi kesalahan saat memproses pesanan Anda. Silakan coba lagi dengan menyebutkan produk dan jumlah yang ingin dipesan dengan jelas.`;
    }
  }

  /**
   * Extracts order details from customer input
   * @param {string} input - Customer's order or purchase inquiry
   * @returns {Promise<Array>} - Extracted order details
   * @private
   */
  async extractOrderDetails(input) {
    const prompt = new PromptTemplate({
      template: `
        Kamu adalah AI yang bertugas mengekstrak detail pesanan dari input pelanggan.
        
        Input pelanggan: {input}
        
        Ekstrak informasi berikut:
        1. Nama produk yang ingin dibeli
        2. Jumlah yang diinginkan (default 1 jika tidak disebutkan)
        3. Varian produk jika disebutkan
        
        Format output sebagai JSON array dengan struktur:
        [
          {
            "productName": "nama produk",
            "quantity": jumlah,
            "variant": "varian produk (opsional)"
          }
        ]
        
        Jika tidak ada pesanan yang terdeteksi, kembalikan array kosong [].
        Berikan HANYA output JSON tanpa penjelasan tambahan.
      `,
      inputVariables: ['input'],
    });
    
    const formattedPrompt = await prompt.format({ input });
    const response = await this.chatModel.invoke(formattedPrompt);
    
    try {
      return JSON.parse(response.content);
    } catch (error) {
      logger.error(`Failed to parse order details: ${error.message}`);
      return [];
    }
  }

  /**
   * Validates order against product catalog
   * @param {Array} orderDetails - Extracted order details
   * @returns {Promise<Object>} - Validation result
   * @private
   */
  async validateOrder(orderDetails) {
    if (!orderDetails || orderDetails.length === 0) {
      return {
        valid: false,
        message: 'Tidak ditemukan pesanan dalam permintaan Anda. Silakan sebutkan produk dan jumlah yang ingin dipesan dengan jelas.'
      };
    }
    
    const validOrders = [];
    const invalidItems = [];
    
    for (const item of orderDetails) {
      const { productName, quantity, variant } = item;
      const matchedProduct = this.findProductMatch(productName, variant);
      
      if (!matchedProduct) {
        invalidItems.push({
          item: productName + (variant ? ` ${variant}` : ''),
          reason: 'Produk tidak ditemukan'
        });
        continue;
      }
      
      if (parseInt(matchedProduct.stock) < quantity) {
        invalidItems.push({
          item: matchedProduct.name,
          reason: `Stok tidak mencukupi (tersedia: ${matchedProduct.stock})`
        });
        continue;
      }
      
      validOrders.push({
        code: matchedProduct.code,
        name: matchedProduct.name,
        price: matchedProduct.price,
        quantity: quantity
      });
    }
    
    if (invalidItems.length > 0) {
      const errorMessages = invalidItems.map(item => `- ${item.item}: ${item.reason}`).join('\n');
      return {
        valid: false,
        message: `Beberapa item tidak dapat diproses:\n${errorMessages}\n\nSilakan periksa kembali pesanan Anda.`
      };
    }
    
    return {
      valid: true,
      orders: validOrders
    };
  }

  /**
   * Finds matching product in catalog
   * @param {string} productName - Product name to find
   * @param {string} variant - Product variant (optional)
   * @returns {Object|null} - Matched product or null if not found
   * @private
   */
  findProductMatch(productName, variant) {
    if (!productName || !this.productData) return null;
    
    const normalizedName = productName.toLowerCase();
    let bestMatch = null;
    let bestScore = 0;
    
    for (const [name, product] of Object.entries(this.productData)) {
      const productNameLower = name.toLowerCase();
      let score = 0;
      
      // Check for exact match first
      if (productNameLower === normalizedName) {
        score = 100;
      } else {
        // Use fuzzy matching
        score = require('fuzzball').ratio(normalizedName, productNameLower);
      }
      
      // If variant is specified, check if it's in the product name
      if (variant && score > 70) {
        const normalizedVariant = variant.toLowerCase();
        if (!productNameLower.includes(normalizedVariant)) {
          score -= 30; // Penalize if variant doesn't match
        } else {
          score += 10; // Bonus for variant match
        }
      }
      
      if (score > bestScore) {
        bestScore = score;
        bestMatch = { ...product, name };
      }
    }
    
    return bestScore >= 70 ? bestMatch : null;
  }

  /**
   * Formats order summary for customer
   * @param {Array} orders - Valid orders
   * @returns {string} - Formatted order summary
   * @private
   */
  formatOrderSummary(orders) {
    let totalPrice = 0;
    const orderList = orders.map(order => {
      const price = parseInt(order.price.replace(/[^0-9]/g, '')) || 0;
      const subtotal = price * order.quantity;
      totalPrice += subtotal;
      
      return `• ${order.name} (${order.code})\n  Harga: ${order.price} x ${order.quantity} = Rp ${subtotal.toLocaleString()}`;
    }).join('\n\n');
    
    return `📋 **Daftar Pesanan Anda:**\n\n${orderList}\n\n💰 **Total: Rp ${totalPrice.toLocaleString()}**\n\n✅ Untuk melanjutkan pemesanan, silakan konfirmasi dengan mengetik "konfirmasi" atau hubungi admin untuk pembayaran.`;
  }
}

/**
 * ProductInquiryTool class
 * A LangChain tool for handling product inquiries and availability checks
 */
class ProductInquiryTool extends Tool {
  constructor(options = {}) {
    super();
    this.name = 'product_inquiry_tool';
    this.description = 'Use this tool to handle product inquiries and availability checks';
    this.productData = options.productData || {};
    this.chatModel = options.chatModel || null;
    this.returnDirect = true;
    
    // Log initialization status
    if (this.chatModel) {
      logger.info('ProductInquiryTool initialized with chat model');
    } else {
      logger.warn('ProductInquiryTool initialized without chat model');
    }
  }

  /**
   * Processes a product inquiry
   * @param {string} input - Customer's product inquiry
   * @returns {Promise<string>} - Processing result with product information
   */
  async _call(input) {
    try {
      // Validasi input
      if (!input || typeof input !== 'string') {
        logger.warn('Invalid input to ProductInquiryTool: Input is not a string');
        return 'Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi dengan pertanyaan yang lebih spesifik.';
      }
      
      logger.info(`Processing product inquiry with ProductInquiryTool: "${input.substring(0, 100)}${input.length > 100 ? '...' : ''}"`);
      
      // Pastikan chat model tersedia
      if (!this.chatModel) {
        logger.error('Chat model not initialized in ProductInquiryTool');
        return 'Maaf, layanan AI sedang tidak tersedia. Silakan coba lagi nanti.';
      }
      
      // Pastikan product data tersedia
      if (!this.productData || typeof this.productData !== 'object' || Object.keys(this.productData).length === 0) {
        logger.error('Product data not available in ProductInquiryTool');
        return 'Maaf, data produk tidak tersedia saat ini. Silakan coba lagi nanti.';
      }

      // Extract product inquiry details
      const inquiryDetails = await this.extractInquiryDetails(input);
      if (!inquiryDetails) {
        logger.error('Failed to extract inquiry details');
        return 'Maaf, saya tidak dapat memahami pertanyaan Anda. Silakan coba dengan pertanyaan yang lebih spesifik.';
      }
      
      // Find matching products
      const matchingProducts = await this.findMatchingProducts(inquiryDetails);
      if (!Array.isArray(matchingProducts)) {
        logger.error('findMatchingProducts did not return an array');
        return 'Maaf, terjadi kesalahan saat mencari produk. Silakan coba lagi nanti.';
      }
      
      // Format product information
      return await this.formatProductInformation(matchingProducts, inquiryDetails);
    } catch (error) {
      logger.error(`Error in ProductInquiryTool: ${error.message}`);
      logger.debug(`Stack trace: ${error.stack}`);
      return `Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi dengan pertanyaan yang lebih spesifik.`;
    }
  }

  /**
   * Extracts inquiry details from customer input
   * @param {string} input - Customer's product inquiry
   * @returns {Promise<Object>} - Extracted inquiry details
   * @private
   */
  async extractInquiryDetails(input) {
    try {
      // Gunakan string template sederhana untuk menghindari masalah dengan PromptTemplate
      const promptText = `
        Kamu adalah AI yang bertugas mengekstrak detail pertanyaan produk dari input pelanggan.
        
        Input pelanggan: ${input}
        
        Ekstrak informasi berikut:
        1. Nama produk yang ditanyakan
        2. Jenis informasi yang dicari (harga, ketersediaan, fitur, dll)
        3. Varian produk jika disebutkan
        
        Format output sebagai JSON dengan struktur:
        {
          "productName": "nama produk (kosong jika tidak spesifik)",
          "inquiryType": "jenis pertanyaan (price, availability, features, general)",
          "variant": "varian produk (opsional)"
        }
        
        Berikan HANYA output JSON tanpa penjelasan tambahan. Jangan sertakan backtick, tanda kutip tambahan, atau teks lain di luar JSON.
      `;
      
      const response = await this.chatModel.invoke(promptText);
      
      if (!response || !response.content) {
        logger.warn('Empty response from chat model in extractInquiryDetails');
        return { productName: '', inquiryType: 'general', variant: '' };
      }
      
      // Bersihkan respons untuk memastikan hanya JSON yang diproses
      let content = response.content.trim();
      
      // Hapus backtick jika ada
      if (content.startsWith('```json')) {
        content = content.substring(7);
      } else if (content.startsWith('```')) {
        content = content.substring(3);
      }
      
      if (content.endsWith('```')) {
        content = content.substring(0, content.length - 3);
      }
      
      content = content.trim();
      
      // Coba ekstrak JSON jika ada di dalam teks
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        content = jsonMatch[0];
      }
      
      try {
        return JSON.parse(content);
      } catch (error) {
        logger.error(`Failed to parse inquiry details: ${error.message}`);
        logger.debug(`Raw content: ${content}`);
        
        // Jika parsing gagal, coba ekstrak informasi dari teks
        let inquiryType = 'general';
        
        // Deteksi jenis pertanyaan dari input
        const inputLower = input.toLowerCase();
        if (inputLower.includes('harga') || inputLower.includes('berapa') || inputLower.includes('mahal')) {
          inquiryType = 'price';
        } else if (inputLower.includes('tersedia') || inputLower.includes('stok') || inputLower.includes('ada') || inputLower.includes('stock')) {
          inquiryType = 'availability';
        } else if (inputLower.includes('fitur') || inputLower.includes('spesifikasi') || inputLower.includes('spec')) {
          inquiryType = 'features';
        }
        
        return { productName: '', inquiryType, variant: '' };
      }
    } catch (error) {
      logger.error(`Error in extractInquiryDetails: ${error.message}`);
      return { productName: '', inquiryType: 'general', variant: '' };
    }
  }

  /**
   * Finds matching products based on inquiry
   * @param {Object} inquiryDetails - Extracted inquiry details
   * @returns {Promise<Array>} - Matching products
   * @private
   */
  async findMatchingProducts(inquiryDetails) {
    try {
      // Pastikan inquiryDetails valid
      if (!inquiryDetails) {
        logger.warn('Invalid inquiry details provided to findMatchingProducts');
        return [];
      }
      
      const { productName, variant } = inquiryDetails;
      const matches = [];
      
      // Pastikan productData ada dan valid
      if (!this.productData || typeof this.productData !== 'object' || Object.keys(this.productData).length === 0) {
        logger.warn('No product data available for product inquiry');
        return [];
      }
      
      if (!productName || productName.trim() === '') {
        // Return all products for general inquiries
        for (const [name, product] of Object.entries(this.productData)) {
          matches.push({ ...product, name });
        }
        return matches.slice(0, 5); // Limit to 5 products for general inquiries
      }
      
      const normalizedName = productName.toLowerCase();
      
      for (const [name, product] of Object.entries(this.productData)) {
        const productNameLower = name.toLowerCase();
        let score = 0;
        
        // Check for exact match first
        if (productNameLower === normalizedName) {
          score = 100;
        } else if (productNameLower.includes(normalizedName) || normalizedName.includes(productNameLower)) {
          score = 85;
        } else {
          // Use fuzzy matching
          score = require('fuzzball').ratio(normalizedName, productNameLower);
        }
        
        // If variant is specified, check if it's in the product name
        if (variant && score > 60) {
          const normalizedVariant = variant.toLowerCase();
          if (productNameLower.includes(normalizedVariant)) {
            score += 15; // Bonus for variant match
          }
        }
        
        if (score >= 60) {
          matches.push({
            ...product,
            name,
            score
          });
        }
      }
      
      // Sort by score descending
      return matches.sort((a, b) => b.score - a.score);
    } catch (error) {
      logger.error(`Error in findMatchingProducts: ${error.message}`);
      return [];
    }
  }

  /**
   * Formats product information based on inquiry type
   * @param {Array} products - Matching products
   * @param {Object} inquiryDetails - Inquiry details
   * @returns {string} - Formatted product information
   * @private
   */
  async formatProductInformation(products, inquiryDetails) {
    try {
      // Pastikan products adalah array
      if (!Array.isArray(products)) {
        logger.error('Products is not an array in formatProductInformation');
        return `Maaf, terjadi kesalahan saat memproses informasi produk. Silakan coba lagi nanti.`;
      }
      
      // Pastikan inquiryDetails valid
      if (!inquiryDetails || typeof inquiryDetails !== 'object') {
        logger.error('Invalid inquiry details in formatProductInformation');
        inquiryDetails = { inquiryType: 'general' };
      }
      
      const { inquiryType = 'general' } = inquiryDetails;
      
      if (products.length === 0) {
        return `Maaf, saya tidak menemukan produk yang sesuai dengan pertanyaan Anda. Silakan coba dengan nama produk yang berbeda atau tanyakan tentang produk yang tersedia.`;
      }
      
      switch ((inquiryType || 'general').toLowerCase()) {
        case 'price':
          return this.formatPriceInformation(products);
        case 'availability':
          return this.formatAvailabilityInformation(products);
        case 'features':
          return this.formatFeaturesInformation(products);
        default:
          return this.formatGeneralInformation(products);
      }
    } catch (error) {
      logger.error(`Error in formatProductInformation: ${error.message}`);
      return `Maaf, terjadi kesalahan saat memproses informasi produk. Silakan coba lagi nanti.`;
    }
  }

  /**
   * Formats price information for products
   * @param {Array} products - Matching products
   * @returns {string} - Formatted price information
   * @private
   */
  formatPriceInformation(products) {
    if (products.length === 1) {
      const product = products[0];
      return `💰 **Informasi Harga**\n\n• ${product.name} (${product.code})\n  Harga: ${product.price}\n  Status: ${parseInt(product.stock) > 0 ? '✅ Tersedia' : '❌ Kosong'}\n\nAda yang ingin ditanyakan lagi? 😊`;
    }
    
    const productList = products.map(product => {
      return `• ${product.name} (${product.code})\n  Harga: ${product.price}\n  Status: ${parseInt(product.stock) > 0 ? '✅ Tersedia' : '❌ Kosong'}`;
    }).join('\n\n');
    
    return `💰 **Informasi Harga Produk**\n\n${productList}\n\nAda yang ingin ditanyakan lagi atau ingin memesan salah satu produk di atas? 😊`;
  }

  /**
   * Formats availability information for products
   * @param {Array} products - Matching products
   * @returns {string} - Formatted availability information
   * @private
   */
  formatAvailabilityInformation(products) {
    if (products.length === 1) {
      const product = products[0];
      const isAvailable = parseInt(product.stock) > 0;
      
      return `📦 **Informasi Ketersediaan**\n\n• ${product.name} (${product.code})\n  Status: ${isAvailable ? '✅ Tersedia' : '❌ Kosong'}\n  ${isAvailable ? `Stok: ${product.stock}` : 'Mohon maaf, produk ini sedang kosong'}\n\n${isAvailable ? 'Apakah Anda ingin memesan produk ini?' : 'Apakah Anda ingin melihat produk lain yang tersedia?'} 😊`;
    }
    
    const availableProducts = products.filter(p => parseInt(p.stock) > 0);
    const unavailableProducts = products.filter(p => parseInt(p.stock) <= 0);
    
    let response = `📦 **Informasi Ketersediaan Produk**\n\n`;
    
    if (availableProducts.length > 0) {
      response += `✅ **Produk Tersedia:**\n${availableProducts.map(p => `• ${p.name} (${p.code}) - Stok: ${p.stock}`).join('\n')}\n\n`;
    }
    
    if (unavailableProducts.length > 0) {
      response += `❌ **Produk Kosong:**\n${unavailableProducts.map(p => `• ${p.name} (${p.code})`).join('\n')}\n\n`;
    }
    
    response += `Ada yang ingin ditanyakan lagi atau ingin memesan salah satu produk yang tersedia? 😊`;
    
    return response;
  }

  /**
   * Formats feature information for products
   * @param {Array} products - Matching products
   * @returns {string} - Formatted feature information
   * @private
   */
  formatFeaturesInformation(products) {
    if (products.length === 1) {
      const product = products[0];
      return `✨ **Informasi Fitur**\n\n• ${product.name} (${product.code})\n  ${product.desc || 'Tidak ada deskripsi detail untuk produk ini'}\n\nAda yang ingin ditanyakan lagi? 😊`;
    }
    
    const productList = products.map(product => {
      return `• ${product.name} (${product.code})\n  ${product.desc || 'Tidak ada deskripsi detail untuk produk ini'}`;
    }).join('\n\n');
    
    return `✨ **Informasi Fitur Produk**\n\n${productList}\n\nAda yang ingin ditanyakan lagi atau ingin memesan salah satu produk di atas? 😊`;
  }

  /**
   * Formats general information for products
   * @param {Array} products - Matching products
   * @returns {string} - Formatted general information
   * @private
   */
  formatGeneralInformation(products) {
    if (products.length === 1) {
      const product = products[0];
      const isAvailable = parseInt(product.stock) > 0;
      
      return `📝 **Informasi Produk**\n\n• ${product.name} (${product.code})\n  Harga: ${product.price}\n  Status: ${isAvailable ? '✅ Tersedia' : '❌ Kosong'}\n  ${product.desc ? `Deskripsi: ${product.desc}` : ''}\n\nAda yang ingin ditanyakan lagi? 😊`;
    }
    
    const productList = products.slice(0, 5).map(product => {
      const isAvailable = parseInt(product.stock) > 0;
      return `• ${product.name} (${product.code})\n  Harga: ${product.price}\n  Status: ${isAvailable ? '✅ Tersedia' : '❌ Kosong'}`;
    }).join('\n\n');
    
    return `📝 **Informasi Produk**\n\n${productList}\n\n${products.length > 5 ? `...dan ${products.length - 5} produk lainnya\n\n` : ''}Ada yang ingin ditanyakan lebih detail tentang salah satu produk di atas? 😊`;
  }
}

module.exports = {
  OrderProcessingTool,
  ProductInquiryTool
};