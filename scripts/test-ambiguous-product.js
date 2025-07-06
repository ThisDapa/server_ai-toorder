const OllamaService = require('../src/services/OllamaService');
const logger = require('../src/utils/logger');

async function testAmbiguousProductHandling() {
  try {
    console.log('Initializing OllamaService...');
    const ollamaService = new OllamaService();
    await ollamaService.init();
    console.log('OllamaService initialized successfully');

    // Test case 1: Ambiguous Netflix order
    console.log('\n\n===== Test Case 1: Ambiguous Netflix Order =====');
    const mockHistory1 = [
      { type: 'user', content: 'mau netflix nya 1', timestamp: new Date().toISOString() }
    ];
    const result1 = await ollamaService.handleOrderTag(mockHistory1);
    console.log('Result for ambiguous Netflix order:');
    console.log(result1.response);

    // Test case 2: Ambiguous Gmail order
    console.log('\n\n===== Test Case 2: Ambiguous Gmail Order =====');
    const mockHistory2 = [
      { type: 'user', content: 'mau beli gmail', timestamp: new Date().toISOString() }
    ];
    const result2 = await ollamaService.handleOrderTag(mockHistory2);
    console.log('Result for ambiguous Gmail order:');
    console.log(result2.response);

    // Test case 3: Product clarification response
    console.log('\n\n===== Test Case 3: Product Clarification Response =====');
    const mockHistory3 = [
      { type: 'user', content: 'mau beli gmail', timestamp: new Date().toISOString() },
      { 
        type: 'ai', 
        content: '⚠️ **Mohon Klarifikasi**\n\nSaya menemukan beberapa varian produk Gmail:\n\n1. **Gmail Fresh** (GM-FRESH) - Rp 15.000\n2. **Gmail Bekas** (GM-BEKAS) - Rp 8.000\n\nMohon tentukan varian mana yang Anda inginkan.', 
        timestamp: new Date().toISOString() 
      },
      { type: 'user', content: 'gmail fresh', timestamp: new Date().toISOString() }
    ];
    
    // Get question tag first
    const tag = await ollamaService.getQuestionTag('gmail fresh', mockHistory3);
    console.log(`Question tag: ${tag}`);
    
    let result3;
    if (tag === 'product_clarification') {
      result3 = await ollamaService.handleProductClarification('gmail fresh', mockHistory3);
    } else {
      result3 = await ollamaService.processWithAI('gmail fresh', {}, 'test-user');
    }
    
    console.log('Result for product clarification response:');
    console.log(result3.response);
    console.log('Product info:', JSON.stringify(result3.productInfo, null, 2));
    
    // Test case 4: Direct product clarification
    console.log('\n\n===== Test Case 4: Direct Product Clarification =====');
    const mockHistory4 = [
      { type: 'user', content: 'mau beli gmail', timestamp: new Date().toISOString() },
      { 
        type: 'ai', 
        content: '⚠️ **Mohon Klarifikasi**\n\nSaya menemukan beberapa varian produk "gmail" yang tersedia. Mohon tentukan varian mana yang Anda inginkan:\n\n• **Akun Gmail Fresh** (GMF): Rp 15.000/akun\n  Akun baru, belum pernah dipakai\n\n• **Akun Gmail Aged** (GMA): Rp 15.000/akun\n  Akun berumur 1+ tahun\n\nSilakan balas dengan nama lengkap atau kode produk yang Anda inginkan.', 
        timestamp: new Date().toISOString() 
      },
      { type: 'user', content: 'GMF', timestamp: new Date().toISOString() }
    ];
    
    // Get question tag first
    const tag4 = await ollamaService.getQuestionTag('GMF', mockHistory4);
    console.log(`Question tag: ${tag4}`);
    
    let result4;
    if (tag4 === 'product_clarification') {
      result4 = await ollamaService.handleProductClarification('GMF', mockHistory4);
    } else {
      result4 = await ollamaService.processWithAI('GMF', {}, 'test-user');
    }
    
    console.log('Result for product clarification response:');
    console.log(result3.response);
    console.log('Product info:', JSON.stringify(result3.productInfo, null, 2));

    console.log('\n\nAll tests completed!');
  } catch (error) {
    console.error('Error during testing:', error);
  }
}

testAmbiguousProductHandling();