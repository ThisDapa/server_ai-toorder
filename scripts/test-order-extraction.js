#!/usr/bin/env node

/**
 * Script untuk menguji ekstraksi pesanan dari chat dengan Ollama
 * Jalankan dengan: node scripts/test-order-extraction.js
 */

require('dotenv').config();
const OllamaService = require('../src/services/OllamaService');
const readline = require('readline');

// Setup readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function colorLog(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

// Inisialisasi OllamaService
const ollamaService = new OllamaService();

// Fungsi untuk menguji ekstraksi pesanan
async function testOrderExtraction(chatHistory) {
  try {
    // Inisialisasi service jika belum
    if (!ollamaService.initialized) {
      colorLog('\n🔄 Initializing Ollama Service...', 'cyan');
      await ollamaService.init();
      colorLog('✅ Ollama Service Initialized', 'green');
    }

    // Dapatkan data produk
    const productData = ollamaService.getProductData();
    
    // Format chat history untuk pengujian
    const formattedHistory = chatHistory.map(msg => {
      return msg.role === 'user' 
        ? { type: 'user', content: msg.content }
        : { type: 'ai', content: msg.content };
    });

    colorLog('\n🔍 Extracting Orders from Chat...', 'cyan');
    console.log('Chat History:', JSON.stringify(formattedHistory, null, 2));
    
    // Ekstrak pesanan dari chat
    const startTime = Date.now();
    const result = await ollamaService.extractOrdersFromChat(formattedHistory, productData);
    const processingTime = Date.now() - startTime;
    
    if (result.success) {
      colorLog('\n✅ Orders Extracted Successfully', 'green');
      colorLog(`⏱️ Processing Time: ${processingTime}ms`, 'magenta');
      colorLog('\n📋 Extracted Orders:', 'bright');
      console.log(JSON.stringify(result.orders, null, 2));
    } else {
      colorLog('\n❌ Order Extraction Failed', 'red');
      colorLog(`Error: ${result.error}`, 'red');
    }
    
    return result;
  } catch (error) {
    colorLog('\n💥 Error Testing Order Extraction', 'red');
    console.error(error);
    return { success: false, error: error.message };
  }
}

// Fungsi untuk menguji handleOrderTag
async function testHandleOrderTag(chatHistory) {
  try {
    // Inisialisasi service jika belum
    if (!ollamaService.initialized) {
      colorLog('\n🔄 Initializing Ollama Service...', 'cyan');
      await ollamaService.init();
      colorLog('✅ Ollama Service Initialized', 'green');
    }
    
    // Format chat history untuk pengujian
    const formattedHistory = chatHistory.map(msg => {
      return msg.role === 'user' 
        ? { type: 'user', content: msg.content }
        : { type: 'ai', content: msg.content };
    });

    colorLog('\n🔍 Testing handleOrderTag...', 'cyan');
    console.log('Chat History:', JSON.stringify(formattedHistory, null, 2));
    
    // Panggil handleOrderTag
    const startTime = Date.now();
    const result = await ollamaService.handleOrderTag(formattedHistory);
    const processingTime = Date.now() - startTime;
    
    colorLog('\n✅ handleOrderTag Completed', 'green');
    colorLog(`⏱️ Processing Time: ${processingTime}ms`, 'magenta');
    colorLog('\n📋 Result:', 'bright');
    console.log(JSON.stringify(result, null, 2));
    
    return result;
  } catch (error) {
    colorLog('\n💥 Error Testing handleOrderTag', 'red');
    console.error(error);
    return { success: false, error: error.message };
  }
}

// Fungsi untuk menjalankan mode interaktif
async function runInteractiveMode() {
  colorLog('\n🤖 Order Extraction Test Mode', 'bright');
  colorLog('Choose a test option:', 'cyan');
  colorLog('1. Test with sample chat history', 'blue');
  colorLog('2. Test with custom chat history', 'blue');
  colorLog('3. Test handleOrderTag function', 'blue');
  colorLog('4. Exit', 'blue');
  
  rl.question('\n🔢 Enter your choice (1-4): ', async (choice) => {
    switch (choice) {
      case '1':
        // Sample chat history
        const sampleChatHistory = [
          { role: 'user', content: 'Halo, saya mau pesan Netflix 1P2U 2 akun' },
          { role: 'ai', content: 'Baik, ada yang bisa saya bantu lagi?' },
          { role: 'user', content: 'Saya juga mau pesan Disney+ Hotstar 1 akun' }
        ];
        await testOrderExtraction(sampleChatHistory);
        break;
        
      case '2':
        // Custom chat history
        rl.question('\n📝 Enter your chat message (user): ', async (userMessage) => {
          const customChatHistory = [
            { role: 'user', content: userMessage }
          ];
          await testOrderExtraction(customChatHistory);
          runInteractiveMode();
        });
        return; // Don't continue to avoid closing rl
        
      case '3':
        // Test handleOrderTag
        rl.question('\n📝 Enter your chat message (user): ', async (userMessage) => {
          const customChatHistory = [
            { role: 'user', content: userMessage }
          ];
          await testHandleOrderTag(customChatHistory);
          runInteractiveMode();
        });
        return; // Don't continue to avoid closing rl
        
      case '4':
        colorLog('👋 Goodbye!', 'green');
        rl.close();
        return;
        
      default:
        colorLog('⚠️ Invalid choice. Please try again.', 'yellow');
        break;
    }
    
    // Return to menu after test completes
    setTimeout(() => {
      runInteractiveMode();
    }, 1000);
  });
}

// Fungsi untuk menjalankan test otomatis
async function runAutomaticTest() {
  try {
    colorLog('\n🤖 Running Automatic Order Extraction Test', 'bright');
    
    // Test dengan sample chat history
    const sampleChatHistory = [
      { role: 'user', content: 'Halo, saya mau pesan Netflix 1P2U 2 akun' },
      { role: 'ai', content: 'Baik, ada yang bisa saya bantu lagi?' },
      { role: 'user', content: 'Saya juga mau pesan Disney+ Hotstar 1 akun' }
    ];
    
    colorLog('\n🔍 Test 1: Sample Chat History', 'cyan');
    await testOrderExtraction(sampleChatHistory);
    
    // Test dengan chat yang tidak jelas
    const ambiguousChatHistory = [
      { role: 'user', content: 'Berapa harga Netflix?' },
      { role: 'ai', content: 'Harga Netflix 1P2U adalah Rp 13.000 dan Netflix 1P1U adalah Rp 24.000' },
      { role: 'user', content: 'Bagaimana dengan Disney+ Hotstar?' }
    ];
    
    colorLog('\n🔍 Test 2: Ambiguous Chat (Should Not Extract Orders)', 'cyan');
    await testOrderExtraction(ambiguousChatHistory);
    
    // Test dengan chat yang berisi pesanan jelas
    const clearOrderChatHistory = [
      { role: 'user', content: 'Mau dong Gmail 2' }
    ];
    
    colorLog('\n🔍 Test 3: Clear Order Chat', 'cyan');
    await testOrderExtraction(clearOrderChatHistory);
    
    // Test handleOrderTag
    colorLog('\n🔍 Test 4: handleOrderTag Function', 'cyan');
    await testHandleOrderTag(clearOrderChatHistory);
    
    colorLog('\n✅ All tests completed!', 'green');
    process.exit(0);
  } catch (error) {
    colorLog('\n💥 Error in automatic test', 'red');
    console.error(error);
    process.exit(1);
  }
}

// Main function
async function main() {
  try {
    // Check if running in automatic mode
    const args = process.argv.slice(2);
    if (args.includes('--auto')) {
      await runAutomaticTest();
    } else {
      await runInteractiveMode();
    }
  } catch (error) {
    colorLog('\n💥 Error in main function', 'red');
    console.error(error);
    rl.close();
    process.exit(1);
  }
}

// Run the main function
main();