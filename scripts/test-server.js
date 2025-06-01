#!/usr/bin/env node

/**
 * Script untuk menguji server AI secara manual
 * Jalankan dengan: node scripts/test-server.js
 */

const axios = require('axios');
const readline = require('readline');

const BASE_URL = 'http://localhost:3000';

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

// Test functions
async function testHealthCheck() {
  try {
    colorLog('\n🔍 Testing Health Check...', 'cyan');
    const response = await axios.get(`${BASE_URL}/health`);
    colorLog('✅ Health Check Passed', 'green');
    console.log(JSON.stringify(response.data, null, 2));
    return true;
  } catch (error) {
    colorLog('❌ Health Check Failed', 'red');
    console.log(error.message);
    return false;
  }
}

async function submitQuestion(question, context = {}) {
  try {
    colorLog(`\n📝 Submitting Question: "${question}"`, 'cyan');
    const response = await axios.post(`${BASE_URL}/api/questions/ask`, {
      question,
      context
    });
    
    if (response.data.success) {
      colorLog('✅ Question Submitted Successfully', 'green');
      console.log(JSON.stringify(response.data, null, 2));
      return response.data.questionId;
    } else {
      colorLog('❌ Question Submission Failed', 'red');
      console.log(response.data);
      return null;
    }
  } catch (error) {
    colorLog('❌ Error Submitting Question', 'red');
    console.log(error.response?.data || error.message);
    return null;
  }
}

async function checkStatus(questionId) {
  try {
    const response = await axios.get(`${BASE_URL}/api/questions/status/${questionId}`);
    return response.data;
  } catch (error) {
    colorLog('❌ Error Checking Status', 'red');
    console.log(error.response?.data || error.message);
    return null;
  }
}

async function waitForCompletion(questionId, maxWaitTime = 60000) {
  colorLog(`\n⏳ Waiting for question ${questionId} to complete...`, 'yellow');
  
  const startTime = Date.now();
  const checkInterval = 2000; // Check every 2 seconds
  
  while (Date.now() - startTime < maxWaitTime) {
    const status = await checkStatus(questionId);
    
    if (!status) {
      break;
    }
    
    colorLog(`📊 Status: ${status.stage} - ${status.message}`, 'blue');
    
    if (status.stage === 'completed') {
      colorLog('🎉 Question Processing Completed!', 'green');
      console.log('\n📋 Final Result:');
      console.log(JSON.stringify(status.response, null, 2));
      colorLog(`⏱️  Processing Time: ${status.processingTime}ms`, 'magenta');
      return status;
    }
    
    if (status.stage === 'error') {
      colorLog('💥 Question Processing Failed', 'red');
      console.log(status.message);
      return status;
    }
    
    // Wait before next check
    await new Promise(resolve => setTimeout(resolve, checkInterval));
  }
  
  colorLog('⏰ Timeout waiting for completion', 'yellow');
  return null;
}

async function runInteractiveMode() {
  colorLog('\n🤖 AI Server Interactive Test Mode', 'bright');
  colorLog('Type your questions or commands:', 'cyan');
  colorLog('- "health" - Check server health', 'blue');
  colorLog('- "exit" - Exit the program', 'blue');
  colorLog('- Any other text will be sent as a question\n', 'blue');
  
  const askQuestion = () => {
    rl.question('❓ Enter your question: ', async (input) => {
      const trimmedInput = input.trim();
      
      if (trimmedInput.toLowerCase() === 'exit') {
        colorLog('👋 Goodbye!', 'green');
        rl.close();
        return;
      }
      
      if (trimmedInput.toLowerCase() === 'health') {
        await testHealthCheck();
        askQuestion();
        return;
      }
      
      if (trimmedInput.length === 0) {
        colorLog('⚠️  Please enter a question', 'yellow');
        askQuestion();
        return;
      }
      
      // Submit question and wait for completion
      const questionId = await submitQuestion(trimmedInput, {
        user_id: 'test_user',
        session_id: Date.now().toString(),
        timestamp: new Date().toISOString()
      });
      
      if (questionId) {
        await waitForCompletion(questionId);
      }
      
      colorLog('\n' + '='.repeat(50), 'cyan');
      askQuestion();
    });
  };
  
  askQuestion();
}

async function runAutomatedTests() {
  colorLog('\n🧪 Running Automated Tests', 'bright');
  
  // Test 1: Health Check
  const healthOk = await testHealthCheck();
  if (!healthOk) {
    colorLog('❌ Server is not healthy. Please check if server is running.', 'red');
    return;
  }
  
  // Test 2: Submit various questions
  const testQuestions = [
    'What is artificial intelligence?',
    'How to cook pasta?',
    'What is the weather like today?',
    'How to learn programming?'
  ];
  
  for (const question of testQuestions) {
    const questionId = await submitQuestion(question, {
      test: true,
      automated: true
    });
    
    if (questionId) {
      // Wait a bit then check status
      await new Promise(resolve => setTimeout(resolve, 3000));
      const status = await checkStatus(questionId);
      if (status) {
        colorLog(`📊 Current Status: ${status.stage}`, 'blue');
      }
    }
    
    // Small delay between questions
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  colorLog('\n✅ Automated tests completed', 'green');
}

// Main function
async function main() {
  const args = process.argv.slice(2);
  
  if (args.includes('--auto') || args.includes('-a')) {
    await runAutomatedTests();
  } else {
    // Check if server is running first
    const healthOk = await testHealthCheck();
    if (!healthOk) {
      colorLog('\n❌ Server is not running or not accessible.', 'red');
      colorLog('Please make sure the server is started with: npm start or npm run dev', 'yellow');
      process.exit(1);
    }
    
    await runInteractiveMode();
  }
}

// Handle Ctrl+C gracefully
process.on('SIGINT', () => {
  colorLog('\n👋 Goodbye!', 'green');
  process.exit(0);
});

// Run the script
if (require.main === module) {
  main().catch(error => {
    colorLog(`💥 Unexpected error: ${error.message}`, 'red');
    process.exit(1);
  });
}