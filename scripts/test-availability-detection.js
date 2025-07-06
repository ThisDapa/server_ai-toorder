const OllamaService = require('../src/services/OllamaService');
const logger = require('../src/utils/logger');

async function testAvailabilityDetection() {
  console.log('Initializing OllamaService...');
  const ollamaService = new OllamaService();
  await ollamaService.init();
  console.log('OllamaService initialized successfully');

  // Test cases for availability questions
  const availabilityQuestions = [
    'Apakah Netflix 1P1U masih ada stok?',
    'Stock Netflix Premium 4K UHD masih tersedia?',
    'Stok Gmail Fresh masih ada?',
    'Apakah masih tersedia akun Netflix?',
    'Gmail Aged masih ready?',
    'Berapa stok Netflix yang tersisa?',
    'Netflix 1P2U sudah habis?',
    'Masih ada Disney+ Hotstar?',
    'Stoknya Netflix 1P1U masih ada?',
    'Netflix Premium 4K UHD masih bisa dibeli?'
  ];

  // Test cases for order questions that should NOT be detected as availability
  const orderQuestions = [
    'Saya mau beli Netflix 1P1U',
    'Pesan Gmail Fresh 1 akun',
    'Mau order Netflix Premium 4K UHD',
    'Beli 2 akun Gmail Aged',
    'Saya butuh 3 Netflix 1P2U',
    'Tambahkan Disney+ Hotstar ke keranjang',
    'Checkout pesanan saya',
    'Saya mau Netflix 1P1U'
  ];

  console.log('\n===== Testing Availability Questions =====');
  for (const question of availabilityQuestions) {
    const tag = await ollamaService.getQuestionTag(question, []);
    console.log(`Question: "${question}"`)
    console.log(`Detected Tag: ${tag}`);
    console.log(`Correct Detection: ${tag === 'availability' ? '✅' : '❌'}`);
    console.log('---');
  }

  console.log('\n===== Testing Order Questions =====');
  for (const question of orderQuestions) {
    const tag = await ollamaService.getQuestionTag(question, []);
    console.log(`Question: "${question}"`)
    console.log(`Detected Tag: ${tag}`);
    console.log(`Correct Detection: ${tag === 'order' ? '✅' : '❌'}`);
    console.log('---');
  }

  // Test the specific case mentioned by the user
  const specificCase = 'stock netflix 1p1u masih ada?';
  console.log('\n===== Testing Specific Case =====');
  console.log(`Question: "${specificCase}"`);
  const tag = await ollamaService.getQuestionTag(specificCase, []);
  console.log(`Detected Tag: ${tag}`);
  console.log(`Correct Detection: ${tag === 'availability' ? '✅' : '❌'}`);
}

testAvailabilityDetection().catch(err => {
  console.error('Error during testing:', err);
});