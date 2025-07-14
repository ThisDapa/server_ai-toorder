/**
 * Test script for BrainService with FAISSstore
 */

const BrainService = require('./src/services/BrainService');

async function testBrainService() {
  console.log('Initializing BrainService...');
  const brainService = new BrainService();
  
  try {
    // Initialize the service
    await brainService.init();
    console.log('BrainService initialized successfully!');
    
    // Get model stats
    const stats = await brainService.getStats();
    console.log('Model stats:', stats);
    
    // Test with a sample question
    const question = 'Halo kak siang';
    console.log(`\nTesting with question: "${question}"`);
    
    // Process context
    console.log('\nProcessing context...');
    const context = await brainService.processContext(question);
    console.log('Context results:', context);
    
    // Get predicted tags
    console.log('\nGetting predicted tags...');
    const tags = await brainService.getPredictedTags(question);
    console.log('Predicted tags:', tags);
    
    // Find answer
    console.log('\nFinding answer...');
    const answer = await brainService.findAnswer(question);
    console.log('Answer:', answer);
    
    console.log('\nAll tests completed successfully!');
  } catch (error) {
    console.error('Error during test:', error);
  }
}

// Run the test
testBrainService();