/**
 * Context utilities for BrainService
 * Extracted from BrainService.js to improve maintainability
 */

const textUtils = require('./textUtils');
const keywords = require('./keywords');

module.exports = {
  /**
   * Calculates contextual relevance between a question and a context entry
   * @param {string} question - User question
   * @param {Object} entry - Context entry with question and answer
   * @param {Object} options - Calculation options
   * @returns {number} - Relevance score between 0-1
   */
  calculateContextualRelevance(question, entry, options = {}) {
    if (!question || !entry || !entry.question) return 0;
    
    const {
      useExactMatch = true,
      useStemmedMatch = true,
      useSemanticMatch = true,
      useEntityMatch = true,
      useKeywordMatch = true,
      useTopicMatch = true,
      useIntentMatch = true,
      weights = {
        exact: 1.0,
        stemmed: 0.8,
        semantic: 0.7,
        entity: 0.9,
        keyword: 0.6,
        topic: 0.5,
        intent: 0.4
      }
    } = options;
    
    // Normalize and prepare texts
    const normalizedQuestion = textUtils.normalizeText(question);
    const normalizedEntryQuestion = textUtils.normalizeText(entry.question);
    
    // Calculate different similarity scores
    let scores = {};
    let totalWeight = 0;
    let weightedScore = 0;
    
    // Exact match (fastest)
    if (useExactMatch) {
      scores.exact = normalizedQuestion === normalizedEntryQuestion ? 1 : 0;
      totalWeight += weights.exact;
      weightedScore += scores.exact * weights.exact;
    }
    
    // Stemmed match
    if (useStemmedMatch) {
      const stemmedQuestion = textUtils.stemText(normalizedQuestion);
      const stemmedEntryQuestion = textUtils.stemText(normalizedEntryQuestion);
      scores.stemmed = stemmedQuestion === stemmedEntryQuestion ? 1 : 0;
      totalWeight += weights.stemmed;
      weightedScore += scores.stemmed * weights.stemmed;
    }
    
    // Semantic similarity (using word overlap as approximation)
    if (useSemanticMatch) {
      scores.semantic = textUtils.calculateWordOverlap(normalizedQuestion, normalizedEntryQuestion);
      totalWeight += weights.semantic;
      weightedScore += scores.semantic * weights.semantic;
    }
    
    // Entity match
    if (useEntityMatch) {
      scores.entity = textUtils.calculateEntityOverlap(normalizedQuestion, normalizedEntryQuestion);
      totalWeight += weights.entity;
      weightedScore += scores.entity * weights.entity;
    }
    
    // Keyword match
    if (useKeywordMatch) {
      scores.keyword = this.calculateKeywordMatch(normalizedQuestion, normalizedEntryQuestion);
      totalWeight += weights.keyword;
      weightedScore += scores.keyword * weights.keyword;
    }
    
    // Topic match
    if (useTopicMatch) {
      scores.topic = this.calculateTopicMatch(normalizedQuestion, normalizedEntryQuestion);
      totalWeight += weights.topic;
      weightedScore += scores.topic * weights.topic;
    }
    
    // Intent match
    if (useIntentMatch) {
      scores.intent = this.calculateIntentMatch(normalizedQuestion, normalizedEntryQuestion);
      totalWeight += weights.intent;
      weightedScore += scores.intent * weights.intent;
    }
    
    // Calculate final weighted score
    return totalWeight > 0 ? weightedScore / totalWeight : 0;
  },
  
  /**
   * Calculates keyword match score between two texts
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Keyword match score between 0-1
   */
  calculateKeywordMatch(text1, text2) {
    const keywordsInText1 = this.extractKeywords(text1);
    const keywordsInText2 = this.extractKeywords(text2);
    
    if (keywordsInText1.length === 0 || keywordsInText2.length === 0) {
      return 0;
    }
    
    const commonKeywords = keywordsInText1.filter(kw => keywordsInText2.includes(kw));
    return commonKeywords.length / Math.max(keywordsInText1.length, keywordsInText2.length);
  },
  
  /**
   * Extracts keywords from text
   * @param {string} text - Text to extract keywords from
   * @returns {Array<string>} - Array of keywords
   */
  extractKeywords(text) {
    if (!text) return [];
    
    // Split text into words and filter out stopwords
    const words = text.split(/\s+/);
    const filteredWords = words.filter(word => 
      word.length > 2 && !keywords.stopwords.includes(word.toLowerCase())
    );
    
    // Add any specific keywords found in the text
    const extractedKeywords = new Set(filteredWords);
    
    // Check for domain-specific keywords
    for (const keyword of keywords.domainKeywords) {
      if (text.toLowerCase().includes(keyword.toLowerCase())) {
        extractedKeywords.add(keyword);
      }
    }
    
    return Array.from(extractedKeywords);
  },
  
  /**
   * Calculates topic match score between two texts
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Topic match score between 0-1
   */
  calculateTopicMatch(text1, text2) {
    // Extract topics using a simple keyword-based approach
    const topics1 = this.extractTopics(text1);
    const topics2 = this.extractTopics(text2);
    
    if (topics1.length === 0 || topics2.length === 0) {
      return 0;
    }
    
    const commonTopics = topics1.filter(topic => topics2.includes(topic));
    return commonTopics.length / Math.max(topics1.length, topics2.length);
  },
  
  /**
   * Extracts topics from text
   * @param {string} text - Text to extract topics from
   * @returns {Array<string>} - Array of topics
   */
  extractTopics(text) {
    if (!text) return [];
    
    const extractedTopics = new Set();
    const lowerText = text.toLowerCase();
    
    // Check for each topic category
    for (const [topic, keywords] of Object.entries(keywords.topicKeywords)) {
      for (const keyword of keywords) {
        if (lowerText.includes(keyword.toLowerCase())) {
          extractedTopics.add(topic);
          break;
        }
      }
    }
    
    return Array.from(extractedTopics);
  },
  
  /**
   * Calculates intent match score between two texts
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Intent match score between 0-1
   */
  calculateIntentMatch(text1, text2) {
    // Extract intents using a simple keyword-based approach
    const intents1 = this.extractIntents(text1);
    const intents2 = this.extractIntents(text2);
    
    if (intents1.length === 0 || intents2.length === 0) {
      return 0;
    }
    
    const commonIntents = intents1.filter(intent => intents2.includes(intent));
    return commonIntents.length / Math.max(intents1.length, intents2.length);
  },
  
  /**
   * Extracts intents from text
   * @param {string} text - Text to extract intents from
   * @returns {Array<string>} - Array of intents
   */
  extractIntents(text) {
    if (!text) return [];
    
    const extractedIntents = new Set();
    const lowerText = text.toLowerCase();
    
    // Check for each intent category
    for (const [intent, keywords] of Object.entries(keywords.intentKeywords)) {
      for (const keyword of keywords) {
        if (lowerText.includes(keyword.toLowerCase())) {
          extractedIntents.add(intent);
          break;
        }
      }
    }
    
    return Array.from(extractedIntents);
  }
};