/**
 * Text processing utilities for BrainService
 * Extracted from BrainService.js to improve maintainability
 */

const fuzz = require('fuzzball');

module.exports = {
  /**
   * Normalizes text by converting to lowercase, removing extra spaces and special characters
   * @param {string} text - Text to normalize
   * @returns {string} - Normalized text
   */
  normalizeText(text) {
    if (!text) return "";
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ") // Replace special chars with space
      .replace(/\s+/g, " ") // Replace multiple spaces with single space
      .trim();
  },

  /**
   * Indonesian Porter Stemmer implementation
   * Removes common Indonesian suffixes to find word roots
   * @param {string} word - Word to stem
   * @returns {string} - Stemmed word
   */
  indonesianStemmer(word) {
    if (!word || word.length < 3) return word;

    let stemmed = word.toLowerCase();

    // Remove possessive pronouns (ku-, mu-, nya)
    stemmed = stemmed.replace(/^(ku|mu)/, "");
    stemmed = stemmed.replace(/nya$/, "");

    // Remove particles (lah, kah, tah, pun)
    stemmed = stemmed.replace(/(lah|kah|tah|pun)$/, "");

    // Remove derivational suffixes
    // First precedence: -kan, -an, -i
    if (stemmed.length > 4) {
      stemmed = stemmed.replace(/(kan|an|i)$/, "");
    }

    // Second precedence: -nya after other suffixes
    stemmed = stemmed.replace(/nya$/, "");

    // Remove prefixes
    // Remove "ber-" prefix
    if (stemmed.startsWith("ber") && stemmed.length > 5) {
      stemmed = stemmed.substring(3);
    }
    // Remove "me-" prefix variations
    else if (stemmed.startsWith("me") && stemmed.length > 4) {
      if (stemmed.startsWith("men") && stemmed.length > 5) {
        stemmed = stemmed.substring(3);
      } else if (stemmed.startsWith("mem") && stemmed.length > 5) {
        stemmed = stemmed.substring(3);
      } else if (stemmed.startsWith("meng") && stemmed.length > 6) {
        stemmed = stemmed.substring(4);
      } else {
        stemmed = stemmed.substring(2);
      }
    }
    // Remove "di-" prefix
    else if (stemmed.startsWith("di") && stemmed.length > 4) {
      stemmed = stemmed.substring(2);
    }
    // Remove "ter-" prefix
    else if (stemmed.startsWith("ter") && stemmed.length > 5) {
      stemmed = stemmed.substring(3);
    }
    // Remove "ke-" prefix
    else if (stemmed.startsWith("ke") && stemmed.length > 4) {
      stemmed = stemmed.substring(2);
    }
    // Remove "se-" prefix
    else if (stemmed.startsWith("se") && stemmed.length > 4) {
      stemmed = stemmed.substring(2);
    }

    return stemmed.length >= 2 ? stemmed : word;
  },

  /**
   * Stems a text by applying indonesianStemmer to each word
   * @param {string} text - Text to stem
   * @returns {string} - Stemmed text
   */
  stemText(text) {
    if (!text) return "";

    return text
      .split(" ")
      .map((word) => this.indonesianStemmer(word))
      .join(" ");
  },

  /**
   * Calculates Jaccard similarity between two texts
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Similarity score between 0-1
   */
  calculateJaccardSimilarity(text1, text2) {
    if (!text1 || !text2) return 0;
    if (text1 === text2) return 1;

    const set1 = new Set(text1.split(' '));
    const set2 = new Set(text2.split(' '));
    
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    
    return intersection.size / union.size;
  },

  /**
   * Calculates fuzzy similarity between two texts
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Similarity score between 0-1
   */
  calculateFuzzySimilarity(text1, text2) {
    if (!text1 || !text2) return 0;
    if (text1 === text2) return 1;
    
    // Use fuzzball for fuzzy matching
    const ratio = fuzz.ratio(text1, text2) / 100;
    return ratio;
  },

  /**
   * Checks if text contains any of the keywords
   * @param {string} text - Text to check
   * @param {Array<string>} keywords - Keywords to look for
   * @returns {boolean} - True if any keyword is found
   */
  containsAnyKeyword(text, keywords) {
    if (!text || !keywords || !keywords.length) return false;
    
    const lowerText = text.toLowerCase();
    return keywords.some(keyword => lowerText.includes(keyword.toLowerCase()));
  },

  /**
   * Fuzzy matches text against a list of patterns
   * @param {string} text - Text to match
   * @param {Array<string>} patterns - Patterns to match against
   * @param {number} threshold - Matching threshold (0-100)
   * @returns {boolean} - True if any pattern matches above threshold
   */
  fuzzyMatch(text, patterns, threshold = 80) {
    if (!text || !patterns || !patterns.length) return false;
    
    const lowerText = text.toLowerCase();
    
    for (const pattern of patterns) {
      const lowerPattern = pattern.toLowerCase();
      
      // Direct inclusion check first (faster)
      if (lowerText.includes(lowerPattern)) {
        return true;
      }
      
      // Fuzzy match for more complex cases
      const ratio = fuzz.partial_ratio(lowerText, lowerPattern);
      if (ratio >= threshold) {
        return true;
      }
    }
    
    return false;
  },

  /**
   * Detects if text is likely in English
   * @param {string} text - Text to analyze
   * @param {Array<string>} englishMarkers - English marker words
   * @returns {boolean} - True if text is likely in English
   */
  isEnglishText(text, englishMarkers) {
    if (!text || !englishMarkers) return false;
    
    // Count English marker words
    const words = text.split(" ");
    let englishCount = 0;

    for (const word of words) {
      if (englishMarkers.includes(word)) {
        englishCount++;
      }
    }

    // Check if a significant portion of words are English markers
    return englishCount >= 1 && englishCount / words.length > 0.15;
  },

  /**
   * Calculates sentiment score for text
   * @param {string} text - Text to analyze
   * @param {Array<string>} positiveWords - Positive sentiment words
   * @param {Array<string>} negativeWords - Negative sentiment words
   * @returns {number} - Sentiment score from -1 (negative) to 1 (positive)
   */
  calculateSentiment(text, positiveWords, negativeWords) {
    if (!text) return 0;
    
    const lowerText = text.toLowerCase();
    let positiveScore = 0;
    let negativeScore = 0;

    // Count positive words
    for (const word of positiveWords) {
      if (lowerText.includes(word.toLowerCase())) {
        positiveScore++;
      }
    }

    // Count negative words
    for (const word of negativeWords) {
      if (lowerText.includes(word.toLowerCase())) {
        negativeScore++;
      }
    }

    // Calculate total words for normalization
    const totalWords = text.split(/\s+/).length;
    
    // Normalize scores by text length
    const normalizedPositive = totalWords > 0 ? positiveScore / totalWords : 0;
    const normalizedNegative = totalWords > 0 ? negativeScore / totalWords : 0;
    
    // Calculate final sentiment score (-1 to 1)
    return normalizedPositive - normalizedNegative;
  },

  /**
   * Extracts intent from text using regex patterns
   * @param {string} text - Text to analyze
   * @param {Object} intentPatterns - Map of intent names to regex patterns
   * @returns {string} - Extracted intent or "general" if none found
   */
  extractIntent(text, intentPatterns) {
    if (!text || !intentPatterns) return "general";
    
    const lowerText = text.toLowerCase();
    
    for (const [intent, pattern] of Object.entries(intentPatterns)) {
      if (pattern.test(lowerText)) {
        return intent;
      }
    }
    
    return "general";
  },

  /**
   * Fast entity extraction for large datasets
   * @param {string} text - Text to analyze
   * @param {Array<string>} commonEntities - Common entities to look for
   * @returns {Array<string>} - Extracted entities
   */
  extractEntities(text, commonEntities) {
    if (!text || !commonEntities) return [];
    
    const entities = [];
    const lowerText = text.toLowerCase();

    for (const entity of commonEntities) {
      if (lowerText.includes(entity.toLowerCase())) {
        entities.push(entity);
      }
    }

    return entities;
  },

  /**
   * Calculate entity similarity between two texts
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @param {Array<string>} commonEntities - Common entities to look for
   * @returns {number} - Similarity score between 0-1
   */
  calculateEntitySimilarity(text1, text2, commonEntities) {
    if (!text1 || !text2 || !commonEntities) return 0;
    if (text1 === text2) return 1;
    
    const entities1 = this.extractEntities(text1, commonEntities);
    const entities2 = this.extractEntities(text2, commonEntities);
    
    if (entities1.length === 0 && entities2.length === 0) return 0;
    
    // Use Set operations for faster intersection and union
    const set1 = new Set(entities1);
    const set2 = new Set(entities2);

    // Calculate Jaccard similarity (intersection / union)
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);

    return intersection.size / union.size;
  }
};