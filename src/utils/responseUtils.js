/**
 * Response utilities for BrainService
 * Extracted from BrainService.js to improve maintainability
 */

const keywords = require('./keywords');
const textUtils = require('./textUtils');

module.exports = {
  /**
   * Generates intelligent fallback response based on intent
   * @param {string} question - User question
   * @param {string} intent - Detected intent
   * @returns {string} - Fallback response
   */
  generateIntelligentFallback(question, intent = 'general') {
    if (!question) return "Maaf, saya tidak mengerti pertanyaan Anda.";
    
    // Get fallback responses for the intent
    const responses = keywords.fallbackResponses[intent] || keywords.fallbackResponses.default;
    
    // Select a random response from the available options
    const randomIndex = Math.floor(Math.random() * responses.length);
    return responses[randomIndex];
  },

  /**
   * Generates contextual response based on relevant entries
   * @param {string} question - User question
   * @param {Array<Object>} relevantEntries - Relevant context entries
   * @param {Object} options - Response options
   * @returns {Object} - Response with answer and metadata
   */
  generateContextualResponse(question, relevantEntries, options = {}) {
    if (!question || !relevantEntries || relevantEntries.length === 0) {
      const intent = this.extractIntent(question);
      return {
        answer: this.generateIntelligentFallback(question, intent),
        confidence: 0,
        source: 'fallback',
        intent
      };
    }
    
    const {
      confidenceThreshold = 0.6,
      highConfidenceThreshold = 0.8,
      combineThreshold = 0.75
    } = options;
    
    // Get the best match
    const bestMatch = relevantEntries[0];
    const confidence = bestMatch.brainRelevance || bestMatch.relevanceScore || 0;
    
    // Extract intent for potential fallback
    const intent = this.extractIntent(question);
    
    // Handle based on confidence level
    if (confidence < confidenceThreshold) {
      // Low confidence - use fallback
      return {
        answer: this.generateIntelligentFallback(question, intent),
        confidence,
        source: 'fallback',
        intent
      };
    } else if (confidence >= highConfidenceThreshold) {
      // High confidence - use best match directly
      return {
        answer: bestMatch.answer,
        confidence,
        source: 'direct',
        intent,
        tags: bestMatch.tags
      };
    } else if (relevantEntries.length > 1 && 
               (relevantEntries[1].brainRelevance || relevantEntries[1].relevanceScore || 0) >= combineThreshold * confidence) {
      // Medium confidence with close second match - combine answers
      const combinedAnswer = this.combineAnswers(
        [bestMatch.answer, relevantEntries[1].answer],
        [confidence, relevantEntries[1].brainRelevance || relevantEntries[1].relevanceScore || 0]
      );
      
      return {
        answer: combinedAnswer,
        confidence,
        source: 'combined',
        intent,
        tags: [...new Set([...(bestMatch.tags || []), ...(relevantEntries[1].tags || [])])]
      };
    } else {
      // Medium confidence with no close second - use best match
      return {
        answer: bestMatch.answer,
        confidence,
        source: 'direct',
        intent,
        tags: bestMatch.tags
      };
    }
  },

  /**
   * Combines multiple answers into a coherent response
   * @param {Array<string>} answers - Array of answers to combine
   * @param {Array<number>} confidences - Confidence scores for each answer
   * @returns {string} - Combined answer
   */
  combineAnswers(answers, confidences) {
    if (!answers || answers.length === 0) {
      return "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan Anda.";
    }
    
    if (answers.length === 1) return answers[0];
    
    // Simple combination for two answers
    if (answers.length === 2) {
      // Check if answers are very similar
      const similarity = textUtils.calculateJaccardSimilarity(
        textUtils.normalizeText(answers[0]),
        textUtils.normalizeText(answers[1])
      );
      
      if (similarity > 0.7) {
        // If very similar, just return the highest confidence answer
        return confidences[0] >= confidences[1] ? answers[0] : answers[1];
      }
      
      // Otherwise combine them
      return `${answers[0]} ${this.getTransitionPhrase()} ${answers[1]}`;
    }
    
    // For more than two answers, combine with bullet points
    let combinedAnswer = "Berikut beberapa informasi yang mungkin membantu:\n";
    
    answers.forEach((answer, index) => {
      combinedAnswer += `\n- ${answer}`;
    });
    
    return combinedAnswer;
  },

  /**
   * Gets a random transition phrase for combining answers
   * @returns {string} - Transition phrase
   */
  getTransitionPhrase() {
    const phrases = [
      "Selain itu,",
      "Juga perlu diketahui bahwa",
      "Tambahan informasi:",
      "Perlu diingat juga bahwa",
      "Informasi lainnya,"
    ];
    
    const randomIndex = Math.floor(Math.random() * phrases.length);
    return phrases[randomIndex];
  },

  /**
   * Refines Indonesian response for better readability
   * @param {string} answer - Original answer
   * @param {Object} options - Refinement options
   * @returns {string} - Refined answer
   */
  refineIndonesianResponse(answer, options = {}) {
    if (!answer) return answer;
    
    const {
      intent = 'general',
      tags = [],
      improveFormatting = true,
      fixPunctuation = true,
      addGreeting = false,
      addClosing = false
    } = options;
    
    let refined = answer;
    
    // Apply category-specific refinements
    if (tags.includes('price_inquiry')) {
      refined = this.refinePriceResponse(refined);
    } else if (tags.includes('availability')) {
      refined = this.refineAvailabilityResponse(refined);
    } else if (tags.includes('payment')) {
      refined = this.refinePaymentResponse(refined);
    }
    
    // Improve formatting if needed
    if (improveFormatting) {
      refined = this.improveTextFormatting(refined);
    }
    
    // Fix punctuation if needed
    if (fixPunctuation) {
      refined = this.fixIndonesianPunctuation(refined);
    }
    
    // Add greeting if needed
    if (addGreeting) {
      refined = this.addIndonesianGreeting() + " " + refined;
    }
    
    // Add closing if needed
    if (addClosing) {
      refined = refined + " " + this.addIndonesianClosing();
    }
    
    return refined;
  },

  /**
   * Refines price-related responses
   * @param {string} text - Original text
   * @returns {string} - Refined text
   */
  refinePriceResponse(text) {
    if (!text) return text;
    
    // Ensure price information is clearly formatted
    let refined = text;
    
    // Format price numbers with dots for thousands
    refined = refined.replace(/(Rp\.?\s*)(\d+)(\s*ribu|rb|k)/gi, (match, prefix, number, suffix) => {
      const formattedNumber = parseInt(number) * 1000;
      return `${prefix}${formattedNumber.toLocaleString('id-ID')}`;
    });
    
    // Format price ranges
    refined = refined.replace(/(Rp\.?\s*)(\d+)(\s*-\s*)(\d+)/gi, (match, prefix, num1, separator, num2) => {
      const formattedNum1 = parseInt(num1).toLocaleString('id-ID');
      const formattedNum2 = parseInt(num2).toLocaleString('id-ID');
      return `${prefix}${formattedNum1}${separator}${prefix}${formattedNum2}`;
    });
    
    return refined;
  },

  /**
   * Refines availability-related responses
   * @param {string} text - Original text
   * @returns {string} - Refined text
   */
  refineAvailabilityResponse(text) {
    if (!text) return text;
    
    // Make availability status more prominent
    let refined = text;
    
    // Highlight positive availability
    if (/\b(tersedia|ada|ready|available|stock|stok)\b/i.test(refined)) {
      refined = refined.replace(
        /(\b(tersedia|ada|ready|available|stock|stok)\b)/gi,
        match => `*${match}*`
      );
    }
    
    // Highlight negative availability
    if (/\b(tidak tersedia|habis|kosong|out of stock|sold out)\b/i.test(refined)) {
      refined = refined.replace(
        /(\b(tidak tersedia|habis|kosong|out of stock|sold out)\b)/gi,
        match => `*${match}*`
      );
    }
    
    return refined;
  },

  /**
   * Refines payment-related responses
   * @param {string} text - Original text
   * @returns {string} - Refined text
   */
  refinePaymentResponse(text) {
    if (!text) return text;
    
    // Format payment method information
    let refined = text;
    
    // Convert payment method lists to bullet points if there are multiple methods
    if ((
      /\b(transfer|bank|bca|mandiri|bni|bri|dana|ovo|gopay|shopeepay|linkaja|qris)\b.*\b(transfer|bank|bca|mandiri|bni|bri|dana|ovo|gopay|shopeepay|linkaja|qris)\b/i.test(refined)
    ) && !refined.includes('\n-')) {
      
      // Extract the part that likely contains payment methods
      const parts = refined.split(/(?:\.|\n|:)/);
      let methodsPart = '';
      
      for (const part of parts) {
        if (/\b(transfer|bank|bca|mandiri|bni|bri|dana|ovo|gopay|shopeepay|linkaja|qris)\b.*\b(transfer|bank|bca|mandiri|bni|bri|dana|ovo|gopay|shopeepay|linkaja|qris)\b/i.test(part)) {
          methodsPart = part;
          break;
        }
      }
      
      if (methodsPart) {
        // Convert to bullet points
        const methods = methodsPart.split(/(?:,|dan|atau|and|or)/);
        let bulletPoints = "\nMetode pembayaran yang tersedia:\n";
        
        methods.forEach(method => {
          const trimmed = method.trim();
          if (trimmed && /\b(transfer|bank|bca|mandiri|bni|bri|dana|ovo|gopay|shopeepay|linkaja|qris)\b/i.test(trimmed)) {
            bulletPoints += `\n- ${trimmed}`;
          }
        });
        
        refined = refined.replace(methodsPart, bulletPoints);
      }
    }
    
    return refined;
  },

  /**
   * Improves text formatting for better readability
   * @param {string} text - Original text
   * @returns {string} - Formatted text
   */
  improveTextFormatting(text) {
    if (!text) return text;
    
    let formatted = text;
    
    // Add line breaks for better readability if text is long
    if (formatted.length > 150 && !formatted.includes('\n')) {
      // Split into sentences
      formatted = formatted.replace(/\.\s+/g, '.\n');
    }
    
    // Convert numbered lists to proper format
    formatted = formatted.replace(/\b(\d+)\.(\s+)/g, '\n$1.$2');
    
    // Ensure consistent spacing
    formatted = formatted.replace(/\s+/g, ' ');
    formatted = formatted.replace(/\n\s+/g, '\n');
    formatted = formatted.replace(/\s+\n/g, '\n');
    
    return formatted.trim();
  },

  /**
   * Fixes Indonesian punctuation issues
   * @param {string} text - Original text
   * @returns {string} - Text with fixed punctuation
   */
  fixIndonesianPunctuation(text) {
    if (!text) return text;
    
    let fixed = text;
    
    // Ensure space after punctuation
    fixed = fixed.replace(/([.,:;!?])([^\s\n])/g, '$1 $2');
    
    // Fix multiple punctuation
    fixed = fixed.replace(/([.,:;!?]){2,}/g, '$1');
    
    // Ensure sentences start with capital letter
    fixed = fixed.replace(/([.!?]\s+)([a-z])/g, (match, p1, p2) => p1 + p2.toUpperCase());
    
    // Ensure first letter is capital
    if (/^[a-z]/.test(fixed)) {
      fixed = fixed.charAt(0).toUpperCase() + fixed.slice(1);
    }
    
    // Ensure text ends with punctuation
    if (!/[.!?]$/.test(fixed)) {
      fixed += '.';
    }
    
    return fixed;
  },

  /**
   * Adds an Indonesian greeting
   * @returns {string} - Random greeting
   */
  addIndonesianGreeting() {
    const greetings = [
      "Halo",
      "Hai",
      "Selamat datang",
      "Terima kasih atas pertanyaannya"
    ];
    
    const randomIndex = Math.floor(Math.random() * greetings.length);
    return greetings[randomIndex];
  },

  /**
   * Adds an Indonesian closing
   * @returns {string} - Random closing
   */
  addIndonesianClosing() {
    const closings = [
      "Semoga membantu",
      "Ada yang bisa saya bantu lagi?",
      "Jika ada pertanyaan lain, silakan tanyakan",
      "Terima kasih"
    ];
    
    const randomIndex = Math.floor(Math.random() * closings.length);
    return closings[randomIndex];
  },

  /**
   * Extracts intent from text
   * @param {string} text - Text to analyze
   * @returns {string} - Extracted intent
   */
  extractIntent(text) {
    if (!text) return "general";
    
    const lowerText = text.toLowerCase();
    
    for (const [intent, pattern] of Object.entries(keywords.intentPatterns)) {
      if (pattern.test(lowerText)) {
        return intent;
      }
    }
    
    return "general";
  }
};