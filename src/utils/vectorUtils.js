/**
 * Vector utilities for BrainService
 * Replacement for modelUtils.js to work with FAISSstore and vector embeddings
 */

const fs = require('fs');
const path = require('path');
const textUtils = require('./textUtils');

module.exports = {
  /**
   * Extracts unique tags from dataset
   * @param {Array<Object>} dataset - Dataset to extract tags from
   * @returns {Array<string>} - Array of unique tags
   */
  extractUniqueTags(dataset) {
    if (!dataset || dataset.length === 0) {
      return [];
    }
    
    // Extract all tags from dataset
    const allTags = dataset
      .flatMap(entry => entry.tags || [])
      .filter(tag => tag); // Filter out empty tags
    
    // Get unique tags
    return [...new Set(allTags)];
  },

  /**
   * Prepares text for embedding by normalizing and cleaning
   * @param {string} text - Text to prepare
   * @returns {string} - Prepared text
   */
  prepareTextForEmbedding(text) {
    if (!text) return '';
    
    // Normalize text
    const normalizedText = textUtils.normalizeText(text);
    
    // Remove excessive whitespace
    return normalizedText.replace(/\s+/g, ' ').trim();
  },

  /**
   * Converts dataset entries to documents for vector store
   * @param {Array<Object>} dataset - Dataset to convert
   * @returns {Array<Object>} - Array of documents for vector store
   */
  datasetToDocuments(dataset) {
    if (!dataset || dataset.length === 0) {
      return [];
    }
    
    return dataset.map(entry => {
      // Combine question and answer for better context
      const content = `${entry.question}\n${entry.answer}`;
      const metadata = {
        question: entry.question,
        answer: entry.answer,
        tags: entry.tags || []
      };
      return { pageContent: content, metadata };
    });
  },

  /**
   * Gets vector store statistics
   * @param {Object} vectorStore - Vector store
   * @param {Array<Object>} dataset - Dataset
   * @returns {Object} - Vector store statistics
   */
  getVectorStoreStats(vectorStore, dataset) {
    if (!vectorStore || !dataset) {
      return { documentCount: 0, vectorDimensions: 0, isInitialized: false };
    }
    
    // Extract unique tags for statistics
    const uniqueTags = this.extractUniqueTags(dataset);
    
    // Count entries per tag
    const tagCounts = {};
    for (const entry of dataset) {
      if (Array.isArray(entry.tags)) {
        for (const tag of entry.tags) {
          tagCounts[tag] = (tagCounts[tag] || 0) + 1;
        }
      } else if (typeof entry.tags === 'string') {
        // Handle case where tags is a single string
        tagCounts[entry.tags] = (tagCounts[entry.tags] || 0) + 1;
      }
    }
    
    return {
      documentCount: dataset.length,
      uniqueTagsCount: uniqueTags.length,
      uniqueTags,
      tagDistribution: tagCounts,
      isInitialized: true,
      lastUpdated: new Date().toISOString()
    };
  }
};