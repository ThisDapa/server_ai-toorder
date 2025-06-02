const brain = require('brain.js');
const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');

class BrainService {
  constructor() {
    this.brainNetwork = null;
    this.dataset = null;
    this.isInitialized = false;
    this.modelPath = path.resolve("./models/brain-model.json");

    // Bind methods to ensure correct 'this' context
    this.calculateContextualRelevance = this.calculateContextualRelevance.bind(this);
    this.processContext = this.processContext.bind(this);
    this.findAnswer = this.findAnswer.bind(this);
    this.trainBrainNetwork = this.trainBrainNetwork.bind(this);
    this.generateAugmentedDataset = this.generateAugmentedDataset.bind(this);
    this.generateLimitedAugmentedDataset = this.generateLimitedAugmentedDataset.bind(this);
    this.calculateSimilarity = this.calculateSimilarity.bind(this);
    this.calculateSemanticSimilarity = this.calculateSemanticSimilarity.bind(this);
    this.calculateContextSimilarity = this.calculateContextSimilarity.bind(this);
    this.calculateSentiment = this.calculateSentiment.bind(this);
    this.normalizeText = this.normalizeText.bind(this);
    this.indonesianStemmer = this.indonesianStemmer.bind(this);
    this.textToEnhancedVector = this.textToEnhancedVector.bind(this);
    this.extractUniqueTags = this.extractUniqueTags.bind(this);
    this.getPredictedTags = this.getPredictedTags.bind(this);
    this.retrainModel = this.retrainModel.bind(this);
    this.trainWithDataset = this.trainWithDataset.bind(this);
    this.exportModel = this.exportModel.bind(this);
  }

  async init() {
    try {
      this.brainNetwork = new brain.NeuralNetwork({
        hiddenLayers: [24, 16, 8],
        activation: "sigmoid",
        learningRate: 0.01,
        momentum: 0.8,
        binaryThresh: 0.5,
        errorThresh: 0.003,
        dropout: 0.1,
        batchSize: 32,
      });

      await this.loadDataset();

      const modelLoaded = await this.loadTrainedModel();

      if (!modelLoaded && this.dataset && this.dataset.length > 0) {
        await this.trainBrainNetwork();
        await this.saveTrainedModel();
      }

      this.isInitialized = true;
      logger.info("BrainService initialized successfully");
    } catch (error) {
      logger.error(`Failed to initialize BrainService: ${error.message}`);
      throw error;
    }
  }

  /**
   * Calculate contextual relevance between a question and a dataset entry
   * @param {string} question - The input question string
   * @param {Object} entry - The dataset entry object containing at least a question field
   * @returns {number} - A numeric relevance score between 0 and 1
   */
  calculateContextualRelevance(question, entry) {
    try {
      if (!entry || !entry.question) {
        return 0;
      }
      return this.calculateContextSimilarity(question, entry.question);
    } catch (error) {
      logger.error(`Error in calculateContextualRelevance: ${error.message}`);
      return 0;
    }
  }

  stemText(text) {
    if (!text) return "";

    return text
      .split(" ")
      .map((word) => this.indonesianStemmer(word))
      .join(" ");
  }
  async loadDataset() {
    try {
      const datasetPath = process.env.DATASET_PATH || "./data/dataset.json";
      const fullPath = path.resolve(datasetPath);

      try {
        await fs.access(fullPath);
        const data = await fs.readFile(fullPath, "utf8");
        this.dataset = JSON.parse(data);
        logger.info(`Dataset loaded: ${this.dataset.length} entries`);
      } catch (fileError) {
        logger.warn(
          `Dataset file not found at ${fullPath}, using default dataset`
        );
        this.dataset = this.getDefaultDataset();
      }
    } catch (error) {
      logger.error(`Error loading dataset: ${error.message}`);
      this.dataset = this.getDefaultDataset();
    }
  }

  getDefaultDataset() {
    return [
      {
        question: "Hi, good morning!",
        answer: "Good morning! How can I assist you today?",
        tags: ["greeting"],
      },
      {
        question: "How much does this product cost?",
        answer:
          "I'll help you check the price. Could you please specify which product you're interested in?",
        tags: ["price_inquiry"],
      },
      {
        question: "Do you have this item in stock?",
        answer:
          "I'll check the availability for you. Which specific item are you looking for?",
        tags: ["available"],
      },
      {
        question: "Can you help me with my order?",
        answer:
          "Of course! I'm here to help. What kind of assistance do you need with your order?",
        tags: ["help"],
      },
    ];
  }

  async loadTrainedModel() {
    try {
      await fs.access(this.modelPath);
      const modelData = await fs.readFile(this.modelPath, "utf8");
      const modelJson = JSON.parse(modelData);

      this.brainNetwork.fromJSON(modelJson);
      logger.info("Pre-trained Brain.js model loaded successfully");
      return true;
    } catch (error) {
      logger.info("No pre-trained model found, will train new model");
      return false;
    }
  }

  async saveTrainedModel() {
    try {
      const modelsDir = path.dirname(this.modelPath);
      await fs.mkdir(modelsDir, { recursive: true });

      const modelJson = this.brainNetwork.toJSON();
      await fs.writeFile(this.modelPath, JSON.stringify(modelJson, null, 2));
      logger.info("Brain.js model saved successfully");
    } catch (error) {
      logger.error(`Error saving trained model: ${error.message}`);
    }
  }

  generateAugmentedDataset() {
    const augmentedDataset = [...this.dataset];
    const originalSize = this.dataset.length;

    const samplesToAugment = Math.min(originalSize, 300);
    const sampledData = this.dataset
      .slice()
      .sort(() => 0.5 - Math.random())
      .slice(0, samplesToAugment);

    for (const item of sampledData) {
      const typoVariations = this.generateTypoVariations(item.question);

      for (const variation of typoVariations) {
        augmentedDataset.push({
          question: variation,
          answer: item.answer,
          tags: item.tags,
        });
      }

      const wordOrderVariations = this.generateWordOrderVariations(
        item.question
      );

      for (const variation of wordOrderVariations) {
        augmentedDataset.push({
          question: variation,
          answer: item.answer,
          tags: item.tags,
        });
      }
    }

    logger.info(
      `Generated augmented dataset with ${augmentedDataset.length} entries (original: ${originalSize})`
    );
    return augmentedDataset;
  }

  generateLimitedAugmentedDataset() {
    const startTime = Date.now();
    logger.info("Generating limited augmented dataset for large dataset...");

    const augmentedDataset = [...this.dataset];
    const originalSize = this.dataset.length;

    const highValueCategories = [
      "greeting",
      "price_inquiry",
      "payment",
      "help",
      "technical",
      "refund",
      "account",
      "subscription",
      "problem",
    ];

    // Filter dataset to only include high-value categories
    const highValueItems = this.dataset.filter((item) => {
      const tags = Array.isArray(item.tags) ? item.tags : [item.tags];
      return tags.some((tag) => highValueCategories.includes(tag));
    });

    // Calculate how many samples to augment based on dataset size
    // Larger datasets get proportionally less augmentation
    const augmentationRatio = Math.max(0.01, 0.1 - originalSize / 100000);
    const maxSamplesToAugment = Math.min(
      Math.ceil(originalSize * augmentationRatio), // Percentage of dataset
      200 // Hard cap for very large datasets
    );

    // Sample from high-value items first, then from general dataset if needed
    let samplesToAugment = Math.min(highValueItems.length, maxSamplesToAugment);
    let sampledData = highValueItems
      .slice()
      .sort(() => 0.5 - Math.random())
      .slice(0, samplesToAugment);

    // If we need more samples, get them from the general dataset
    if (samplesToAugment < maxSamplesToAugment) {
      const remainingSamples = maxSamplesToAugment - samplesToAugment;
      const generalSamples = this.dataset
        .filter((item) => !sampledData.includes(item))
        .slice()
        .sort(() => 0.5 - Math.random())
        .slice(0, remainingSamples);

      sampledData = [...sampledData, ...generalSamples];
    }

    logger.info(
      `Selected ${sampledData.length} samples for limited augmentation`
    );

    // Process in smaller batches to avoid memory pressure
    const batchSize = 20;
    for (let i = 0; i < sampledData.length; i += batchSize) {
      const batch = sampledData.slice(i, i + batchSize);

      for (const item of batch) {
        // For large datasets, only create 1-2 variations per item instead of all possible variations
        // Alternate between typo and word order variations to get good coverage
        if (i % 2 === 0) {
          // Create just 1 typo variation
          const typoVariations = this.generateTypoVariations(
            item.question
          ).slice(0, 1);

          for (const variation of typoVariations) {
            augmentedDataset.push({
              question: variation,
              answer: item.answer,
              tags: item.tags,
            });
          }
        } else {
          // Create just 1 word order variation
          const wordOrderVariations = this.generateWordOrderVariations(
            item.question
          ).slice(0, 1);

          for (const variation of wordOrderVariations) {
            augmentedDataset.push({
              question: variation,
              answer: item.answer,
              tags: item.tags,
            });
          }
        }
      }

      // Log progress for large batches
      if (i > 0 && i % 50 === 0) {
        logger.info(
          `Processed ${i}/${sampledData.length} samples for augmentation`
        );
      }
    }

    const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);
    logger.info(
      `Generated limited augmented dataset with ${augmentedDataset.length} entries (original: ${originalSize}) in ${processingTime}s`
    );
    return augmentedDataset;
  }

  generateTypoVariations(text) {
    const variations = [];
    const words = text.split(" ");

    const commonTypos = {
      a: ["s", "q", "z"],
      b: ["v", "g", "h", "n"],
      c: ["x", "v", "d"],
      d: ["s", "f", "e", "r"],
      e: ["w", "r", "d"],
      g: ["f", "h", "t", "y"],
      h: ["g", "j", "y", "u"],
      i: ["u", "o", "k", "j"],
      j: ["h", "k", "u", "i"],
      k: ["j", "l", "i", "o"],
      l: ["k", "p", "o"],
      m: ["n", "j", "k"],
      n: ["m", "b", "h", "j"],
      o: ["i", "p", "l", "k"],
      p: ["o", "l"],
      r: ["e", "t", "d", "f"],
      s: ["a", "d", "w", "e"],
      t: ["r", "y", "f", "g"],
      u: ["y", "i", "h", "j"],
      v: ["c", "b", "f", "g"],
      w: ["q", "e", "a", "s"],
      x: ["z", "c", "s", "d"],
      y: ["t", "u", "g", "h"],
      z: ["a", "x", "s"],
    };

    // Generate up to 3 variations
    for (let i = 0; i < Math.min(3, words.length); i++) {
      // Select a random word to modify
      const wordIndex = Math.floor(Math.random() * words.length);
      const word = words[wordIndex];

      if (word.length < 2) continue;

      // 1. Character substitution (typo)
      const charIndex = Math.floor(Math.random() * word.length);
      const char = word[charIndex].toLowerCase();

      if (commonTypos[char]) {
        const typoChars = commonTypos[char];
        const typoChar =
          typoChars[Math.floor(Math.random() * typoChars.length)];

        const newWords = [...words];
        newWords[wordIndex] =
          word.substring(0, charIndex) +
          typoChar +
          word.substring(charIndex + 1);
        variations.push(newWords.join(" "));
      }

      // 2. Character omission (missing letter)
      if (word.length > 3) {
        const omitIndex = Math.floor(Math.random() * word.length);

        const newWords = [...words];
        newWords[wordIndex] =
          word.substring(0, omitIndex) + word.substring(omitIndex + 1);
        variations.push(newWords.join(" "));
      }

      // 3. Character duplication (repeated letter)
      const dupIndex = Math.floor(Math.random() * word.length);
      const dupChar = word[dupIndex];

      const newWords = [...words];
      newWords[wordIndex] =
        word.substring(0, dupIndex) +
        dupChar +
        dupChar +
        word.substring(dupIndex + 1);
      variations.push(newWords.join(" "));
    }

    return variations;
  }

  generateWordOrderVariations(text) {
    const variations = [];
    const words = text.split(" ");

    // Only generate variations for sentences with 3+ words
    if (words.length < 3) return variations;

    // 1. Swap two adjacent words
    for (let i = 0; i < Math.min(2, words.length - 1); i++) {
      const swapIndex = Math.floor(Math.random() * (words.length - 1));

      const newWords = [...words];
      const temp = newWords[swapIndex];
      newWords[swapIndex] = newWords[swapIndex + 1];
      newWords[swapIndex + 1] = temp;

      variations.push(newWords.join(" "));
    }

    // 2. Move one word to a different position
    if (words.length >= 4) {
      const wordIndex = Math.floor(Math.random() * words.length);
      const targetIndex = (wordIndex + 2) % words.length;

      const newWords = [...words];
      const wordToMove = newWords.splice(wordIndex, 1)[0];
      newWords.splice(targetIndex, 0, wordToMove);

      variations.push(newWords.join(" "));
    }

    return variations;
  }

  extractUniqueTags() {
    try {
      // Create a Set to store unique tags
      const uniqueTagsSet = new Set();

      // Iterate through dataset and collect all unique tags
      this.dataset.forEach((item) => {
        if (Array.isArray(item.tags)) {
          // If tags is an array, add each tag
          item.tags.forEach((tag) => uniqueTagsSet.add(tag));
        } else if (typeof item.tags === "string") {
          // If tags is a string, add it directly
          uniqueTagsSet.add(item.tags);
        }
      });

      // Convert Set to Array and return
      const uniqueTags = Array.from(uniqueTagsSet);
      logger.info(`Extracted ${uniqueTags.length} unique tags from dataset`);
      return uniqueTags;
    } catch (error) {
      logger.error(`Error extracting unique tags: ${error.message}`);
      // Return default tags as fallback
      return [
        "greeting",
        "price_inquiry",
        "available",
        "help",
        "payment",
        "goodbye",
        "unknown",
      ];
    }
  }

  async trainBrainNetwork() {
    try {
      logger.info(
        "Training Brain.js network with enhanced parameters for 50k dataset..."
      );

      // Record start time for training duration calculation
      const startTime = Date.now();

      // Generate augmented training data with variations to handle typos and variations
      // For large datasets (>10k), limit augmentation to avoid memory issues
      const shouldLimitAugmentation = this.dataset.length > 10000;
      const augmentedDataset = shouldLimitAugmentation
        ? this.generateLimitedAugmentedDataset()
        : this.generateAugmentedDataset();

      // Extract unique tags from dataset
      const uniqueTags = this.extractUniqueTags();
      logger.info(
        `Training with ${uniqueTags.length} unique tags: ${uniqueTags.join(
          ", "
        )}`
      );

      // Calculate estimated training time based on dataset size
      const estimatedTimePerSample = 0.0015; // seconds per sample for Ryzen 3 7000 series
      const estimatedTrainingTime = Math.round(
        augmentedDataset.length * estimatedTimePerSample
      );
      const estimatedMinutes = Math.floor(estimatedTrainingTime / 60);
      const estimatedSeconds = estimatedTrainingTime % 60;

      logger.info(
        `Estimated training time for ${augmentedDataset.length} samples: ${estimatedMinutes} minutes ${estimatedSeconds} seconds`
      );

      // Prepare training data with optimized vector generation
      logger.info("Preparing training data vectors...");
      const trainingData = [];

      // Process in batches to avoid memory pressure with large datasets
      const batchSize = 1000;
      for (let i = 0; i < augmentedDataset.length; i += batchSize) {
        const batch = augmentedDataset.slice(i, i + batchSize);

        const batchTrainingData = batch.map((item) => {
          const tags = Array.isArray(item.tags) ? item.tags : [item.tags];

          // Create output object dynamically based on unique tags
          const output = {};
          uniqueTags.forEach((tag) => {
            output[`tag_${tag}`] = tags.includes(tag) ? 1 : 0;
          });

          return {
            input: this.textToEnhancedVector(item.question),
            output: output,
          };
        });

        trainingData.push(...batchTrainingData);

        // Log progress for large datasets
        if (augmentedDataset.length > 5000 && (i + batchSize) % 5000 === 0) {
          logger.info(
            `Prepared ${Math.min(i + batchSize, augmentedDataset.length)}/${
              augmentedDataset.length
            } training vectors`
          );
        }
      }

      // Optimized training parameters for Ryzen 3 7000 series with 8GB RAM and 50k dataset
      logger.info("Starting neural network training...");

      // Track progress for time estimation updates
      let lastIterationTime = Date.now();
      let iterationsCompleted = 0;
      let timePerIteration = 0;

      const result = this.brainNetwork.train(trainingData, {
        iterations: 5000, // Sigmoid biasanya butuh lebih banyak iterasi
        errorThresh: 0.003,
        log: true,
        logPeriod: 100,
        learningRate: 0.01,
        momentum: 0.8,
        callback: (data) => {
          // Hitung waktu per iterasi
          const currentTime = Date.now();
          const iterationTime = currentTime - lastIterationTime;
          lastIterationTime = currentTime;

          // Update rata-rata waktu per iterasi
          timePerIteration =
            iterationsCompleted === 0
              ? iterationTime
              : timePerIteration * 0.7 + iterationTime * 0.3;

          iterationsCompleted = data.iterations;

          // Hitung estimasi waktu sisa dalam menit
          const remainingIterations = 3000 - data.iterations; // Ganti 3000 sesuai max iterations
          const remainingTimeMs = timePerIteration * remainingIterations;
          const remainingMinutes = Math.floor(remainingTimeMs / 60000);
          const remainingSeconds = Math.floor((remainingTimeMs % 60000) / 1000);

          // Early stopping
          if (data.iterations > 500 && data.error < 0.008) {
            logger.info(
              `Early stopping at iteration ${
                data.iterations
              } with error ${data.error.toFixed(6)}`
            );
            logger.info(
              `Training completed in ${Math.round(
                (currentTime - startTime) / 1000
              )} seconds`
            );
            return true;
          }

          // Log estimasi waktu selesai
          logger.info(
            `Training progress - Iterations: ${
              data.iterations
            }, Error: ${data.error.toFixed(
              6
            )}, Estimasi selesai: ~${remainingMinutes}m ${remainingSeconds}s`
          );
        },
      });

      // Calculate actual training time
      const trainingTime = Math.round((Date.now() - startTime) / 1000);
      const trainingMinutes = Math.floor(trainingTime / 60);
      const trainingSeconds = trainingTime % 60;

      logger.info(
        `Brain.js network trained successfully. Final error: ${result.error.toFixed(
          6
        )}`
      );
      logger.info(
        `Training completed in ${trainingMinutes}m ${trainingSeconds}s with ${augmentedDataset.length} samples (original: ${this.dataset.length})`
      );

      // Save training stats for future reference
      this.trainingStats = {
        datasetSize: this.dataset.length,
        augmentedSize: augmentedDataset.length,
        trainingTime: trainingTime,
        iterations: result.iterations,
        finalError: result.error,
        timestamp: new Date().toISOString(),
      };

      return result;
    } catch (error) {
      logger.error(`Error training Brain.js network: ${error.message}`);
      throw error;
    }
  }

  /**
   * Convert text to enhanced vector for Brain.js with more features
   * Optimized for large datasets (50k+) with improved Indonesian language support
   * @param {string} text - Input text to vectorize
   * @param {string} category - Optional category for additional features
   * @returns {Object} - Feature vector for neural network
   */
  textToEnhancedVector(text, category) {
    category = category || "";

    // Cache for performance optimization with large datasets
    const cacheKey = `vector_${text}_${category}`;
    if (this._vectorCache && this._vectorCache[cacheKey]) {
      return this._vectorCache[cacheKey];
    }

    // Initialize vector cache if not exists (for session-level caching)
    if (!this._vectorCache) {
      this._vectorCache = {};
      this._vectorCacheSize = 0;
      this._vectorCacheMaxSize = 10000; // Limit cache size for memory management
    }

    // Normalize and clean text
    const normalizedText = this.normalizeText(text);
    const words = normalizedText.split(" ");
    const vector = {};

    // Detect language for optimized feature extraction
    const isIndonesian = this.isIndonesianText(normalizedText);
    const isEnglish = !isIndonesian && this.isEnglishText(normalizedText);
    vector["is_indonesian"] = isIndonesian ? 1 : 0;
    vector["is_english"] = isEnglish ? 1 : 0;

    // Apply stemming for Indonesian text to improve matching
    const stemmedWords = isIndonesian
      ? words.map((word) => this.indonesianStemmer(word))
      : words;

    // Word-based features with optimized encoding
    // Limit to first 10 words for large datasets to reduce vector size
    const wordLimit = 10;
    stemmedWords.slice(0, wordLimit).forEach((word, index) => {
      if (word.length < 2) {
        vector[`word_${index}`] = 0; // Skip very short words
        return;
      }

      vector[`word_${index}`] = this.wordToNumeric(word);

      // Add character n-gram features for better typo handling
      // Only for important words (first 5) to reduce vector size
      if (word.length >= 3 && index < 5) {
        // Character trigrams for common word parts
        for (let i = 0; i < Math.min(word.length - 2, 3); i++) {
          // Limit trigrams per word
          const trigram = word.substring(i, i + 3);
          vector[`trigram_${index}_${i}`] = this.wordToNumeric(trigram);
        }
      }
    });

    // Pad remaining word slots
    for (let i = stemmedWords.length; i < wordLimit; i++) {
      vector[`word_${i}`] = 0;
    }

    // Text length features - important for distinguishing question types
    vector["text_length"] = Math.min(text.length / 100, 1); // Normalized length
    vector["word_count"] = Math.min(words.length / 20, 1); // Normalized word count
    vector["avg_word_length"] =
      words.length > 0
        ? Math.min(
            words.reduce((sum, word) => sum + word.length, 0) /
              words.length /
              10,
            1
          )
        : 0;

    // Question complexity indicator (useful for determining response style)
    vector["complexity"] = this.calculateTextComplexity(normalizedText);

    // Question type features - Enhanced for Indonesian language
    // Use direct matching for better performance with large datasets
    const questionTypes = {
      is_apa: ["apa", "apakah"],
      is_bagaimana: ["bagaimana", "gimana", "cara", "caranya", "bagaimanakah"],
      is_kenapa: ["kenapa", "mengapa", "alasan", "alasannya", "penyebab"],
      is_kapan: ["kapan", "jam", "waktu", "jadwal", "tanggal", "hari"],
      is_dimana: ["dimana", "mana", "lokasi", "tempat", "alamat"],
      is_siapa: ["siapa", "nama", "kontak", "pengguna", "user"],
      is_berapa: [
        "berapa",
        "harga",
        "biaya",
        "tarif",
        "ongkos",
        "bayar",
        "total",
      ],
    };

    // Optimize question type detection for large datasets
    Object.entries(questionTypes).forEach(([key, keywords]) => {
      // For very large datasets, use direct includes for better performance
      if (this.dataset && this.dataset.length > 20000) {
        vector[key] = keywords.some((keyword) =>
          normalizedText.includes(keyword)
        )
          ? 1
          : 0;
      } else {
        // For smaller datasets, use fuzzy matching for better accuracy
        vector[key] = this.fuzzyMatch(normalizedText, keywords) ? 1 : 0;
      }
    });

    // Customer service intent features - Optimized for Indonesian language
    const intentFeatures = {
      is_help: [
        "bantuan",
        "bantu",
        "tolong",
        "help",
        "assist",
        "panduan",
        "tutorial",
        "petunjuk",
      ],
      is_price: [
        "harga",
        "berapa",
        "biaya",
        "tarif",
        "price",
        "cost",
        "bayar",
        "rupiah",
        "idr",
        "rp",
      ],
      is_available: [
        "ada",
        "tersedia",
        "ready",
        "stock",
        "stok",
        "available",
        "ketersediaan",
        "sedia",
      ],
      is_payment: [
        "bayar",
        "pembayaran",
        "transfer",
        "dana",
        "ovo",
        "gopay",
        "pay",
        "payment",
        "transaksi",
        "kartu",
        "kredit",
        "debit",
        "virtual",
        "account",
        "va",
        "qris",
      ],
      is_greeting: [
        "hai",
        "halo",
        "hi",
        "hello",
        "selamat",
        "pagi",
        "siang",
        "sore",
        "malam",
        "assalamualaikum",
        "shalom",
      ],
      is_goodbye: [
        "terima kasih",
        "makasih",
        "thanks",
        "bye",
        "goodbye",
        "sampai jumpa",
        "selamat tinggal",
        "thx",
        "tq",
      ],
    };

    // Apply intent detection with optimized matching
    Object.entries(intentFeatures).forEach(([key, keywords]) => {
      // For very large datasets, use direct includes for better performance
      if (this.dataset && this.dataset.length > 20000) {
        vector[key] = keywords.some((keyword) =>
          normalizedText.includes(keyword)
        )
          ? 1
          : 0;
      } else {
        // For smaller datasets, use fuzzy matching for better accuracy
        vector[key] = this.fuzzyMatch(normalizedText, keywords) ? 1 : 0;
      }
    });

    // Service-specific features - Enhanced with more Indonesian terms
    const serviceFeatures = {
      is_refund: [
        "refund",
        "pengembalian",
        "uang kembali",
        "batal",
        "cancel",
        "garansi",
        "kembalikan",
        "retur",
        "return",
      ],
      is_technical: [
        "teknis",
        "spesifikasi",
        "requirement",
        "sistem",
        "device",
        "kompatibel",
        "support",
        "error",
        "bug",
        "crash",
        "tidak bisa",
        "gagal",
        "masalah",
      ],
      is_account: [
        "akun",
        "account",
        "profil",
        "profile",
        "login",
        "masuk",
        "daftar",
        "register",
        "password",
        "kata sandi",
        "username",
        "email",
        "verifikasi",
      ],
      is_subscription: [
        "langganan",
        "subscription",
        "paket",
        "package",
        "plan",
        "premium",
        "basic",
        "upgrade",
        "downgrade",
        "perpanjang",
        "extend",
        "renew",
        "cancel",
        "batal",
      ],
      is_referral: [
        "referral",
        "ajak teman",
        "bonus",
        "loyalty",
        "poin",
        "reward",
        "cashback",
        "komisi",
        "affiliate",
        "afiliasi",
        "kode promo",
        "promo code",
        "diskon",
        "discount",
      ],
      is_emerging: [
        "layanan baru",
        "fitur baru",
        "update",
        "terbaru",
        "coming soon",
        "segera hadir",
        "rilis",
        "release",
        "versi baru",
        "new version",
      ],
    };

    // Apply service feature detection
    Object.entries(serviceFeatures).forEach(([key, keywords]) => {
      // For very large datasets, use direct includes for better performance
      if (this.dataset && this.dataset.length > 20000) {
        vector[key] = keywords.some((keyword) =>
          normalizedText.includes(keyword)
        )
          ? 1
          : 0;
      } else {
        // For smaller datasets, use fuzzy matching for better accuracy
        vector[key] = this.fuzzyMatch(normalizedText, keywords) ? 1 : 0;
      }
    });

    // Add sentiment features - important for response tone
    const sentimentScore = this.calculateSentiment(normalizedText);
    vector["sentiment"] = sentimentScore;
    vector["is_positive"] = sentimentScore > 0.6 ? 1 : 0;
    vector["is_negative"] = sentimentScore < 0.4 ? 1 : 0;
    vector["is_neutral"] =
      sentimentScore >= 0.4 && sentimentScore <= 0.6 ? 1 : 0;

    // Add category-specific features if provided
    if (category) {
      vector[`category_${category}`] = 1;
    }

    // Add to cache for future reuse
    if (this._vectorCacheSize < this._vectorCacheMaxSize) {
      this._vectorCache[cacheKey] = vector;
      this._vectorCacheSize++;
    }

    return vector;
  }

  /**
   * Calculate text complexity score based on various factors
   * Used to determine appropriate response style and detail level
   * @param {string} text - Input text
   * @returns {number} - Complexity score between 0-1
   */
  calculateTextComplexity(text) {
    if (!text) return 0;

    const words = text.split(" ");
    const wordCount = words.length;

    // Calculate various complexity indicators
    const avgWordLength =
      words.reduce((sum, word) => sum + word.length, 0) /
      Math.max(wordCount, 1);
    const longWordCount = words.filter((word) => word.length > 6).length;
    const longWordRatio = longWordCount / Math.max(wordCount, 1);
    const questionMarkCount = (text.match(/\?/g) || []).length;
    const hasMultipleQuestions = questionMarkCount > 1;

    // Check for complex sentence structures
    const hasConjunctions =
      /\b(dan|atau|tetapi|namun|karena|sebab|jika|kalau|maka|sehingga|agar|supaya)\b/i.test(
        text
      );
    const hasSubordinateClauses =
      /\b(yang|dimana|ketika|saat|selama|setelah|sebelum|sejak)\b/i.test(text);

    // Calculate complexity score (0-1)
    let complexityScore = 0;
    complexityScore += Math.min(wordCount / 20, 0.3); // Up to 0.3 for length
    complexityScore += Math.min(avgWordLength / 10, 0.2); // Up to 0.2 for avg word length
    complexityScore += Math.min(longWordRatio, 0.15); // Up to 0.15 for long words
    complexityScore += hasMultipleQuestions ? 0.15 : 0; // 0.15 for multiple questions
    complexityScore += hasConjunctions ? 0.1 : 0; // 0.1 for conjunctions
    complexityScore += hasSubordinateClauses ? 0.1 : 0; // 0.1 for subordinate clauses

    return Math.min(complexityScore, 1); // Cap at 1
  }

  /**
   * Normalize text by removing punctuation, extra spaces, and converting to lowercase
   * @param {string} text - The text to normalize
   * @returns {string} - Normalized text
   */
  normalizeText(text) {
    if (!text) return "";
    // Convert to lowercase, replace punctuation with spaces, normalize multiple spaces
    let normalized = text
      .toLowerCase()
      .replace(/[^\w\s\u00C0-\u017F]/g, " ") // Keep Indonesian accented characters
      .replace(/\s+/g, " ")
      .trim();

    // Replace common Indonesian typos and abbreviations
    const commonReplacements = {
      yg: "yang",
      dgn: "dengan",
      utk: "untuk",
      tdk: "tidak",
      tsb: "tersebut",
      krn: "karena",
      spy: "supaya",
      skrg: "sekarang",
      blm: "belum",
      sdh: "sudah",
      bs: "bisa",
      sy: "saya",
      gk: "tidak",
      ga: "tidak",
      gak: "tidak",
      ngga: "tidak",
      nggak: "tidak",
      klo: "kalau",
      kalo: "kalau",
      kl: "kalau",
      tp: "tapi",
      trs: "terus",
      bgt: "banget",
      bngt: "banget",
      byk: "banyak",
      jg: "juga",
      sm: "sama",
      dr: "dari",
      pd: "pada",
      spt: "seperti",
      sprt: "seperti",
      hrs: "harus",
      hr: "hari",
      bln: "bulan",
      thn: "tahun",
      sblm: "sebelum",
      stlh: "setelah",
      sll: "selalu",
      slm: "salam",
      trm: "terima",
      ksh: "kasih",
      trmksh: "terimakasih",
      mksh: "makasih",
      thx: "thanks",
      tq: "thank you",
    };

    // Apply replacements for whole words only
    let words = normalized.split(" ");
    for (let i = 0; i < words.length; i++) {
      if (commonReplacements[words[i]]) {
        words[i] = commonReplacements[words[i]];
      }
    }

    return words.join(" ");
  }

  /**
   * Fuzzy match text against a list of keywords with typo tolerance
   * @param {string} text - The text to check
   * @param {Array<string>} keywords - List of keywords to match against
   * @param {number} threshold - Optional similarity threshold (0-1)
   * @returns {boolean} - True if any keyword matches
   */
  fuzzyMatch(text, keywords, threshold = 0.8) {
    if (!text) return false;

    // Direct match check first (faster)
    for (const keyword of keywords) {
      if (text.includes(keyword)) {
        return true;
      }
    }

    // Split text into words for word-level matching
    const words = text.split(" ");

    // Check each keyword against each word with Levenshtein distance
    for (const word of words) {
      if (word.length <= 2) continue; // Skip very short words

      for (const keyword of keywords) {
        if (keyword.length <= 2) {
          // For very short keywords (2 chars or less), require exact match
          if (word === keyword) return true;
          continue;
        }

        // For longer words, calculate dynamic threshold based on word length
        // Shorter words need higher similarity, longer words can tolerate more typos
        const dynamicThreshold = Math.max(
          0.7,
          1 - (0.1 * Math.min(keyword.length, 10)) / 10
        );
        const actualThreshold = threshold || dynamicThreshold;

        // Calculate similarity
        const distance = this.levenshteinDistance(word, keyword);
        const maxLength = Math.max(word.length, keyword.length);
        const similarity = 1 - distance / maxLength;

        if (similarity >= actualThreshold) {
          return true;
        }

        // Check for partial match at beginning of word (common in Indonesian)
        if (keyword.length > 3 && word.length > 3) {
          const partLength = Math.min(keyword.length, word.length) - 1;
          const wordPart = word.substring(0, partLength);
          const keywordPart = keyword.substring(0, partLength);

          const partDistance = this.levenshteinDistance(wordPart, keywordPart);
          const partSimilarity = 1 - partDistance / partLength;

          if (partSimilarity >= 0.85) {
            // Higher threshold for partial matches
            return true;
          }
        }
      }
    }

    return false;
  }

  /**
   * Calculate Levenshtein distance between two strings
   * @param {string} a - First string
   * @param {string} b - Second string
   * @returns {number} - Edit distance
   */
  levenshteinDistance(a, b) {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;

    const matrix = [];

    // Initialize matrix
    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= a.length; j++) {
      matrix[0][j] = j;
    }

    // Fill matrix
    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        const cost = a[j - 1] === b[i - 1] ? 0 : 1;
        matrix[i][j] = Math.min(
          matrix[i - 1][j] + 1, // deletion
          matrix[i][j - 1] + 1, // insertion
          matrix[i - 1][j - 1] + cost // substitution
        );

        // Transposition (for Indonesian common typos like 'teh' vs 'the')
        if (i > 1 && j > 1 && a[j - 1] === b[i - 2] && a[j - 2] === b[i - 1]) {
          matrix[i][j] = Math.min(matrix[i][j], matrix[i - 2][j - 2] + cost);
        }
      }
    }

    return matrix[b.length][a.length];
  }

  /**
   * Fuzzy match text against a list of keywords
   * This helps handle typos and variations in input
   */
  fuzzyMatch(text, keywords) {
    // Direct match first (faster)
    for (const keyword of keywords) {
      if (text.includes(keyword)) {
        return true;
      }
    }

    // If no direct match, try fuzzy matching for longer text
    if (text.length > 10) {
      const words = text.split(" ");

      for (const word of words) {
        if (word.length < 3) continue; // Skip very short words

        for (const keyword of keywords) {
          if (keyword.length < 3) continue; // Skip very short keywords

          // Calculate Levenshtein distance for words of similar length
          if (Math.abs(word.length - keyword.length) <= 2) {
            const distance = this.levenshteinDistance(word, keyword);
            // Allow 1 error for short words, 2 for longer words
            const maxAllowedErrors = keyword.length <= 5 ? 1 : 2;

            if (distance <= maxAllowedErrors) {
              return true;
            }
          }
        }
      }
    }

    return false;
  }

  /**
   * Calculate simple sentiment score
   */
  calculateSentiment(text) {
    const positiveWords = [
      "bagus",
      "baik",
      "senang",
      "suka",
      "puas",
      "terima kasih",
      "makasih",
      "mantap",
      "keren",
      "hebat",
      "wow",
      "luar biasa",
      "cepat",
      "ramah",
    ];

    const negativeWords = [
      "buruk",
      "jelek",
      "lambat",
      "mahal",
      "kecewa",
      "rugi",
      "marah",
      "kesal",
      "bingung",
      "sulit",
      "masalah",
      "problem",
      "gagal",
      "error",
    ];

    let score = 0.5; // Neutral starting point

    // Count positive and negative words
    let positiveCount = 0;
    let negativeCount = 0;

    for (const word of positiveWords) {
      if (text.includes(word)) positiveCount++;
    }

    for (const word of negativeWords) {
      if (text.includes(word)) negativeCount++;
    }

    // Adjust score based on counts
    if (positiveCount > 0 || negativeCount > 0) {
      const total = positiveCount + negativeCount;
      score = 0.5 + (0.5 * (positiveCount - negativeCount)) / total;
    }

    return score;
  }

  wordToNumeric(word) {
    let sum = 0;
    for (let i = 0; i < word.length; i++) {
      sum += word.charCodeAt(i);
    }
    return (sum % 255) / 255;
  }

  /**
   * Indonesian Porter Stemmer implementation
   * Removes common Indonesian suffixes to find word roots
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
  }

  /**
   * Enhanced context processing with Indonesian Porter Stemmer
   * and improved similarity matching
   */
  async processContext(question) {
    try {
      if (!this.isInitialized) {
        await this.init();
      }

      const normalizedQuestion = this.normalizeText(question);
      const stemmedQuestion = this.stemText(normalizedQuestion);

      const questionVector = this.textToEnhancedVector(question);
      const brainResult = this.brainNetwork.run(questionVector);

      // Calculate relevance scores with multiple methods
      const relevantEntries = this.dataset
        .map((entry) => {
          const normalizedEntry = this.normalizeText(entry.question);
          const stemmedEntry = this.stemText(normalizedEntry);

          // Calculate multiple similarity scores
          const exactSimilarity = this.calculateSimilarity(
            normalizedQuestion,
            normalizedEntry
          );
          const stemmedSimilarity = this.calculateSimilarity(
            stemmedQuestion,
            stemmedEntry
          );
          const semanticSimilarity = this.calculateSemanticSimilarity(
            normalizedQuestion,
            normalizedEntry
          );
          const contextualSimilarity = this.calculateContextualRelevance(
            question,
            entry
          );

          // Weighted combined score with emphasis on stemmed similarity for Indonesian
          const combinedScore =
            exactSimilarity * 0.3 +
            stemmedSimilarity * 0.4 +
            semanticSimilarity * 0.2 +
            contextualSimilarity * 0.1;

          return {
            ...entry,
            exactSimilarity,
            stemmedSimilarity,
            semanticSimilarity,
            contextualSimilarity,
            combinedScore,
          };
        })
        // Enhanced filtering with dynamic thresholds
        .filter((entry) => {
          const questionLength = normalizedQuestion.split(" ").length;
          const baseThreshold = questionLength > 5 ? 0.18 : 0.22;

          return (
            entry.combinedScore > baseThreshold ||
            entry.stemmedSimilarity > 0.35 ||
            entry.semanticSimilarity > 0.4
          );
        })
        // Sort by combined score
        .sort((a, b) => b.combinedScore - a.combinedScore)
        .slice(0, 5);

      const brainRelevance = this.calculateBrainRelevance(brainResult);

      const bestMatch = relevantEntries.length > 0 ? relevantEntries[0] : null;
      const context = bestMatch ? bestMatch.answer : "";
      const confidence = bestMatch ? bestMatch.combinedScore : 0;

      return {
        brainRelevance,
        context,
        confidence,
        relevantEntries: relevantEntries.map((entry) => ({
          question: entry.question,
          answer: entry.answer,
          tags: entry.tags,
          exactSimilarity: entry.exactSimilarity,
          stemmedSimilarity: entry.stemmedSimilarity,
          semanticSimilarity: entry.semanticSimilarity,
          contextualSimilarity: entry.contextualSimilarity,
          combinedScore: entry.combinedScore,
        })),
        totalDatasetSize: this.dataset.length,
        brainAnalysis: brainResult,
        stemmedQuestion,
        normalizedQuestion,
      };
    } catch (error) {
      logger.error(`Error processing context: ${error.message}`);
      return {
        brainRelevance: 0.3,
        predictedCategory: "unknown",
        context: "",
        confidence: 0,
        relevantEntries: [],
        totalDatasetSize: 0,
        brainAnalysis: {},
        stemmedQuestion: question,
        normalizedQuestion: question,
      };
    }
  }


  /**
   * Get predicted tags from brain.js output
   * @param {Object} brainResult - Output from brain.js network
   * @returns {Array<string>} - Array of predicted tags
   */
  getPredictedTags(brainResult) {
    const tagThreshold = 0.5;
    const tags = [];

    // Get all unique tags from the dataset
    const uniqueTags = this.extractUniqueTags();

    // Check each tag output dynamically
    uniqueTags.forEach((tag) => {
      const tagKey = `tag_${tag}`;
      if (brainResult[tagKey] && brainResult[tagKey] > tagThreshold) {
        tags.push(tag);
      }
    });

    // If no tags predicted, use 'unknown'
    return tags.length > 0 ? tags : ["unknown"];
  }

  /**
   * Calculate text similarity with multiple methods, optimized for large datasets
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Similarity score between 0-1
   */
  calculateSimilarity(text1, text2) {
    // Use caching for large datasets to avoid redundant calculations
    const cacheKey = `sim_${text1.substring(0, 40)}_${text2.substring(0, 40)}`;
    if (this._similarityCache && this._similarityCache[cacheKey]) {
      return this._similarityCache[cacheKey];
    }

    // Initialize similarity cache if not exists
    if (!this._similarityCache) {
      this._similarityCache = {};
      this._similarityCacheSize = 0;
      this._similarityCacheMaxSize = 1000; // Limit cache size for memory management
    }

    // Fast path for identical texts
    if (text1 === text2) {
      return 1.0;
    }

    // Fast path for very different length texts (likely not similar)
    const lengthRatio =
      Math.min(text1.length, text2.length) /
      Math.max(text1.length, text2.length);
    if (lengthRatio < 0.3) {
      return lengthRatio * 0.5; // Return a low similarity score based on length difference
    }

    // Normalize texts - reuse if already normalized
    const normalized1 = text1.startsWith("norm:")
      ? text1.substring(5)
      : this.normalizeText(text1);
    const normalized2 = text2.startsWith("norm:")
      ? text2.substring(5)
      : this.normalizeText(text2);

    // For large datasets, use a faster approach
    const isLargeDataset = this.dataset && this.dataset.length > 20000;

    // Get words and filter out very short ones
    const words1 = new Set(normalized1.split(" ").filter((w) => w.length > 1));
    const words2 = new Set(normalized2.split(" ").filter((w) => w.length > 1));

    // Calculate exact word match (Jaccard similarity)
    const exactIntersection = new Set([...words1].filter((x) => words2.has(x)));
    const union = new Set([...words1, ...words2]);
    const jaccardSimilarity =
      union.size > 0 ? exactIntersection.size / union.size : 0;

    // For very short texts or high exact similarity, skip expensive calculations
    if ((words1.size <= 3 && words2.size <= 3) || jaccardSimilarity > 0.8) {
      const result = jaccardSimilarity;

      // Cache result for future reuse if cache isn't full
      if (this._similarityCacheSize < this._similarityCacheMaxSize) {
        this._similarityCache[cacheKey] = result;
        this._similarityCacheSize++;
      }

      return result;
    }

    // Calculate fuzzy word match for handling typos - optimized for large datasets
    let fuzzyMatches = 0;

    // For large datasets, limit the number of words to compare
    const maxWordsToCompare = isLargeDataset ? 10 : words1.size;
    const words1Array = [...words1]
      .slice(0, maxWordsToCompare)
      .filter((w) => w.length >= 3);

    for (const word1 of words1Array) {
      // For large datasets, limit the number of comparisons per word
      const words2Array = isLargeDataset
        ? [...words2]
            .filter(
              (w) => Math.abs(w.length - word1.length) <= 2 && w.length >= 3
            )
            .slice(0, 5)
        : [...words2].filter(
            (w) => Math.abs(w.length - word1.length) <= 2 && w.length >= 3
          );

      for (const word2 of words2Array) {
        // Use faster distance calculation for large datasets
        let distance;
        if (isLargeDataset) {
          // Simple character difference count for large datasets
          distance = this.calculateFastDistance(word1, word2);
        } else {
          // Full Levenshtein distance for smaller datasets
          distance = this.levenshteinDistance(word1, word2);
        }

        // Allow 1 error for short words, 2 for longer words
        const maxAllowedErrors =
          Math.min(word1.length, word2.length) <= 5 ? 1 : 2;

        if (distance <= maxAllowedErrors) {
          fuzzyMatches++;
          break; // Found a match for this word, move to next
        }
      }
    }

    // Calculate fuzzy similarity score
    const fuzzyJaccardSimilarity =
      union.size > 0 ? (exactIntersection.size + fuzzyMatches) / union.size : 0;

    // For large datasets, skip or simplify expensive calculations
    let charSimilarity, keywordSimilarity, ngramSimilarity;

    if (isLargeDataset) {
      // Simplified character similarity for large datasets
      charSimilarity = this.calculateFastCharacterSimilarity(
        normalized1,
        normalized2
      );

      // Simplified keyword importance for large datasets
      keywordSimilarity = this.calculateFastKeywordSimilarity(
        normalized1,
        normalized2
      );

      // Skip n-gram similarity for large datasets
      ngramSimilarity = 0.5; // Default value
    } else {
      // Full calculations for smaller datasets
      charSimilarity = this.calculateCharacterSimilarity(
        normalized1,
        normalized2
      );
      keywordSimilarity = this.calculateKeywordSimilarity(
        normalized1,
        normalized2
      );
      ngramSimilarity = this.calculateNgramSimilarity(normalized1, normalized2);
    }

    // Adjust weights based on dataset size
    let weights;
    if (isLargeDataset) {
      weights = {
        jaccard: 0.4,
        fuzzyJaccard: 0.3,
        char: 0.1,
        keyword: 0.2,
        ngram: 0.0, // Skip n-gram for large datasets
      };
    } else {
      weights = {
        jaccard: 0.3,
        fuzzyJaccard: 0.25,
        char: 0.15,
        keyword: 0.2,
        ngram: 0.1,
      };
    }

    // Weighted combination of all similarity measures
    const result =
      jaccardSimilarity * weights.jaccard +
      fuzzyJaccardSimilarity * weights.fuzzyJaccard +
      charSimilarity * weights.char +
      keywordSimilarity * weights.keyword +
      ngramSimilarity * weights.ngram;

    // Cache result for future reuse if cache isn't full
    if (this._similarityCacheSize < this._similarityCacheMaxSize) {
      this._similarityCache[cacheKey] = result;
      this._similarityCacheSize++;
    }

    return result;
  }

    /**
   * Menghitung relevansi hasil prediksi brain.js terhadap pertanyaan.
   * Nilai lebih tinggi berarti prediksi lebih yakin dan relevan.
   * @param {Object} brainResult - Output dari brain.js network (hasil .run)
   * @returns {number} - Skor relevansi antara 0 dan 1
   */
  calculateBrainRelevance(brainResult) {
    if (!brainResult || typeof brainResult !== "object") return 0;
    // Ambil semua nilai output tag_*
    const tagScores = Object.entries(brainResult)
      .filter(([key]) => key.startsWith("tag_"))
      .map(([, value]) => typeof value === "number" ? value : 0);

    if (tagScores.length === 0) return 0;
    // Ambil skor tertinggi sebagai relevansi utama
    const maxScore = Math.max(...tagScores);
    // Rata-rata juga bisa dipakai jika ingin lebih konservatif:
    // const avgScore = tagScores.reduce((a, b) => a + b, 0) / tagScores.length;
    return maxScore;
  }

  /**
   * Calculate fast character similarity for large datasets
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Similarity score between 0-1
   */
  calculateFastCharacterSimilarity(text1, text2) {
    // For very different length texts, return a low score
    const lengthRatio =
      Math.min(text1.length, text2.length) /
      Math.max(text1.length, text2.length);
    if (lengthRatio < 0.5) {
      return lengthRatio;
    }

    // Sample characters from both texts
    const sampleSize = Math.min(text1.length, text2.length, 20);
    let matchCount = 0;

    for (let i = 0; i < sampleSize; i++) {
      const index = Math.floor((i * text1.length) / sampleSize);
      const char1 = text1.charAt(index);

      // Check if character exists in text2
      if (text2.includes(char1)) {
        matchCount++;
      }
    }

    return matchCount / sampleSize;
  }

  /**
   * Calculate fast keyword similarity for large datasets
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Similarity score between 0-1
   */
  calculateFastKeywordSimilarity(text1, text2) {
    // Important keywords for Indonesian customer service context
    const importantKeywords = [
      "harga",
      "tersedia",
      "netflix",
      "spotify",
      "bayar",
      "bantuan",
      "akun",
      "password",
      "error",
      "masalah",
      "langganan",
      "premium",
    ];

    // Count keywords in both texts
    let matches = 0;
    let total = 0;

    for (const keyword of importantKeywords) {
      const inText1 = text1.includes(keyword);
      const inText2 = text2.includes(keyword);

      if (inText1 || inText2) {
        total++;
        if (inText1 && inText2) {
          matches++;
        }
      }
    }

    return total > 0 ? matches / total : 0;
  }

  /**
   * Calculate fast distance between two words
   * @param {string} word1 - First word
   * @param {string} word2 - Second word
   * @returns {number} - Distance score
   */
  calculateFastDistance(word1, word2) {
    // If lengths differ by more than 2, they're definitely not similar
    if (Math.abs(word1.length - word2.length) > 2) {
      return 3; // Return a value higher than our threshold
    }

    // Check first and last characters
    let distance = 0;
    if (word1.charAt(0) !== word2.charAt(0)) distance++;
    if (word1.charAt(word1.length - 1) !== word2.charAt(word2.length - 1))
      distance++;

    // Check for character presence
    for (let i = 0; i < word1.length; i++) {
      if (!word2.includes(word1.charAt(i))) {
        distance++;
      }
      if (distance > 2) break; // Early termination
    }

    return distance;
  }

  /**
   * Calculate n-gram similarity between two texts
   * This helps with phrase matching and word order
   */
  calculateNgramSimilarity(text1, text2) {
    // Generate bigrams (pairs of consecutive words)
    const getBigrams = (text) => {
      const words = text.split(" ");
      const bigrams = [];
      for (let i = 0; i < words.length - 1; i++) {
        bigrams.push(`${words[i]} ${words[i + 1]}`);
      }
      return new Set(bigrams);
    };

    const bigrams1 = getBigrams(text1);
    const bigrams2 = getBigrams(text2);

    if (bigrams1.size === 0 || bigrams2.size === 0) return 0;

    // Calculate bigram overlap
    const intersection = new Set([...bigrams1].filter((x) => bigrams2.has(x)));
    const union = new Set([...bigrams1, ...bigrams2]);

    return intersection.size / union.size;
  }

  /**
   * Calculate semantic similarity between two texts, optimized for large datasets
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Semantic similarity score between 0-1
   */
  calculateSemanticSimilarity(text1, text2) {
    // Use caching for large datasets to avoid redundant calculations
    const cacheKey = `${text1.substring(0, 50)}_${text2.substring(0, 50)}`;
    if (this._semanticCache && this._semanticCache[cacheKey]) {
      return this._semanticCache[cacheKey];
    }

    // Initialize semantic cache if not exists (limited size for memory management)
    if (!this._semanticCache) {
      this._semanticCache = {};
      this._semanticCacheSize = 0;
      this._semanticCacheMaxSize = 500; // Limit cache size
    }

    // Normalize texts - reuse if already normalized
    const normalized1 = text1.startsWith("norm:")
      ? text1.substring(5)
      : this.normalizeText(text1);
    const normalized2 = text2.startsWith("norm:")
      ? text2.substring(5)
      : this.normalizeText(text2);

    // Fast path for identical texts
    if (normalized1 === normalized2) {
      return 1.0;
    }

    // Fast path for very different length texts (likely not similar)
    const lengthRatio =
      Math.min(normalized1.length, normalized2.length) /
      Math.max(normalized1.length, normalized2.length);
    if (lengthRatio < 0.3) {
      return lengthRatio * 0.5; // Return a low similarity score based on length difference
    }

    // For large datasets, use a faster approach with fewer components
    const isLargeDataset = this.dataset && this.dataset.length > 20000;

    // Extract patterns - only if needed and with caching for large datasets
    let patterns1, patterns2, patternSimilarity;
    if (!isLargeDataset || normalized1.length < 100) {
      // Skip for very long texts in large datasets
      patterns1 = this.extractPatterns(normalized1);
      patterns2 = this.extractPatterns(normalized2);

      // Calculate pattern similarity with Set operations for speed
      const patternSet1 = new Set(patterns1);
      const patternSet2 = new Set(patterns2);
      const intersection = new Set(
        [...patternSet1].filter((x) => patternSet2.has(x))
      );
      const union = new Set([...patternSet1, ...patternSet2]);

      patternSimilarity = union.size > 0 ? intersection.size / union.size : 0;
    } else {
      // For large datasets with long texts, use a simplified approach
      patternSimilarity = 0.5; // Default value
    }

    // Extract intent - use direct regex for speed in large datasets
    let intent1, intent2, intentSimilarity;
    if (isLargeDataset) {
      // Fast intent extraction for large datasets
      intent1 = this.extractIntentFast(normalized1);
      intent2 = this.extractIntentFast(normalized2);
    } else {
      intent1 = this.extractIntent(normalized1);
      intent2 = this.extractIntent(normalized2);
    }

    // Calculate intent similarity with optimized lookup
    if (intent1 === intent2) {
      intentSimilarity = 1.0; // Exact match
    } else if (intent1 === "general" || intent2 === "general") {
      intentSimilarity = 0.3; // One is general, partial match
    } else {
      // Use static map for related intents to avoid recreating object
      const relatedIntentsMap = this._getRelatedIntentsMap();

      if (
        relatedIntentsMap[intent1] &&
        relatedIntentsMap[intent1].includes(intent2)
      ) {
        intentSimilarity = 0.5; // Related intents
      } else {
        intentSimilarity = 0.1; // Different intents
      }
    }

    // Calculate entity similarity - optimize for large datasets
    let entitySimilarity;
    if (isLargeDataset) {
      // Fast entity similarity for large datasets
      const entities1 = this.extractEntitiesFast(normalized1);
      const entities2 = this.extractEntitiesFast(normalized2);
      entitySimilarity = this.calculateEntitySimilarityFast(
        entities1,
        entities2
      );
    } else {
      // More accurate entity similarity for smaller datasets
      const entities1 = this.extractEntities(normalized1);
      const entities2 = this.extractEntities(normalized2);
      entitySimilarity = this.calculateEntitySimilarity(entities1, entities2);
    }

    // Calculate context similarity - skip for large datasets with long texts
    let contextSimilarity;
    if (
      isLargeDataset &&
      (normalized1.length > 100 || normalized2.length > 100)
    ) {
      // Use a simplified approach for large datasets with long texts
      contextSimilarity = 0.5; // Default value
    } else {
      contextSimilarity = this.calculateContextSimilarity(
        normalized1,
        normalized2
      );
    }

    // Calculate sentiment similarity - only if needed
    let sentimentSimilarity;
    if (
      patterns1 &&
      (patterns1.includes("positive_sentiment") ||
        patterns1.includes("negative_sentiment") ||
        (patterns2 &&
          (patterns2.includes("positive_sentiment") ||
            patterns2.includes("negative_sentiment"))))
    ) {
      // Calculate sentiment only when sentiment patterns are detected
      const sentiment1 = this.calculateSentiment(normalized1);
      const sentiment2 = this.calculateSentiment(normalized2);
      sentimentSimilarity = 1 - Math.abs(sentiment1 - sentiment2);
    } else {
      sentimentSimilarity = 0.5; // Default neutral value
    }

    // Adjust weights based on dataset size for optimal performance
    let weights;
    if (isLargeDataset) {
      weights = {
        intent: 0.4,
        pattern: 0.3,
        entity: 0.2,
        context: 0.1,
        sentiment: 0.0, // Skip sentiment for large datasets
      };
    } else {
      weights = {
        intent: 0.35,
        pattern: 0.25,
        entity: 0.2,
        context: 0.15,
        sentiment: 0.05,
      };
    }

    // Weighted combination of all semantic measures
    const result =
      intentSimilarity * weights.intent +
      patternSimilarity * weights.pattern +
      entitySimilarity * weights.entity +
      contextSimilarity * weights.context +
      sentimentSimilarity * weights.sentiment;

    // Cache result for future reuse if cache isn't full
    if (this._semanticCacheSize < this._semanticCacheMaxSize) {
      this._semanticCache[cacheKey] = result;
      this._semanticCacheSize++;
    }

    return result;
  }

  /**
   * Get related intents map (singleton pattern to avoid recreation)
   * @returns {Object} Map of related intents
   * @private
   */
  _getRelatedIntentsMap() {
    if (!this._relatedIntentsMap) {
      this._relatedIntentsMap = {
        price_inquiry: ["payment", "available"],
        payment: ["price_inquiry", "refund_policy"],
        help: ["technical_details", "refund_policy"],
        available: ["price_inquiry", "technical_details"],
        refund_policy: ["payment", "help"],
        technical_details: ["help", "available"],
        referral_loyalty: ["price_inquiry", "emerging_services"],
        emerging_services: ["referral_loyalty", "technical_details"],
      };
    }
    return this._relatedIntentsMap;
  }

  /**
   * Fast intent extraction for large datasets
   * @param {string} text - Normalized text
   * @returns {string} - Extracted intent
   */
  extractIntentFast(text) {
    // Direct keyword matching for speed
    if (
      text.includes("harga") ||
      text.includes("biaya") ||
      text.includes("tarif") ||
      text.includes("berapa")
    ) {
      return "price_inquiry";
    }
    if (
      text.includes("bayar") ||
      text.includes("pembayaran") ||
      text.includes("transfer")
    ) {
      return "payment";
    }
    if (
      text.includes("bantuan") ||
      text.includes("bantu") ||
      text.includes("tolong") ||
      text.includes("help")
    ) {
      return "help";
    }
    if (
      text.includes("ada") ||
      text.includes("tersedia") ||
      text.includes("ready") ||
      text.includes("stok")
    ) {
      return "available";
    }
    if (
      text.includes("hai") ||
      text.includes("halo") ||
      text.includes("hello") ||
      text.includes("hi")
    ) {
      return "greeting";
    }
    if (
      text.includes("terima kasih") ||
      text.includes("makasih") ||
      text.includes("thanks") ||
      text.includes("bye")
    ) {
      return "goodbye";
    }

    return "general";
  }

  /**
   * Fast entity extraction for large datasets
   * @param {string} text - Normalized text
   * @returns {Array<string>} - Extracted entities
   */
  extractEntitiesFast(text) {
    const entities = [];
    const lowerText = text.toLowerCase();

    // Direct string inclusion check for speed
    const commonEntities = [
      "netflix",
      "spotify",
      "youtube",
      "disney",
      "canva",
      "vidio",
      "amazon",
      "hbo",
      "game pass",
      "chatgpt",
      "prime",
      "viu",
      "wetv",
    ];

    for (const entity of commonEntities) {
      if (lowerText.includes(entity)) {
        entities.push(entity);
      }
    }

    return entities;
  }

  /**
   * Fast entity similarity calculation for large datasets
   * @param {Array<string>} entities1 - First set of entities
   * @param {Array<string>} entities2 - Second set of entities
   * @returns {number} - Similarity score between 0-1
   */
  calculateEntitySimilarityFast(entities1, entities2) {
    if (entities1.length === 0 && entities2.length === 0) return 1.0; // Both empty = perfect match
    if (entities1.length === 0 || entities2.length === 0) return 0.0; // One empty = no match

    // Use Set operations for faster intersection and union
    const set1 = new Set(entities1);
    const set2 = new Set(entities2);

    // Calculate Jaccard similarity (intersection / union)
    const intersection = new Set([...set1].filter((x) => set2.has(x)));
    const union = new Set([...set1, ...set2]);

    return intersection.size / union.size;
  }

  /**
   * Calculate context similarity based on topic modeling, optimized for large datasets
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Context similarity score between 0-1
   */
  calculateContextSimilarity(text1, text2) {
    // Use caching for large datasets to avoid redundant calculations
    const cacheKey = `ctx_${text1.substring(0, 30)}_${text2.substring(0, 30)}`;
    if (this._contextSimCache && this._contextSimCache[cacheKey]) {
      return this._contextSimCache[cacheKey];
    }

    // Initialize context similarity cache if not exists
    if (!this._contextSimCache) {
      this._contextSimCache = {};
      this._contextSimCacheSize = 0;
      this._contextSimCacheMaxSize = 300; // Limit cache size for memory management
    }

    // Fast path for identical texts
    if (text1 === text2) {
      return 1.0;
    }

    // For large datasets, use a faster approach
    const isLargeDataset = this.dataset && this.dataset.length > 20000;

    // Get or initialize topic keywords map (singleton pattern)
    const topics = this._getTopicKeywordsMap();

    // Calculate topic vectors for both texts with optimizations
    const getTopicVector = (text) => {
      const vector = {};
      const lowerText = text.toLowerCase();

      // For large datasets, only check most important topics
      const topicsToCheck = isLargeDataset
        ? ["streaming", "payment", "technical"] // Most important topics for large datasets
        : Object.keys(topics); // All topics for smaller datasets

      for (const topic of topicsToCheck) {
        const keywords = topics[topic];

        // For large datasets, use a faster approach with fewer keywords
        const keywordsToCheck = isLargeDataset
          ? keywords.slice(0, 5) // Only check first 5 keywords for large datasets
          : keywords; // Check all keywords for smaller datasets

        let matchCount = 0;
        for (const keyword of keywordsToCheck) {
          // Direct string inclusion check for speed
          if (lowerText.includes(keyword)) {
            matchCount++;
          }
        }

        // Normalize by the number of keywords checked
        vector[topic] = matchCount / keywordsToCheck.length;
      }

      return vector;
    };

    // Calculate topic vectors with optimizations for large datasets
    let vector1, vector2;

    // For very short texts, use a simplified approach
    if (text1.length < 10 || text2.length < 10) {
      // For very short texts, just check if they contain similar keywords
      const keywords1 = text1.toLowerCase().split(" ");
      const keywords2 = text2.toLowerCase().split(" ");

      // Calculate direct keyword overlap
      const set1 = new Set(keywords1);
      const set2 = new Set(keywords2);
      const intersection = new Set([...set1].filter((x) => set2.has(x)));
      const union = new Set([...set1, ...set2]);

      const result = union.size > 0 ? intersection.size / union.size : 0;

      // Cache result for future reuse if cache isn't full
      if (this._contextSimCacheSize < this._contextSimCacheMaxSize) {
        this._contextSimCache[cacheKey] = result;
        this._contextSimCacheSize++;
      }

      return result;
    }

    // For normal texts, calculate topic vectors
    vector1 = getTopicVector(text1);
    vector2 = getTopicVector(text2);

    // Calculate cosine similarity between topic vectors with optimizations
    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;

    // Only calculate for topics that exist in both vectors
    const commonTopics = Object.keys(vector1).filter(
      (topic) => vector2[topic] !== undefined
    );

    for (const topic of commonTopics) {
      dotProduct += vector1[topic] * vector2[topic];
      magnitude1 += vector1[topic] * vector1[topic];
      magnitude2 += vector2[topic] * vector2[topic];
    }

    // Fast path for zero magnitudes
    if (magnitude1 === 0 || magnitude2 === 0) {
      return 0;
    }

    magnitude1 = Math.sqrt(magnitude1);
    magnitude2 = Math.sqrt(magnitude2);

    const result = dotProduct / (magnitude1 * magnitude2);

    // Cache result for future reuse if cache isn't full
    if (this._contextSimCacheSize < this._contextSimCacheMaxSize) {
      this._contextSimCache[cacheKey] = result;
      this._contextSimCacheSize++;
    }

    return result;
  }

  /**
   * Get topic keywords map (singleton pattern to avoid recreation)
   * @returns {Object} Map of topics to keywords
   * @private
   */
  _getTopicKeywordsMap() {
    if (!this._topicKeywordsMap) {
      this._topicKeywordsMap = {
        streaming: [
          "netflix",
          "spotify",
          "youtube",
          "disney",
          "vidio",
          "film",
          "musik",
          "lagu",
          "nonton",
          "dengar",
        ],
        payment: [
          "bayar",
          "pembayaran",
          "harga",
          "biaya",
          "tarif",
          "dana",
          "ovo",
          "gopay",
          "transfer",
          "bank",
        ],
        account: [
          "akun",
          "daftar",
          "register",
          "login",
          "masuk",
          "password",
          "kata sandi",
          "email",
          "profil",
        ],
        technical: [
          "error",
          "masalah",
          "problem",
          "tidak bisa",
          "gagal",
          "bug",
          "crash",
          "loading",
          "lambat",
          "berat",
        ],
        support: [
          "bantuan",
          "help",
          "tolong",
          "customer service",
          "cs",
          "kontak",
          "hubungi",
          "tanya",
        ],
        product: [
          "produk",
          "paket",
          "langganan",
          "premium",
          "basic",
          "standard",
          "fitur",
          "layanan",
        ],
      };
    }
    return this._topicKeywordsMap;
  }

  /**
   * Extract patterns from text with improved pattern recognition
   * Enhanced to handle more Indonesian language patterns and variations
   */
  extractPatterns(text) {
    const patterns = [];
    const lowerText = this.normalizeText(text);

    // Question type patterns - Enhanced with more Indonesian variations
    if (
      this.fuzzyMatch(lowerText, [
        "apa",
        "apakah",
        "apa itu",
        "apa sih",
        "apa ya",
      ])
    )
      patterns.push("what_question");

    if (
      this.fuzzyMatch(lowerText, [
        "bagaimana",
        "gimana",
        "cara",
        "caranya",
        "gmn",
        "bgmn",
        "bgaimana",
        "bagaimn",
      ])
    )
      patterns.push("how_question");

    if (
      this.fuzzyMatch(lowerText, [
        "kenapa",
        "mengapa",
        "knp",
        "knapa",
        "ngapain",
        "alasan",
      ])
    )
      patterns.push("why_question");

    if (
      this.fuzzyMatch(lowerText, [
        "kapan",
        "jam",
        "waktu",
        "tanggal",
        "hari",
        "kpn",
        "jadwal",
      ])
    )
      patterns.push("when_question");

    if (
      this.fuzzyMatch(lowerText, [
        "dimana",
        "mana",
        "lokasi",
        "tempat",
        "dmn",
        "dmana",
      ])
    )
      patterns.push("where_question");

    if (this.fuzzyMatch(lowerText, ["siapa", "sp", "nama"]))
      patterns.push("who_question");

    if (
      this.fuzzyMatch(lowerText, [
        "berapa",
        "harga",
        "biaya",
        "tarif",
        "brp",
        "brapa",
        "harganya",
        "biayanya",
      ])
    )
      patterns.push("price_question");

    // Greeting patterns - Enhanced with more variations
    if (
      this.fuzzyMatch(lowerText, [
        "hai",
        "halo",
        "hallo",
        "hello",
        "hi",
        "hey",
        "selamat",
        "pagi",
        "siang",
        "sore",
        "malam",
        "assalamualaikum",
        "salam",
      ])
    )
      patterns.push("greeting");

    // Availability patterns - Enhanced with more variations
    if (
      this.fuzzyMatch(lowerText, [
        "ada",
        "tersedia",
        "ready",
        "stock",
        "stok",
        "masih",
        "bisa",
        "available",
        "sedia",
        "redi",
        "stok",
        "msh",
      ])
    )
      patterns.push("availability");

    // Payment patterns - Enhanced with more payment methods
    if (
      this.fuzzyMatch(lowerText, [
        "bayar",
        "pembayaran",
        "transfer",
        "dana",
        "ovo",
        "gopay",
        "shopeepay",
        "bca",
        "mandiri",
        "bni",
        "bri",
        "qris",
        "virtual account",
        "va",
        "kartu kredit",
        "debit",
        "ewallet",
        "e-wallet",
        "payment",
      ])
    )
      patterns.push("payment");

    // Help patterns - Enhanced with more variations
    if (
      this.fuzzyMatch(lowerText, [
        "bantuan",
        "bantu",
        "tolong",
        "help",
        "tlg",
        "tlng",
        "bantuin",
        "assistance",
        "support",
      ])
    )
      patterns.push("help");

    // Goodbye patterns - Enhanced with more variations
    if (
      this.fuzzyMatch(lowerText, [
        "terima kasih",
        "makasih",
        "thanks",
        "thx",
        "tq",
        "bye",
        "selamat tinggal",
        "sampai jumpa",
        "dadah",
        "good bye",
        "terimakasih",
        "mksh",
      ])
    )
      patterns.push("goodbye");

    // Service/product patterns - Enhanced with more services
    if (
      this.fuzzyMatch(lowerText, [
        "netflix",
        "spotify",
        "youtube",
        "disney",
        "canva",
        "vidio",
        "amazon",
        "hbo",
        "game pass",
        "chatgpt",
        "prime video",
        "viu",
        "wetv",
        "iqiyi",
        "mola tv",
        "catchplay",
        "apple music",
        "youtube premium",
        "youtube music",
        "joox",
        "deezer",
        "tidal",
      ])
    )
      patterns.push("service_inquiry");

    // Emerging services patterns - Enhanced with more variations
    if (
      this.fuzzyMatch(lowerText, [
        "layanan baru",
        "fitur baru",
        "update",
        "terbaru",
        "coming soon",
        "segera hadir",
        "baru rilis",
        "baru keluar",
        "baru diluncurkan",
        "akan datang",
        "rilis",
        "launch",
      ])
    )
      patterns.push("emerging_services");

    // Refund policy patterns - Enhanced with more variations
    if (
      this.fuzzyMatch(lowerText, [
        "refund",
        "pengembalian",
        "uang kembali",
        "batal",
        "cancel",
        "garansi",
        "kembalikan",
        "batalkan",
        "dibatalkan",
        "refund policy",
        "kebijakan pengembalian",
        "money back",
        "jaminan",
        "warranty",
      ])
    )
      patterns.push("refund_policy");

    // Technical details patterns - Enhanced with more variations
    if (
      this.fuzzyMatch(lowerText, [
        "teknis",
        "spesifikasi",
        "requirement",
        "sistem",
        "device",
        "kompatibel",
        "support",
        "spec",
        "spek",
        "compatible",
        "compatibility",
        "perangkat",
        "hp",
        "laptop",
        "pc",
        "android",
        "ios",
        "windows",
        "mac",
        "technical",
        "specs",
      ])
    )
      patterns.push("technical_details");

    // Referral loyalty patterns - Enhanced with more variations
    if (
      this.fuzzyMatch(lowerText, [
        "referral",
        "ajak teman",
        "bonus",
        "loyalty",
        "poin",
        "reward",
        "cashback",
        "diskon",
        "discount",
        "promo",
        "promosi",
        "kode",
        "code",
        "voucher",
        "kupon",
        "invite",
        "undang",
        "point",
        "rewards",
      ])
    )
      patterns.push("referral_loyalty");

    // Account-related patterns
    if (
      this.fuzzyMatch(lowerText, [
        "akun",
        "account",
        "daftar",
        "register",
        "login",
        "masuk",
        "password",
        "kata sandi",
        "email",
        "profil",
        "profile",
        "sign up",
        "sign in",
        "logout",
        "keluar",
        "username",
        "user",
        "id",
      ])
    )
      patterns.push("account");

    // Problem-related patterns
    if (
      this.fuzzyMatch(lowerText, [
        "error",
        "masalah",
        "problem",
        "tidak bisa",
        "gagal",
        "bug",
        "crash",
        "loading",
        "lambat",
        "berat",
        "eror",
        "trouble",
        "issue",
        "kendala",
        "gangguan",
        "tidak jalan",
        "tidak work",
        "ga bisa",
        "gak bisa",
        "ngga bisa",
        "nggak bisa",
      ])
    )
      patterns.push("problem");

    // Subscription-related patterns
    if (
      this.fuzzyMatch(lowerText, [
        "langganan",
        "subscription",
        "subscribe",
        "berlangganan",
        "paket",
        "plan",
        "premium",
        "basic",
        "standard",
        "family",
        "individual",
        "bulanan",
        "tahunan",
        "monthly",
        "yearly",
        "annual",
      ])
    )
      patterns.push("subscription");

    // Detect sentence structure patterns
    if (lowerText.includes("?")) patterns.push("question_mark");
    if (lowerText.includes("!")) patterns.push("exclamation");
    if (lowerText.length < 15) patterns.push("short_query");
    if (lowerText.length > 50) patterns.push("long_query");
    if (lowerText.split(" ").length <= 3) patterns.push("very_short_query");

    return patterns;
  }

  /**
   * Enhanced tag generation for questions with improved accuracy and typo handling
   * @param {string} question - The input text to tag
   * @param {Object} context - Context information including brain output and dataset entries
   * @returns {Array} - Generated tags
   */
  async tagQuestion(question, context) {
    try {
      const tags = new Set();
      const tagScores = {}; // Track confidence scores for each tag
      const normalizedText = this.normalizeText(question);

      // Add predicted category as tag
      if (
        context.predictedCategory &&
        context.predictedCategory !== "unknown"
      ) {
        tags.add(context.predictedCategory);
        tagScores[context.predictedCategory] = 0.9;
      }

      // Extract tags from relevant entries with similarity weighting
      if (context.relevantEntries && context.relevantEntries.length > 0) {
        // Consider only top 5 matches
        const topEntries = context.relevantEntries.slice(0, 5);

        topEntries.forEach((entry, index) => {
          if (entry.tags) {
            // Handle both array and string tag formats
            const entryTags = Array.isArray(entry.tags)
              ? entry.tags
              : [entry.tags];
            const similarityWeight = 1 - index * 0.15; // Weight by similarity rank

            entryTags.forEach((tag) => {
              // Update tag score based on dataset entry similarity
              const currentScore = tagScores[tag] || 0;
              const newScore = Math.max(currentScore, similarityWeight);
              tagScores[tag] = newScore;

              // Add tag if it has good similarity
              if (similarityWeight > 0.6) {
                tags.add(tag);
              }
            });
          }
        });
      }

      // Add question type tags
      const questionLower = normalizedText.toLowerCase();
      const questionTypeTags = {
        what: "question",
        how: "instruction",
        why: "explanation",
        when: "time",
        where: "location",
        who: "person",
      };

      Object.keys(questionTypeTags).forEach((keyword) => {
        if (questionLower.includes(keyword)) {
          const tag = questionTypeTags[keyword];
          tags.add(tag);
          tagScores[tag] = 0.8;
        }
      });

      // Add pattern tags
      const patterns = this.extractPatterns(question);
      patterns.forEach((pattern) => {
        tags.add(pattern);

        // Boost related tag scores based on patterns
        if (pattern === "price_question") {
          tagScores["price_inquiry"] = Math.max(
            tagScores["price_inquiry"] || 0,
            0.8
          );
          tags.add("price_inquiry");
        } else if (pattern === "what_question") {
          tagScores["information_request"] = Math.max(
            tagScores["information_request"] || 0,
            0.7
          );
        } else if (pattern === "how_question") {
          tagScores["help"] = Math.max(tagScores["help"] || 0, 0.7);
        }
      });

      // Add confidence level tag
      if (context.brainRelevance > 0.8) {
        tags.add("high-confidence");
      } else if (context.brainRelevance < 0.4) {
        tags.add("low-confidence");
      }

      // Add language detection tag
      if (this.isIndonesianText(normalizedText)) {
        tags.add("indonesian");
      } else if (this.isEnglishText(normalizedText)) {
        tags.add("english");
      }

      // Add sentiment tags if strong sentiment is detected
      const sentiment = this.calculateSentiment(normalizedText);
      if (sentiment > 0.7) {
        tags.add("positive_sentiment");
      } else if (sentiment < 0.3) {
        tags.add("negative_sentiment");
      }

      // Add complexity tags
      if (normalizedText.length > 100) {
        tags.add("complex_query");
      } else if (normalizedText.split(" ").length <= 3) {
        tags.add("simple_query");
      }

      return Array.from(tags);
    } catch (error) {
      logger.error(`Error tagging question: ${error.message}`);
      return ["general"];
    }
  }

  /**
   * Detect if text is primarily in Indonesian
   * @param {string} text - The text to analyze
   * @returns {boolean} - True if text appears to be Indonesian
   */
  isIndonesianText(text) {
    if (!text) return false;

    // Common Indonesian words that rarely appear in English
    const indonesianMarkers = [
      "yang",
      "dan",
      "dengan",
      "untuk",
      "tidak",
      "ini",
      "itu",
      "dari",
      "dalam",
      "akan",
      "pada",
      "juga",
      "saya",
      "ke",
      "bisa",
      "ada",
      "oleh",
      "sudah",
      "atau",
      "seperti",
      "saat",
      "harus",
      "mereka",
      "jika",
      "tersebut",
      "karena",
      "kita",
      "kami",
      "adalah",
      "tahun",
      "apa",
      "bagaimana",
      "kenapa",
      "kapan",
      "dimana",
      "siapa",
      "berapa",
      "nya",
      "lah",
      "kah",
      "pun",
      "kan",
      "sih",
      "deh",
      "kok",
      "dong",
      "ya",
    ];

    // Count Indonesian marker words
    const words = text.split(" ");
    let indonesianCount = 0;

    for (const word of words) {
      if (indonesianMarkers.includes(word)) {
        indonesianCount++;
      }
    }

    // Check if a significant portion of words are Indonesian markers
    return indonesianCount >= 1 && indonesianCount / words.length > 0.15;
  }

  /**
   * Detect if text is primarily in English
   * @param {string} text - The text to analyze
   * @returns {boolean} - True if text appears to be English
   */
  isEnglishText(text) {
    if (!text) return false;

    // Common English words that rarely appear in Indonesian
    const englishMarkers = [
      "the",
      "is",
      "are",
      "am",
      "was",
      "were",
      "be",
      "been",
      "being",
      "have",
      "has",
      "had",
      "do",
      "does",
      "did",
      "will",
      "would",
      "shall",
      "should",
      "can",
      "could",
      "may",
      "might",
      "must",
      "ought",
      "i",
      "you",
      "he",
      "she",
      "it",
      "we",
      "they",
      "me",
      "him",
      "her",
      "us",
      "them",
      "my",
      "your",
      "his",
      "its",
      "our",
      "their",
      "mine",
      "yours",
      "hers",
      "ours",
      "theirs",
      "this",
      "that",
      "these",
      "those",
      "what",
      "which",
      "who",
      "whom",
      "whose",
      "when",
      "where",
      "why",
      "how",
    ];

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
  }

  /**
   * Calculate sentiment score for text (primarily for Indonesian)
   * @param {string} text - The text to analyze
   * @returns {number} - Sentiment score from -1 (negative) to 1 (positive)
   */
  calculateSentiment(text) {
    if (!text) return 0;

    // Indonesian positive sentiment words
    const positiveWords = [
      // General positive words
      "bagus",
      "baik",
      "hebat",
      "keren",
      "mantap",
      "mantab",
      "mantul",
      "oke",
      "ok",
      "sip",
      "top",
      "senang",
      "gembira",
      "bahagia",
      "puas",
      "sukses",
      "berhasil",
      "luar biasa",
      "wow",
      "kece",
      "cakep",
      "cantik",
      "ganteng",
      "indah",
      "menarik",
      "menyenangkan",
      "ramah",
      "sopan",
      "cepat",
      "mudah",
      "praktis",
      "efisien",
      "efektif",
      "berguna",
      "bermanfaat",
      "membantu",
      "suka",
      "cinta",
      "sayang",
      "setuju",
      "benar",
      "tepat",
      "akurat",
      "lengkap",
      "sempurna",

      // Customer service specific
      "terima kasih",
      "makasih",
      "thanks",
      "thx",
      "tq",
      "recommended",
      "rekomen",
      "rekomendasi",
      "lancar",
      "cepat",
      "responsif",
      "informatif",
      "jelas",
      "murah",
      "terjangkau",
      "worth it",
      "worth",
      "puas",
      "kepuasan",
      "satisfied",
      "memuaskan",
      "terpercaya",
      "amanah",
      "jujur",
      "profesional",
      "handal",
      "reliable",
      "berkualitas",
      "quality",
      "premium",
      "original",
      "asli",
      "berhasil",
      "sukses",
      "success",
      "beruntung",
      "lucky",
      "promo",
      "diskon",
      "hemat",
      "bonus",
    ];

    // Indonesian negative sentiment words
    const negativeWords = [
      // General negative words
      "buruk",
      "jelek",
      "rusak",
      "busuk",
      "hancur",
      "parah",
      "payah",
      "lemah",
      "lambat",
      "lelet",
      "mahal",
      "kemahalan",
      "boros",
      "rugi",
      "kecewa",
      "sedih",
      "marah",
      "kesal",
      "jengkel",
      "benci",
      "bete",
      "bt",
      "sebel",
      "sebal",
      "bosan",
      "bodo",
      "bodoh",
      "tolol",
      "goblok",
      "gagal",
      "salah",
      "error",
      "eror",
      "cacat",
      "kurang",
      "tidak",
      "tak",
      "ga",
      "gak",
      "nggak",
      "ngga",
      "bukan",
      "jangan",
      "males",
      "malas",
      "susah",
      "sulit",
      "rumit",
      "ribet",
      "repot",

      // Customer service specific
      "komplain",
      "complaint",
      "keluhan",
      "protes",
      "klaim",
      "claim",
      "refund",
      "pengembalian",
      "batal",
      "cancel",
      "dibatalkan",
      "palsu",
      "fake",
      "tipu",
      "penipuan",
      "scam",
      "penipu",
      "lama",
      "lambat",
      "delay",
      "telat",
      "terlambat",
      "pending",
      "tertunda",
      "tidak sampai",
      "hilang",
      "rusak",
      "cacat",
      "tidak sesuai",
      "tidak cocok",
      "tidak lengkap",
      "kurang lengkap",
      "mahal",
      "kemahalan",
      "tidak worth",
      "tidak worth it",
      "buang-buang",
      "sia-sia",
      "percuma",
      "kecewa",
      "disappointed",
      "mengecewakan",
      "tidak puas",
      "tidak memuaskan",
      "unsatisfied",
      "buruk",
      "jelek",
      "tidak bagus",
      "tidak baik",
      "tidak ramah",
      "kasar",
      "jutek",
      "galak",
      "tidak jelas",
      "membingungkan",
      "ambigu",
      "tidak informatif",
      "tidak membantu",
      "unhelpful",
    ];

    // Intensifiers that strengthen sentiment
    const intensifiers = [
      "sangat",
      "amat",
      "sekali",
      "banget",
      "bgt",
      "bngt",
      "sungguh",
      "terlalu",
      "teramat",
      "super",
      "extra",
      "ekstra",
      "paling",
      "lebih",
      "kurang",
    ];

    // Negators that flip sentiment
    const negators = [
      "tidak",
      "tak",
      "ga",
      "gak",
      "nggak",
      "ngga",
      "bukan",
      "jangan",
      "belum",
    ];

    // Split text into words
    const words = text.toLowerCase().split(/\s+/);
    let score = 0;
    let wordCount = 0;
    let intensifierPresent = false;
    let negatorPresent = false;

    // Analyze each word
    for (let i = 0; i < words.length; i++) {
      const word = words[i];

      // Check for intensifiers
      if (intensifiers.includes(word)) {
        intensifierPresent = true;
        continue;
      }

      // Check for negators
      if (negators.includes(word)) {
        negatorPresent = true;
        continue;
      }

      // Check for positive words
      if (positiveWords.includes(word)) {
        let wordScore = 1;

        // Apply intensifier if present
        if (intensifierPresent) {
          wordScore *= 1.5;
          intensifierPresent = false;
        }

        // Apply negator if present
        if (negatorPresent) {
          wordScore *= -1;
          negatorPresent = false;
        }

        score += wordScore;
        wordCount++;
        continue;
      }

      // Check for negative words
      if (negativeWords.includes(word)) {
        let wordScore = -1;

        // Apply intensifier if present
        if (intensifierPresent) {
          wordScore *= 1.5;
          intensifierPresent = false;
        }

        // Apply negator if present (double negative becomes positive)
        if (negatorPresent) {
          wordScore *= -1;
          negatorPresent = false;
        }

        score += wordScore;
        wordCount++;
        continue;
      }

      // Reset modifiers if not used
      if (i > 0 && !words[i - 1].match(/[,.;:!?]$/)) {
        intensifierPresent = false;
        negatorPresent = false;
      }
    }

    // Check for specific phrases (multi-word expressions)
    const positiveExpressions = [
      "terima kasih",
      "thank you",
      "makasih banyak",
      "sangat bagus",
      "sangat baik",
      "sangat membantu",
      "sangat puas",
      "sangat senang",
      "worth it",
      "worth the price",
      "harga terjangkau",
      "pelayanan bagus",
      "pelayanan baik",
      "respon cepat",
      "fast response",
    ];

    const negativeExpressions = [
      "tidak bagus",
      "tidak baik",
      "tidak puas",
      "tidak senang",
      "tidak suka",
      "kurang bagus",
      "kurang baik",
      "kurang puas",
      "kurang senang",
      "kurang suka",
      "terlalu mahal",
      "terlalu lama",
      "terlalu lambat",
      "sangat kecewa",
      "sangat marah",
      "pelayanan buruk",
      "respon lambat",
      "slow response",
      "not worth it",
      "not worth the price",
    ];

    // Check for positive expressions
    for (const expression of positiveExpressions) {
      if (text.toLowerCase().includes(expression)) {
        score += 1.5;
        wordCount++;
      }
    }

    // Check for negative expressions
    for (const expression of negativeExpressions) {
      if (text.toLowerCase().includes(expression)) {
        score -= 1.5;
        wordCount++;
      }
    }

    // Normalize score
    return wordCount > 0 ? score / (wordCount * 1.5) : 0;
  }

  /**
   * Calculate character-level similarity using Levenshtein distance
   * Optimized for large datasets with caching and fast paths
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Similarity score between 0-1
   */
  calculateCharacterSimilarity(text1, text2) {
    // Use caching for large datasets to avoid redundant calculations
    const cacheKey = `char_${text1.substring(0, 20)}_${text2.substring(0, 20)}`;
    if (this._charSimCache && this._charSimCache[cacheKey]) {
      return this._charSimCache[cacheKey];
    }

    // Initialize cache if not exists
    if (!this._charSimCache) {
      this._charSimCache = {};
      this._charSimCacheSize = 0;
      this._charSimCacheMaxSize = 500; // Limit cache size
    }

    // Fast path for identical texts
    if (text1 === text2) {
      return 1.0;
    }

    const maxLength = Math.max(text1.length, text2.length);
    if (maxLength === 0) return 1;

    // Fast path for very different length texts
    const lengthRatio = Math.min(text1.length, text2.length) / maxLength;
    if (lengthRatio < 0.5) {
      const result = lengthRatio;

      // Cache result
      if (this._charSimCacheSize < this._charSimCacheMaxSize) {
        this._charSimCache[cacheKey] = result;
        this._charSimCacheSize++;
      }

      return result;
    }

    // For large datasets or long texts, use sampling approach
    const isLargeDataset = this.dataset && this.dataset.length > 20000;
    const isLongText = text1.length > 100 || text2.length > 100;

    if (isLargeDataset && isLongText) {
      // Sample characters instead of processing all
      // Take first 30 chars, middle 20 chars, and last 30 chars
      const sample1 = this._sampleText(text1, 30, 20, 30);
      const sample2 = this._sampleText(text2, 30, 20, 30);

      const distance = this.levenshteinDistance(sample1, sample2);
      const sampleMaxLength = Math.max(sample1.length, sample2.length);
      const result = 1 - distance / sampleMaxLength;

      // Cache result
      if (this._charSimCacheSize < this._charSimCacheMaxSize) {
        this._charSimCache[cacheKey] = result;
        this._charSimCacheSize++;
      }

      return result;
    }

    // For smaller datasets and shorter texts, use the original approach
    const distance = this.levenshteinDistance(text1, text2);
    const result = 1 - distance / maxLength;

    // Cache result
    if (this._charSimCacheSize < this._charSimCacheMaxSize) {
      this._charSimCache[cacheKey] = result;
      this._charSimCacheSize++;
    }

    return result;
  }

  /**
   * Helper method to sample text from beginning, middle and end
   * @private
   * @param {string} text - Text to sample
   * @param {number} startChars - Number of characters to take from start
   * @param {number} middleChars - Number of characters to take from middle
   * @param {number} endChars - Number of characters to take from end
   * @returns {string} - Sampled text
   */
  _sampleText(text, startChars, middleChars, endChars) {
    if (text.length <= startChars + middleChars + endChars) {
      return text; // Text is shorter than sample size, return as is
    }

    const start = text.substring(0, startChars);
    const middleStart = Math.floor((text.length - middleChars) / 2);
    const middle = text.substring(middleStart, middleStart + middleChars);
    const end = text.substring(text.length - endChars);

    return start + middle + end;
  }

  /**
   * Calculate Levenshtein distance between two strings
   */
  /**
   * Calculate Levenshtein distance between two strings
   * Optimized for large datasets with caching and fast paths
   * @param {string} str1 - First string
   * @param {string} str2 - Second string
   * @returns {number} - Levenshtein distance
   */
  levenshteinDistance(str1, str2) {
    // Use caching for large datasets to avoid redundant calculations
    const cacheKey = `lev_${str1.substring(0, 15)}_${str2.substring(0, 15)}`;
    if (this._levCache && this._levCache[cacheKey]) {
      return this._levCache[cacheKey];
    }

    // Initialize cache if not exists
    if (!this._levCache) {
      this._levCache = {};
      this._levCacheSize = 0;
      this._levCacheMaxSize = 500; // Limit cache size
    }

    // Fast path for identical strings
    if (str1 === str2) {
      return 0;
    }

    // Fast path for empty strings
    if (str1.length === 0) return str2.length;
    if (str2.length === 0) return str1.length;

    // Fast path for single character difference
    if (str1.length === 1 && str2.length === 1) {
      return str1 === str2 ? 0 : 1;
    }

    // Fast path for strings with very different lengths
    const lengthDiff = Math.abs(str1.length - str2.length);
    if (lengthDiff > Math.min(str1.length, str2.length)) {
      const result = lengthDiff;

      // Cache result
      if (this._levCacheSize < this._levCacheMaxSize) {
        this._levCache[cacheKey] = result;
        this._levCacheSize++;
      }

      return result;
    }

    // Optimization for large datasets: use row-based approach to save memory
    const isLargeDataset = this.dataset && this.dataset.length > 20000;
    const isLongString = str1.length > 100 || str2.length > 100;

    if (isLargeDataset && isLongString) {
      // Use memory-efficient version for large strings
      let previousRow = new Array(str2.length + 1);
      let currentRow = new Array(str2.length + 1);

      // Initialize first row
      for (let i = 0; i <= str2.length; i++) {
        previousRow[i] = i;
      }

      // Fill in the rest of the matrix
      for (let i = 0; i < str1.length; i++) {
        currentRow[0] = i + 1;

        for (let j = 0; j < str2.length; j++) {
          const insertCost = previousRow[j + 1] + 1;
          const deleteCost = currentRow[j] + 1;
          let replaceCost;

          if (str1[i] === str2[j]) {
            replaceCost = previousRow[j];
          } else {
            replaceCost = previousRow[j] + 1;
          }

          currentRow[j + 1] = Math.min(insertCost, deleteCost, replaceCost);
        }

        // Swap rows
        [previousRow, currentRow] = [currentRow, previousRow];
      }

      const result = previousRow[str2.length];

      // Cache result
      if (this._levCacheSize < this._levCacheMaxSize) {
        this._levCache[cacheKey] = result;
        this._levCacheSize++;
      }

      return result;
    }

    // For smaller datasets and shorter strings, use the original approach
    const matrix = [];

    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }

    const result = matrix[str2.length][str1.length];

    // Cache result
    if (this._levCacheSize < this._levCacheMaxSize) {
      this._levCache[cacheKey] = result;
      this._levCacheSize++;
    }

    return result;
  }

  /**
   * Calculate keyword importance similarity
   */
  /**
   * Calculate keyword similarity based on important Indonesian keywords
   * Optimized for large datasets with caching and fast paths
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Similarity score between 0-1
   */
  calculateKeywordSimilarity(text1, text2) {
    // Use caching for large datasets to avoid redundant calculations
    const cacheKey = `kw_${text1.substring(0, 20)}_${text2.substring(0, 20)}`;
    if (this._kwSimCache && this._kwSimCache[cacheKey]) {
      return this._kwSimCache[cacheKey];
    }

    // Initialize cache if not exists
    if (!this._kwSimCache) {
      this._kwSimCache = {};
      this._kwSimCacheSize = 0;
      this._kwSimCacheMaxSize = 500; // Limit cache size
    }

    // Fast path for identical texts
    if (text1 === text2) {
      return 1.0;
    }

    // Get important keywords map (singleton pattern)
    const importantKeywords = this._getImportantKeywordsMap();

    // Fast path for large datasets: use direct includes instead of toLowerCase for each check
    const isLargeDataset = this.dataset && this.dataset.length > 20000;

    // Convert to lowercase once for efficiency
    const lowerText1 = text1.toLowerCase();
    const lowerText2 = text2.toLowerCase();

    let keywords1, keywords2;

    if (isLargeDataset) {
      // For large datasets, use direct includes check without additional processing
      keywords1 = importantKeywords.filter((keyword) =>
        lowerText1.includes(keyword)
      );
      keywords2 = importantKeywords.filter((keyword) =>
        lowerText2.includes(keyword)
      );
    } else {
      // For smaller datasets, use the original approach
      keywords1 = importantKeywords.filter((keyword) =>
        lowerText1.includes(keyword)
      );
      keywords2 = importantKeywords.filter((keyword) =>
        lowerText2.includes(keyword)
      );
    }

    // Fast path for no keywords
    if (keywords1.length === 0 && keywords2.length === 0) {
      // Cache result
      if (this._kwSimCacheSize < this._kwSimCacheMaxSize) {
        this._kwSimCache[cacheKey] = 0;
        this._kwSimCacheSize++;
      }
      return 0;
    }

    // Fast path for identical keyword sets
    if (
      keywords1.length === keywords2.length &&
      keywords1.every((k) => keywords2.includes(k))
    ) {
      // Cache result
      if (this._kwSimCacheSize < this._kwSimCacheMaxSize) {
        this._kwSimCache[cacheKey] = 1;
        this._kwSimCacheSize++;
      }
      return 1;
    }

    // Calculate Jaccard similarity for keyword sets
    const intersection = keywords1.filter((k) => keywords2.includes(k));
    const union = [...new Set([...keywords1, ...keywords2])];

    const result = intersection.length / union.length;

    // Cache result
    if (this._kwSimCacheSize < this._kwSimCacheMaxSize) {
      this._kwSimCache[cacheKey] = result;
      this._kwSimCacheSize++;
    }

    return result;
  }

  /**
   * Get important keywords map (singleton pattern)
   * @private
   * @returns {Array<string>} - Array of important keywords
   */
  _getImportantKeywordsMap() {
    // Create singleton to avoid recreating this array repeatedly
    if (!this._importantKeywordsMap) {
      this._importantKeywordsMap = [
        "harga",
        "berapa",
        "biaya",
        "tarif",
        "price",
        "ada",
        "tersedia",
        "ready",
        "stock",
        "available",
        "netflix",
        "spotify",
        "youtube",
        "disney",
        "canva",
        "vidio",
        "bayar",
        "pembayaran",
        "transfer",
        "dana",
        "ovo",
        "gopay",
        "bantuan",
        "bantu",
        "help",
        "tolong",
      ];
    }

    return this._importantKeywordsMap;
  }

  /**
   * Extract intent from text
   */
  extractIntent(text) {
    const lowerText = text.toLowerCase();

    if (lowerText.match(/\b(harga|berapa|biaya|tarif)\b/))
      return "price_inquiry";
    if (lowerText.match(/\b(ada|tersedia|ready|stock|masih)\b/))
      return "availability";
    if (lowerText.match(/\b(bayar|pembayaran|transfer|dana|ovo|gopay)\b/))
      return "payment";
    if (lowerText.match(/\b(bantuan|bantu|help|tolong)\b/)) return "help";
    if (lowerText.match(/\b(hai|halo|selamat|pagi|siang|sore|malam)\b/))
      return "greeting";
    if (lowerText.match(/\b(terima kasih|makasih|thanks|bye)\b/))
      return "goodbye";

    return "general";
  }

  /**
   * Calculate entity similarity (products/services)
   * Optimized for large datasets with caching and fast paths
   * @param {string} text1 - First text
   * @param {string} text2 - Second text
   * @returns {number} - Similarity score between 0-1
   */
  calculateEntitySimilarity(text1, text2) {
    // Use caching for large datasets to avoid redundant calculations
    const cacheKey = `ent_${text1.substring(0, 20)}_${text2.substring(0, 20)}`;
    if (this._entSimCache && this._entSimCache[cacheKey]) {
      return this._entSimCache[cacheKey];
    }

    // Initialize cache if not exists
    if (!this._entSimCache) {
      this._entSimCache = {};
      this._entSimCacheSize = 0;
      this._entSimCacheMaxSize = 500; // Limit cache size
    }

    // Fast path for identical texts
    if (text1 === text2) {
      return 1.0;
    }

    // Get entities map (singleton pattern)
    const entities = this._getEntitiesMap();

    // Fast path for large datasets: use direct includes instead of toLowerCase for each check
    const isLargeDataset = this.dataset && this.dataset.length > 20000;

    // Convert to lowercase once for efficiency
    const lowerText1 = text1.toLowerCase();
    const lowerText2 = text2.toLowerCase();

    // Extract entities from texts
    let entities1, entities2;

    if (isLargeDataset) {
      // For large datasets, use direct includes check without additional processing
      entities1 = entities.filter((entity) => lowerText1.includes(entity));
      entities2 = entities.filter((entity) => lowerText2.includes(entity));
    } else {
      // For smaller datasets, use the original approach
      entities1 = entities.filter((entity) => lowerText1.includes(entity));
      entities2 = entities.filter((entity) => lowerText2.includes(entity));
    }

    // Fast path for no entities
    if (entities1.length === 0 && entities2.length === 0) {
      // Cache result
      if (this._entSimCacheSize < this._entSimCacheMaxSize) {
        this._entSimCache[cacheKey] = 0;
        this._entSimCacheSize++;
      }
      return 0;
    }

    // Fast path for identical entity sets
    if (
      entities1.length === entities2.length &&
      entities1.every((e) => entities2.includes(e))
    ) {
      // Cache result
      if (this._entSimCacheSize < this._entSimCacheMaxSize) {
        this._entSimCache[cacheKey] = 1;
        this._entSimCacheSize++;
      }
      return 1;
    }

    // Use Set operations for faster intersection and union (like in calculateEntitySimilarityFast)
    const set1 = new Set(entities1);
    const set2 = new Set(entities2);

    // Calculate Jaccard similarity (intersection / union)
    const intersection = new Set([...set1].filter((x) => set2.has(x)));
    const union = new Set([...set1, ...set2]);

    const result = intersection.size / union.size;

    // Cache result
    if (this._entSimCacheSize < this._entSimCacheMaxSize) {
      this._entSimCache[cacheKey] = result;
      this._entSimCacheSize++;
    }

    return result;
  }

  /**
   * Get entities map (singleton pattern)
   * @private
   * @returns {Array<string>} - Array of entities
   */
  _getEntitiesMap() {
    // Create singleton to avoid recreating this array repeatedly
    if (!this._entitiesMap) {
      this._entitiesMap = [
        "netflix",
        "spotify",
        "youtube",
        "disney",
        "canva",
        "vidio",
        "amazon",
        "hbo",
        "game pass",
        "chatgpt",
        "loklok",
        "prime",
      ];
    }

    return this._entitiesMap;
  }

  /**
   * Retrain the model with new data
   */
  async retrainModel() {
    try {
      logger.info("Retraining Brain.js model...");
      await this.loadDataset();
      await this.trainBrainNetwork();
      await this.saveTrainedModel();
      logger.info("Model retrained successfully");
    } catch (error) {
      logger.error(`Error retraining model: ${error.message}`);
      throw error;
    }
  }

  getModelStats() {
    return {
      isInitialized: this.isInitialized,
      datasetSize: this.dataset ? this.dataset.length : 0,
      modelPath: this.modelPath,
      hasTrainedModel: this.brainNetwork !== null,
    };
  }

  /**
   * Train with provided dataset
   * @param {Array} dataset - The dataset to train with
   * @returns {Object} - Training results
   */
  async trainWithDataset(dataset) {
    try {
      logger.info(
        `Training Brain.js network with provided dataset (${dataset.length} entries)...`
      );

      this.dataset = dataset;

      const result = await this.trainBrainNetwork();

      return {
        success: true,
        error: result.error,
        iterations: result.iterations,
        datasetSize: dataset.length,
      };
    } catch (error) {
      logger.error(`Error training with dataset: ${error.message}`);
      throw error;
    }
  }

  /**
   * Export the trained model
   * @returns {Object} - The exported model data
   */
  exportModel() {
    if (!this.brainNetwork) {
      throw new Error("Brain network not initialized");
    }

    const modelData = this.brainNetwork.toJSON();
    return {
      model: modelData,
      stats: {
        datasetSize: this.dataset ? this.dataset.length : 0,
        timestamp: new Date().toISOString(),
      },
    };
  }

  /**
   * Enhanced findAnswer method with Indonesian Porter Stemmer
   * and advanced context matching for better Indonesian responses
   */
  async findAnswer(question) {
    try {
      if (!this.isInitialized) {
        await this.init();
      }

      // Use enhanced context processing with Indonesian Porter Stemmer
      const contextData = await this.processContext(question);

      // Enhanced fallback handling with better Indonesian responses
      if (
        contextData.relevantEntries.length === 0 ||
        contextData.confidence < 0.1
      ) {
        const intent = this.extractIntent(question);
        const fallbackResponse = this.generateIntelligentFallback(
          question,
          intent
        );

        return {
          answer: fallbackResponse,
          confidence: 0.1,
          tags: [intent || "unknown"],
          source: "intelligent_fallback",
          brainRelevance: contextData.brainRelevance,
          predictedCategory: contextData.predictedCategory,
        };
      }

      const bestMatch = contextData.relevantEntries[0];
      const confidence = contextData.confidence;

      // Enhanced response generation with Indonesian language optimization
      let answer = this.generateContextualResponse(
        question,
        bestMatch,
        confidence,
        contextData
      );

      // Apply Indonesian language refinements
      answer = this.refineIndonesianResponse(
        answer,
        contextData.predictedCategory
      );

      return {
        answer,
        confidence,
        tags: bestMatch.tags || [contextData.predictedCategory],
        source: "enhanced_dataset",
        relevantEntries: contextData.relevantEntries.slice(0, 3),
        brainAnalysis: contextData.brainAnalysis,
        brainRelevance: contextData.brainRelevance,
        predictedCategory: contextData.predictedCategory,
        stemmedQuestion: contextData.stemmedQuestion,
        similarityBreakdown: {
          exact: bestMatch.exactSimilarity,
          stemmed: bestMatch.stemmedSimilarity,
          semantic: bestMatch.semanticSimilarity,
          contextual: bestMatch.contextualSimilarity,
          combined: bestMatch.combinedScore,
        },
      };
    } catch (error) {
      logger.error(`Error finding answer: ${error.message}`);
      return {
        answer:
          "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi dalam beberapa saat.",
        confidence: 0,
        tags: ["error"],
        source: "error",
        brainRelevance: 0,
        predictedCategory: "error",
      };
    }
  }

  /**
   * Generate intelligent fallback responses based on intent and context
   */
  generateIntelligentFallback(question, intent) {
    const fallbackResponses = {
      greeting: [
        "Halo! Selamat datang di layanan kami. Ada yang bisa saya bantu hari ini?",
        "Hai! Senang bertemu dengan Anda. Bagaimana saya bisa membantu?",
        "Selamat datang! Saya siap membantu Anda dengan pertanyaan seputar layanan kami.",
      ],
      price_inquiry: [
        "Untuk informasi harga yang akurat, mohon sebutkan layanan spesifik yang Anda minati. Kami memiliki berbagai paket dengan harga yang kompetitif.",
        "Saya akan senang membantu dengan informasi harga. Bisakah Anda memberitahu layanan mana yang ingin Anda ketahui harganya?",
        "Harga layanan kami bervariasi tergantung paket yang dipilih. Mohon sebutkan layanan yang Anda cari untuk informasi lebih detail.",
      ],
      available: [
        "Untuk mengecek ketersediaan, mohon sebutkan layanan atau produk spesifik yang Anda cari.",
        "Saya dapat membantu mengecek ketersediaan. Layanan apa yang ingin Anda ketahui statusnya?",
        "Ketersediaan layanan dapat berbeda-beda. Mohon sebutkan layanan yang Anda minati untuk pengecekan yang akurat.",
      ],
      payment: [
        "Kami menerima berbagai metode pembayaran. Untuk informasi lebih detail tentang pembayaran, mohon sebutkan layanan yang ingin Anda beli.",
        "Sistem pembayaran kami aman dan mudah. Ada metode pembayaran tertentu yang ingin Anda ketahui?",
        "Proses pembayaran sangat mudah dan aman. Bisakah Anda memberitahu layanan mana yang ingin Anda beli?",
      ],
      help: [
        "Saya di sini untuk membantu Anda! Bisakah Anda menjelaskan lebih detail masalah atau pertanyaan yang Anda hadapi?",
        "Tentu saja saya akan membantu. Mohon jelaskan lebih spesifik apa yang Anda butuhkan.",
        "Saya siap membantu menyelesaikan masalah Anda. Bisa dijelaskan lebih detail situasinya?",
      ],
      goodbye: [
        "Terima kasih telah menggunakan layanan kami! Jangan ragu untuk kembali jika ada pertanyaan lain.",
        "Sampai jumpa! Semoga hari Anda menyenangkan. Kami selalu siap membantu kapan saja.",
        "Terima kasih! Jika ada yang perlu dibantu lagi, jangan sungkan untuk menghubungi kami.",
      ],
    };

    const responses = fallbackResponses[intent] || [
      "Maaf, saya belum sepenuhnya memahami pertanyaan Anda. Bisakah Anda menjelaskan lebih detail atau menggunakan kata-kata yang berbeda?",
      "Saya ingin membantu Anda dengan sebaik-baiknya. Mohon berikan informasi lebih spesifik tentang apa yang Anda cari.",
      "Untuk memberikan jawaban yang tepat, saya memerlukan informasi lebih detail. Bisakah Anda menjelaskan pertanyaan Anda dengan cara lain?",
    ];

    // Select response based on question characteristics
    const questionLength = question.split(" ").length;
    const responseIndex = questionLength > 10 ? 2 : questionLength > 5 ? 1 : 0;

    return responses[responseIndex % responses.length];
  }

  /**
   * Generate contextual response based on confidence and similarity scores
   */
  generateContextualResponse(question, bestMatch, confidence, contextData) {
    let answer = bestMatch.answer;

    // High confidence - direct answer
    if (confidence > 0.8) {
      return answer;
    }

    // Medium-high confidence - confident but polite
    if (confidence > 0.6) {
      const prefixes = [
        "Berdasarkan informasi yang saya miliki, ",
        "Sesuai dengan data kami, ",
        "Menurut informasi terkini, ",
      ];
      const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
      return `${prefix}${answer}`;
    }

    // Medium confidence - helpful but cautious
    if (confidence > 0.4) {
      const templates = [
        `Saya menemukan informasi yang mungkin relevan: ${answer}\n\nApakah ini sesuai dengan yang Anda cari?`,
        `Berdasarkan pencarian saya: ${answer}\n\nJika ini belum tepat, mohon berikan detail lebih spesifik.`,
        `Informasi yang saya temukan: ${answer}\n\nSemoga ini membantu menjawab pertanyaan Anda."`,
      ];
      return templates[Math.floor(Math.random() * templates.length)];
    }

    // Lower confidence - very cautious
    const cautious_templates = [
      `Saya menemukan informasi terkait yang mungkin membantu: ${answer}\n\nNamun, untuk memastikan akurasi, mohon konfirmasi apakah ini sesuai dengan yang Anda maksud.`,
      `Berikut informasi yang berkaitan: ${answer}\n\nJika ini tidak sesuai, bisakah Anda memberikan kata kunci yang lebih spesifik?`,
      `Informasi yang mungkin relevan: ${answer}\n\nUntuk jawaban yang lebih tepat, mohon jelaskan pertanyaan Anda dengan lebih detail."`,
    ];

    return cautious_templates[
      Math.floor(Math.random() * cautious_templates.length)
    ];
  }

  /**
   * Refine response for better Indonesian language flow
   */
  refineIndonesianResponse(answer, category) {
    // Add category-specific refinements
    const categoryRefinements = {
      greeting: (text) => {
        if (!text.includes("!") && !text.includes("?")) {
          return text + "! 😊";
        }
        return text;
      },
      goodbye: (text) => {
        if (!text.includes("!")) {
          return text + "! 👋";
        }
        return text;
      },
      price_inquiry: (text) => {
        // Ensure price information is clear
        if (text.includes("harga") || text.includes("biaya")) {
          return text.replace(/\b(\d+)\b/g, "Rp $1");
        }
        return text;
      },
      payment: (text) => {
        // Add payment security assurance
        if (text.includes("pembayaran") && !text.includes("aman")) {
          return text + " Semua transaksi dijamin aman dan terpercaya.";
        }
        return text;
      },
    };

    const refinement = categoryRefinements[category];
    if (refinement) {
      answer = refinement(answer);
    }

    // General Indonesian language improvements
    answer = answer
      // Improve politeness
      .replace(/\bsaya\b/gi, "saya")
      .replace(/\banda\b/gi, "Anda")
      // Fix common spacing issues
      .replace(/\s+/g, " ")
      .replace(/\s+([.,!?])/g, "$1")
      // Ensure proper sentence ending
      .trim();

    if (answer && !answer.match(/[.!?]$/)) {
      answer += ".";
    }

    return answer;
  }
}

module.exports = BrainService;