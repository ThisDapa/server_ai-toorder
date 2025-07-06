/**
 * Main server file for AI Question Processing System
 * Initializes and configures Fastify server with routes and middleware
 */

'use strict';

require('dotenv').config();
const Fastify = require('fastify');
const fastifyHelmet = require('@fastify/helmet');
const fastifyCors = require('@fastify/cors');
const fastifyFormBody = require('@fastify/formbody');
const logger = require('./utils/logger');
const questionRoutes = require('./routes/questionRoutes');
const errorHandler = require('./middleware/errorHandler');

// Server configuration constants
const PORT = process.env.PORT || 3000;
const HOST = '0.0.0.0';
const TIMEOUT = 600000; // 10 minutes

/**
 * Initialize and start the server
 */
async function startServer() {
  // Create Fastify instance with configuration
  const fastify = Fastify({
    connectionTimeout: TIMEOUT,
    keepAliveTimeout: TIMEOUT,
    logger: false // Using custom logger instead
  });

  // Register plugins
  await registerPlugins(fastify);
  
  // Configure request logging
  setupRequestLogging(fastify);
  
  // Register routes
  await fastify.register(questionRoutes, { prefix: '/api' });
  
  // Setup health check endpoint
  setupHealthCheck(fastify);
  
  // Configure error handling
  setupErrorHandling(fastify);

  // Start the server
  try {
    await fastify.listen({ port: PORT, host: HOST });
    logger.info(`AI Server running on port ${PORT}`);
    logger.info(`Environment: ${process.env.NODE_ENV}`);
  } catch (err) {
    logger.error(`Failed to start server: ${err.message}`);
    process.exit(1);
  }

  // Setup graceful shutdown
  setupGracefulShutdown(fastify);
  
  return fastify;
}

/**
 * Register Fastify plugins
 */
async function registerPlugins(fastify) {
  await fastify.register(fastifyHelmet);
  await fastify.register(fastifyCors);
  await fastify.register(fastifyFormBody);
}

/**
 * Setup request logging
 */
function setupRequestLogging(fastify) {
  fastify.addHook('onRequest', (request, reply, done) => {
    logger.info(`${request.method} ${request.url} - ${request.ip}`);
    done();
  });
}

/**
 * Setup health check endpoint
 */
function setupHealthCheck(fastify) {
  fastify.get('/health', async () => {
    return {
      status: 'OK',
      timestamp: new Date().toISOString(),
      uptime: process.uptime()
    };
  });
}

/**
 * Setup error handling
 */
function setupErrorHandling(fastify) {
  fastify.setErrorHandler((error, request, reply) => {
    errorHandler(error, request, reply);
  });

  fastify.setNotFoundHandler((request, reply) => {
    reply.status(404).send({ 
      success: false,
      error: 'Route not found' 
    });
  });
}

/**
 * Setup graceful shutdown handlers
 */
function setupGracefulShutdown(fastify) {
  const shutdownGracefully = (signal) => {
    logger.info(`${signal} received, shutting down gracefully`);
    fastify.close(() => process.exit(0));
  };

  process.on('SIGTERM', () => shutdownGracefully('SIGTERM'));
  process.on('SIGINT', () => shutdownGracefully('SIGINT'));
}

// Start the server and export the instance
const server = startServer();
module.exports = server;