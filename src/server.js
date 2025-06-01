(async () => {
  require('dotenv').config();
  const Fastify = require('fastify');
  const fastifyHelmet = require('@fastify/helmet');
  const fastifyCors = require('@fastify/cors');
  const logger = require('./utils/logger');
  const questionRoutes = require('./routes/questionRoutes');
  const errorHandler = require('./middleware/errorHandler');

  const PORT = process.env.PORT || 3000;
  const fastify = Fastify({
    connectionTimeout: 600000,
    keepAliveTimeout: 600000
  });

  // await fastify.register(fastifyHelmet);
  // await fastify.register(fastifyCors);
  // await fastify.register(require('@fastify/formbody'));

  fastify.addHook('onRequest', (request, reply, done) => {
    logger.info(`${request.method} ${request.url} - ${request.ip}`);
    done();
  });

  await fastify.register(questionRoutes, { prefix: '/api' });

  fastify.get('/health', async (request, reply) => {
    return {
      status: 'OK',
      timestamp: new Date().toISOString(),
      uptime: process.uptime()
    };
  });

  fastify.setErrorHandler((error, request, reply) => {
    errorHandler(error, request, reply);
  });

  fastify.setNotFoundHandler((request, reply) => {
    reply.status(404).send({ error: 'Route not found' });
  });

  try {
    await fastify.listen({ port: PORT, host: '0.0.0.0' });
    logger.info(`AI Server running on port ${PORT}`);
    logger.info(`Environment: ${process.env.NODE_ENV}`);
  } catch (err) {
    logger.error(err);
    process.exit(1);
  }

  // Graceful shutdown
  process.on('SIGTERM', () => {
    logger.info('SIGTERM received, shutting down gracefully');
    fastify.close(() => process.exit(0));
  });

  process.on('SIGINT', () => {
    logger.info('SIGINT received, shutting down gracefully');
    fastify.close(() => process.exit(0));
  });

  module.exports = fastify;
})();