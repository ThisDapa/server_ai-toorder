const productSchema = {
  type: 'object',
  properties: {
    name: { type: 'string' },
    code: { type: 'string' },
    price: { type: 'string' },
    stock: { type: 'number' },
    desc: { type: 'string' }
  },
  required: ['name', 'code', 'price', 'stock', 'desc']
};

const askSchema = {
  type: 'object',
  properties: {
    name_store: {
      type: 'string'
    },
    question: {
      type: 'string'
    },
    whatsapp_number: {
      type: 'string'
    },
    product_data: {
      type: 'array',
      items: productSchema
    }
  },
  required: ['name_store', 'question', 'whatsapp_number', 'product_data']
};

module.exports = { askSchema }