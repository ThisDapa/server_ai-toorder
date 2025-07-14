if (typeof fetch === "undefined") {
  global.fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
}
fetch('http://localhost:11434/api/embeddings', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({model: 'bge-m3', prompt: 'hello world'})
})
  .then(res => res.json())
  .then(console.log)
  .catch(console.error);