# BrainService dengan FAISSstore dan BGE-M3

## Pengantar

BrainService telah diperbarui untuk menggunakan FAISSstore dengan LangChain.js dan model BGE-M3 dari Ollama untuk pencarian konteks berbasis vektor. Ini menggantikan implementasi sebelumnya yang menggunakan brain.js untuk jaringan saraf.

## Fitur Baru

- **Pencarian Vektor**: Menggunakan FAISS (Facebook AI Similarity Search) untuk pencarian vektor yang cepat dan efisien
- **Embeddings BGE-M3**: Menggunakan model BGE-M3 dari Ollama untuk menghasilkan embeddings berkualitas tinggi
- **Kompatibilitas**: Mempertahankan API yang sama dengan implementasi sebelumnya

## Persyaratan

- Node.js v16 atau lebih tinggi
- Ollama terinstal dan berjalan di mesin lokal atau server yang dapat diakses
- Model BGE-M3 diunduh di Ollama

## Instalasi

### 1. Instal Ollama

Unduh dan instal Ollama dari [https://ollama.ai/download](https://ollama.ai/download)

### 2. Unduh Model BGE-M3

Setelah Ollama terinstal, unduh model BGE-M3:

```bash
ollama pull bge-m3
```

### 3. Konfigurasi .env

Pastikan file `.env` memiliki konfigurasi yang benar:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDINGS_MODEL=bge-m3
```

## Penggunaan

BrainService mempertahankan API yang sama dengan implementasi sebelumnya, sehingga tidak diperlukan perubahan pada kode yang menggunakan BrainService.

```javascript
const BrainService = require('./src/services/BrainService');

async function example() {
  const brainService = new BrainService();
  await brainService.init();
  
  // Proses konteks
  const context = await brainService.processContext('Bagaimana cara memesan?');
  
  // Dapatkan tag yang diprediksi
  const tags = await brainService.getPredictedTags('Bagaimana cara memesan?');
  
  // Temukan jawaban
  const answer = await brainService.findAnswer('Bagaimana cara memesan?');
  
  console.log(answer);
}

example();
```

## Pemecahan Masalah

### Error: ECONNREFUSED

Jika Anda melihat error "ECONNREFUSED" saat menjalankan BrainService, pastikan:

1. Ollama sedang berjalan
2. URL di .env (`OLLAMA_BASE_URL`) benar
3. Port yang digunakan (default: 11434) tidak diblokir oleh firewall

### Model Tidak Ditemukan

Jika Anda melihat error tentang model tidak ditemukan, pastikan Anda telah mengunduh model BGE-M3:

```bash
ollama pull bge-m3
```

## Migrasi dari brain.js

Jika Anda memiliki model yang dilatih dengan brain.js, Anda perlu membuat ulang model dengan FAISSstore. Gunakan metode `updateVectorStore()` untuk membuat ulang model dari dataset yang ada.

```javascript
await brainService.updateVectorStore();
```

## Performa

FAISSstore dengan BGE-M3 memberikan performa yang lebih baik untuk pencarian konteks dibandingkan dengan implementasi brain.js sebelumnya, terutama untuk dataset yang lebih besar.