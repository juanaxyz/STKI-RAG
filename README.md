# STKI RAG Demo
Demo Retrieval Augmented Generation (RAG) untuk mata kuliah STKI. Project ini membandingkan 3 pendekatan retrieval semantik:

1. Prompt-based (in-context LLM reasoning)
2. Vector search (embedding + cosine similarity)
3. KL-Divergence (softmax probability distribution)

Menggunakan model Google Gemini untuk embedding dan generasi jawaban.

## Prerequisites
- Python 3.11+
- API key Gemini dari [Google AI Studio](https://aistudio.google.com/app/apikey)
- Package yang dibutuhkan:
  - `google-generativeai`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `python-dotenv`

Install package:
```
pip install google-generativeai numpy scikit-learn matplotlib python-dotenv
```

## Setup
1. Clone repo ini
2. Buat file `.env` di root project:
   ```
   GEMINI_API_KEY=key-mu-disini
   ```
3. Pastikan folder `dokumen/` berisi file `D1.md`, `D2.md`, dan `D3.md`
4. Jalankan notebook:
   ```
   jupyter notebook main.ipynb
   ```
   atau
   ```
   python -m notebook main.ipynb
   ```
5. Run cell berurutan dari atas sampai bawah

## Struktur Notebook
- **Cell 1**: Setup — import library, init Gemini client, validasi API key
- **Cell 2**: Helper — fungsi embedding dengan retry/fallback + cosine similarity
- **Cell 3**: Load data — load query dan dokumen dari folder `dokumen/`
- **Cell 4**: Vector search — embed dokumen, embed query, hitung cosine similarity, pilih best match
- **Cell 5**: RAG answer — generate jawaban terstruktur dari document terbaik
- **Cell 6**: Visualisasi 2D — PCA untuk lihat posisi dokumen dan query di embedding space
- **Cell 7**: KL-Divergence route — alternatif measurement menggunakan softmax + KL
- **Cell 8**: Summary comparison — bandingkan hasil cosine vs KL-Divergence

## Catatan
- `gemini-3.1-flash-lite` digunakan untuk answer generation
- `gemini-embedding-001` digunakan untuk embedding vektor
- Jika API gagal, ada fallback lokal berbasis hash token (tidak semantik, tapi demo tetap jalan)
- Hasil ranking bisa berbeda tergantung model dan query yang dipakai
