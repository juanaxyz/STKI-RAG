# Revisi `README.md` dan `main.ipynb`

Ringkasan perubahan dalam format + ditambah / - diubah.

---

## 1. `README.md`

Sebelum: file kosong (`e69de29`). Sesudah: dokumen panduan 56 baris.

### Ditambahkan
- Judul proyek dan penjelasan 3 pendekatan yang dibandingkan
- Section **Prerequisites**: Python 3.11+, API key Gemini, daftar package
- Perintah instalasi package via `pip`
- Section **Setup**: clone, buat `.env`, siapkan `dokumen/`, cara menjalankan notebook
- Section **Struktur Notebook**: penjelasan Cell 1 sampai Cell 8
- Section **Catatan**: model yang dipakai, fallback lokal, dan catatan varia

### Dihapus
- Tidak ada konten sebelumnya untuk dihapus (file sebelumnya kosong).

---

## 2. `main.ipynb`

Sebelum: 1 markdown cell + 9 code/output cells berantakan.  
Sesudah: 17 cells terstruktur (8 markdown + 8 code + 1 title), semua `execution_count` direset ke `null`, outputs dikosongkan agar fresh saat dijalankan ulang.

### Ditambahkan / Direstrukturisasi

#### Cells Baru
- **Title markdown** — header proyek dan dekskripsi singkat
- **Cell 1 — Setup** — import semua library, load `.env`, validasi API key
- **Cell 2 — Helper Functions** — `_local_fallback_embedding`, `get_embedding`, `cosine_sim`
- **Cell 3 — Load Query and Documents** — load `D1/D2/D3.md` + preview output
- **Cell 4 — Vector Search** — embed dokumen, embed query, cosine similarity, ranking
- **Cell 5 — RAG Answer Generation** — prompt LLM, parse ANSWER/SNIPPET/REASON
- **Cell 6 — 2D Visualization** — PCA 2D plot matplotlib
- **Cell 7 — KL-Divergence Route** — softmax + `kl_divergence`, ranking KL
- **Cell 8 — Summary Comparison** — bandingkan cosine vs KL side-by-side

#### Perbaikan Kode
- Tambah docstring ke semua helper functions
- Tambah typing hints: `dict[str, str]`, `list[float]`, `np.ndarray`
- Tambah `METRIC` variable agar mudah ganti cosine ↔ KL
- Tambah print progres embedding per dokumen
- Validasi API key lebih jelas dengan `EnvironmentError`
- Guard zero-norm di `cosine_sim` agar tidak division by zero
- Normalisasi numerik di `softmax` (`x - np.max(x)`)
- Tambah `eps` guard di `kl_divergence`

### Dihapus / Diubah

#### Plot 3D → 2D
- Plotly 3D visualization dihapus, diganti matplotlib 2D PCA agar lebih ringkas dan mudah dipahami pemula

#### Output Cached
- Semua cached stdout dan image/png output lama dihapus agar cell bisa dijalankan ulang tanpa noise

#### Query Definition Dipindah
- `query='What time do I leave for school?'` dipindah dari posisi tengah ke Cell 3 agar logis

#### Fungsi `answer_query` Di-inline
- Di notebook lama, `answer_query` didefinisikan di tengah kemudian dipanggil
- Di revisi, logic answer generation langsung di Cell 5 agar alur lebih jelas

#### Header Markdown Diperbaharui
- `# Import Library` → `## Cell 1: Setup`
- `# Load API KEY` → `## Cell 2: Helper Functions`
- `# Load Docs` → `## Cell 3: Load Query and Documents`
- `# Definisi Fungsi` → `## Cell 4: Vector Search`
- `# Vector Search` → `## Cell 6: 2D Visualization`
- Header lama yang ambigu diganti deskriptif

#### Dependency yang Disederhanakan
- Import `plotly.graph_objects` dihapus karena tidak dipakai lagi
- Import `httpx` exception classes dihapus, diganti `Exception` agar lebih generik

---

## Ringkasan Perubahan

| Aspek | Sebelum | Sesudah |
|---|---|---|
| README | Kosong | Panduan setup lengkap |
| Struktur notebook | 1 blob linear | 8 cells terstruktur |
| Visualisasi | 3D Plotly | 2D matplotlib PCA |
| Error handling | Parsial | Retry + fallback + guard zero-norm |
| Output | Cached stdout/gambar | Fresh, bisa di-run ulang |
| Dokumentasi kode | Tidak ada | Docstring + typing hints |
| Cara baca | Tidak jelas untuk pemula | Berurutan, cell-by-cell |

---

*Diff mentah tersimpan di:*
- `/tmp/readme_diff.diff`
- `/tmp/notebook_diff.diff`
