# Kontrak API — IDRIS Design Studio ⇄ ACE Engine

Dokumen ini adalah satu sumber kebenaran untuk sambungan UI (`Idris-studio/index.html`)
ke backend (`Idris-ai`). Disusun dari tiga sumber yang bisa diperiksa ulang:

| Sumber | Cara memeriksa |
|---|---|
| `/health` live | `curl https://idris-ai-rwd6.onrender.com/health` |
| Dispatcher backend | `main.py` (baris 151–160 GET, 637–641 POST) dan `ace_core/analysis.py` |
| Pemanggil di UI | `grep -oE "fetch\([^)]*" Idris-studio/index.html` |

- Base URL yang dipakai UI: `const API = qs.get('api') || 'https://idris-ai-rwd6.onrender.com'`
  (bisa di-override lewat `?api=` untuk backend lokal).
- Backend: `idris-core-backend`, `solver_version 0.35.0`, 27 solver ACE terdaftar,
  `occt: cadquery-ocp 8.0.1.1.0` tersedia, `step_mode_default: faceted`.
- Batas body request: 8 MB (di atas itu `413`).

## 1. Protokol job (berlaku untuk semua analisis)

Semua analisis (`/fea`, `/modal`, `/cfd`, `/review`, `/part-check`, `/weldment-check`)
memakai pola yang sama, dan UI sudah menanganinya (`feaPoll`):

```
POST /<endpoint>  body JSON
  -> 202 {"success":true,"status":"queued","job_id":"<id>","stage":"Queued","progress":0.0}
     (atau 200 langsung, bila hasilnya bisa dihitung seketika)
GET  /<endpoint>/<job_id>
  -> 200 {"success":..,"status":"queued|running|done|failed","stage":"..","progress":0.0-1.0}
     status "done" menambahkan hasil; "failed" menambahkan pesan galat
```

UI melakukan polling tiap ±2 detik dan memakai `stage`/`progress` untuk telemetri.
Setiap stage baru di backend otomatis muncul di UI — tidak perlu kerja UI tambahan.

## 2. Endpoint yang sudah dipanggil Studio (14/14 ada di backend)

| Endpoint | Dipakai untuk | Bukti di backend |
|---|---|---|
| `GET /health` | status `Engine: connected / connecting…` | main.py:169 |
| `POST /chat` | percakapan AI → requirements / `cad` JSON | main.py:638 |
| `POST /fea` + `GET /fea/<id>` | tab Simulate (analisis housing per varian) | main.py:152, analysis.py:3 |
| `POST /modal` + polling | frekuensi alami | main.py:152 |
| `POST /cfd` + polling | analisis termal | main.py:152 |
| `POST /review` + polling | tab Evaluate | main.py:152 |
| `POST /cad/review` | "AI mata": periksa hasil CAD dari gambar | main.py:639 |
| `POST /catalog/bolt-check` | cek baut ↔ lubang saat auto-assembly | main.py:639 |
| `GET /design-methods` | info metode desain | main.py:317 |
| `GET /references` | pilihan referensi (Lewis / ISO 6336) | main.py:324 |
| `POST /generate-housing` | geometri housing | main.py:638 |
| `POST /generate-assembly` | assembly + BOM (+STL per part) | main.py:638 |
| `POST /generate-variants` | varian gearbox ter-skor | main.py:638 |
| `POST /cad/explore` | sweep parameter → varian | main.py:639 |
| `GET /download/<file>` | unduh STL/BOM/drawing | main.py:368 |

Catatan: pesan di UI `"Solver belum punya POST /cad/explore"` **sudah tidak berlaku** —
endpoint itu ada dan menjawab `200`. Pesan itu perlu diperbarui agar tidak menyesatkan.

## 3. Kemampuan backend yang BELUM dipakai UI

Ini yang sebenarnya menutup daftar "fitur wajib" di analisis fitur — bukan kode baru.

| Endpoint | Isinya | Menutup kebutuhan |
|---|---|---|
| `POST /part-check` | FEA statis untuk **part apa pun**: `{positions, indices}` atau `{stl_base64}`, `material`, `supports:[{where}]`, `loads:[{force_N, where}]`, `convergence_study` → job → kontur von Mises, displacement, **safety factor** | Boundary condition per-face + kontur + FoS di tab Simulate |
| `GET /library`, `GET /library/part?id=hex_bolt&size=M6&length=20` | library part mekanis (ISO 4762/4017/4032/7089, bearing, DIN 6885 key, ISO 3601 O-ring, spacer) → mengembalikan `cad` (kode replicad) yang dibangun browser | Tombol **Library** |
| `GET /catalog`, `/catalog/fastener?size=M6`, `/catalog/bearings?d_min=8`, `/catalog/hole?d=5.5` | data part standar + sumber (ISO 273 clearance, ISO 261/2306, dst.) | Properties part standar |
| `POST /edit-variant` | ubah satu parameter varian → varian baru ter-cek ulang + diff + `allowed_range` | **Slider parametrik** (batas aman slider datang dari `allowed_range`) |
| `POST /weldment-check` | pemeriksaan sambungan las (job) | Evaluate tahap lanjut |
| `GET /design-part?q=kubus%2050%20mm` | kalimat → part sederhana + STEP/STL + volume + massa | cadangan Text-to-CAD tanpa LLM |
| `GET /solvers` | katalog solver: fidelity, schema, benchmark, limitations | **Audit log deterministik** (tab Evaluate) |
| `GET /solvers/tools` | 27 function declaration untuk LLM | tool-calling di chat |
| `POST /solve/<name>` | jalankan satu solver ACE langsung | lembar rumus transparan (ISO 6336, dsb.) |
| `GET /verify`, `GET /selfcheck` | benchmark vs nilai referensi; 3 skenario demo ujung-ke-ujung (background, cached) | bukti kredibilitas V&V |
| `GET /occt/selftest` | uji kernel OpenCascade (import, solid eksak, STEP write/read) | status kernel di UI |
| `GET /chat/providers`, `/chat/ping?provider=`, `/chat/selftest` | provider AI mana yang aktif + diagnostik | panel setelan AI |
| `GET /part-view`, `GET /cad/try` | halaman uji part; QA satu kalimat | QA internal |
| `POST /cad/rank` | peringkat CAD | pelengkap explore |
| `GET /materials` | tabel material simulator (`E`, `nu`, `Sy`, `rho`) — satu sumber untuk UI dan laporan | dropdown material di tab Simulate |
| `POST /report` | paket rekayasa HTML siap cetak: requirements, varian, cek, analisis, review, BOM, bukti solver → print to PDF | tombol **Export PDF** |

## 4. Pemetaan rekomendasi → endpoint yang sudah ada

| Rekomendasi | Status sebenarnya | Tindakan benar |
|---|---|---|
| Sambungkan ke **FastAPI** | Backend adalah `http.server`; `requirements.txt` hanya numpy, scipy, `cadquery-ocp` opsional | Jangan migrasi framework. UI hanya butuh JSON — arahkan ke endpoint yang ada |
| **PINNs / meshless FEA** | `ace_core/contract.py`: solver deterministik; PINN/FNO belum diintegrasikan; FEM berbasis mesh (`fem2d`, `fem3d`, `shell_fea`, `part_fea`) | Pakai `fea_*` / `part-check`. Jangan tampilkan klaim PINN/meshless |
| Benchmark **NASA SCL** | Yang ada: `nafems_le1`, benchmark Shigley/AGMA/ISO | Jangan tayangkan NASA SCL sebelum benchmark-nya ada |
| Offline-local mode | `kernel/` sudah memuat OpenCascade WASM (replicad 1.2.1), dimuat hanya saat tool part dipakai | Ubah *kapan* kernel jadi jalur utama saat backend cold, bukan tambah kernel |
| DFM heatmap per-permukaan | `POST /review` mengembalikan temuan DFM **tekstual** (interference, fits, DFM, assembly) | Satu-satunya kerja UI baru: memetakan temuan ke penyorotan face |

## 5. Rencana menyambungkan (bukan membangun ulang)

### Tahap A — Drawer varian terisi
1. `POST /generate-variants` dengan requirements hasil `/chat` (UI sudah menyimpan `v.housing_params`, `v.design`, `v.shaft`, `v.materials`).
2. Render kartu: berat, FoS, biaya — field skor sudah dikirim backend.
3. Klik kartu → muat geometri (`/generate-assembly`) dan hasil analisis yang tersimpan.
4. Slider Properties → `POST /edit-variant` (kirim satu parameter), lalu pakai `allowed_range` sebagai min/max slider dan `diff` sebagai penanda perubahan.

Sebab drawer kosong saat ini: `/cad/explore` dipanggil tanpa parameter, sehingga balasannya
`variants: []`, `ranking: []`, `notes: ["No parameters…"]`. Butuh requirements dulu, bukan endpoint baru.

### Tahap B — Simulate per-face (MVP FEA part)
1. Raycast face di canvas → kumpulkan `positions`/`indices` mesh yang sedang ditampilkan.
2. Fixed support: `supports: [{where: <face/vertex set>}]`; Force: `loads: [{force_N: N, where: …}]`.
3. Material selector → `material` (Al 6061-T6, Steel AISI 4140, Ti-6Al-4V, ABS/Nylon).
4. `POST /part-check` → polling → kontur von Mises + displacement + FoS dipetakan ke vertex warna.
5. Legend bar dan badge FoS dibaca dari respons, bukan dihitung di UI.

### Tahap C — Evaluate
1. DFM temuan dari `/review` → daftar temuan + sorotan face (kerja UI).
2. Audit deterministik: ambil `method`, `benchmarks[].source`, `limitations` dari `GET /solvers`;
   untuk rumus per-kasus panggil `POST /solve/<name>` dan tampilkan `checks[]` + `inputs` + `elapsed_ms`.
3. BOM: `GET /download/<bom csv>` dari `/generate-assembly`; report PDF menyusul (kerja baru).

## 6. Yang belum diverifikasi di dokumen ini

- Kebenaran fisika tiap solver (baru kontrak dan katalognya yang diperiksa, bukan hasil angkanya).
- `GET /verify` dan `GET /selfcheck` tidak dijalankan (keduanya memicu benchmark berat di server produksi).
- Payload detail `/part-check` (`where` untuk support/load) perlu dibaca dari `physics/part_fea.py`
  sebelum implementasi Tahap B.
