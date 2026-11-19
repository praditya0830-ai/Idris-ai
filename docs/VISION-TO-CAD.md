# Vision: Sketch → CAD, Foto → CAD, part hilang dari foto (cabang `solver/vision-to-cad`)

Tiga rute baru, semuanya lewat modul `vision_ext.py` (didaftarkan di `http_ext.MODULES`, jadi `main.py`
tidak menambah satu baris rute pun). `/vision/read` dan `/vision/job` masuk `http_ext.AI_POSTS`,
sehingga ikut kena batas laju per IP dan penghitung biaya #10 seperti `/chat`.

## Aturan yang menentukan semuanya

Gambar memberi **bentuk dan proporsi**, bukan ukuran. Milimeter hanya boleh datang dari:

1. angka yang **tertulis** di gambar (`dims_written`, `text_seen` tidak kosong) dan dikonfirmasi
   engineer di Studio (`confirmed_dims`), atau
2. satu **ukuran acuan** dari engineer (`reference: {what, mm}`).

Kalau tidak ada keduanya: `scale.set = false`, `mm_per_unit = null`, label
`Scale not set - proportions from image`, dan `/vision/job` menolak memulai desain sampai engineer
mencentang `accept_provisional`. Dalam mode itu jawabannya membawa `fea_allowed: false` dengan alasan
"a provisional scale makes a safety factor a made-up number" — FEA tidak ditawarkan.

Angka yang tidak lolos aturan di atas **dibuang**, bukan ditampilkan: `sanitize()` menghapus setiap key
berbau mm di luar `dims_written`, membuang proporsi > 1 (itu size, bukan rasio), membuang
`dims_written` tanpa `text_seen`, dan menghitungnya di `invented_removed` supaya Studio bisa bilang
"3 angka karangan dibuang".

## Rute

| rute | isi |
|---|---|
| `POST /vision/read` | `{image_base64, mime, mode, note?, reference?, region?}` → `interpretation`, `missing_part`, `questions` (maks 3), `readings`, `scale`, `invented_removed`, `image` (resized / metadata_stripped), `read_by` |
| `POST /vision/job` | body yang sama + `confirmed_dims`, `reading`, `accept_provisional` → `job_id` + `poll`, atau `422 blocked: "interfaces" \| "scale"` dengan yang harus diukur/diisi |
| `POST /vision/similarity` | `{model_silhouette, image_silhouette}` — poligon 0..1 **atau** bitmap baris 0/1 → `iou_percent`, `only_model_percent`, `only_image_percent`, `biggest_region`, `label` |
| `GET /vision` | kontraknya sendiri (mode, batas, label skala) supaya Studio tidak perlu membaca kode |

`region` = `{x, y, w, h, unit: "fraction"}`: area yang dilingkari engineer di foto part hilang; ikut
dikirim ke AI sebagai "look only there".

## Gambar

`prepare()`: batas **10 MB**, tipe diperiksa dari byte (bukan dari `mime` kiriman), **EXIF/GPS dan
komentar dibuang** (PNG: `tEXt/iTXt/zTXt/eXIf/tIME`; JPEG: `APP1..APP15` dan `COM`), lalu diperkecil ke
sisi terpanjang **≤ 1600 px**.

- Dengan Pillow terpasang: resize + re-encode, semua metadata hilang.
- Tanpa Pillow: metadata tetap dibuang dengan pengurai marker sendiri; gambar yang lebih dari 1600 px
  **ditolak dengan alasan**, tidak dikirim ke AI dalam ukuran besar.
- Catatan: `main.py` menolak body > 8 MB sebelum rute ini jalan, jadi dalam praktik gambar ~5,8 MB
  ke atas sudah dijawab 413 oleh server. Batas 10 MB di sini adalah batas logisnya.

## Bagaimana konteks sampai ke penulis kode (V2)

`cad_codegen.py` **tidak diubah** di PR ini. `vision.job_request()` menyusun kalimat permintaan yang
dipakai job desain sebagai `message`, dan kalimat itu memuat: objek + fitur, ukuran yang dikonfirmasi,
tafsiran yang dipilih engineer, bentuk dari gambar (proporsi/simetri/unclear), label skala, dan aturan
"proporsi hanya menjadi milimeter lewat skala". `vision.job_context()` mengirim `context.vision` yang
sama untuk pemakai lain (UI/tes): `confirmed`, `scale`, `reading`, `fea_allowed`, `fea_refused_because`.

Alasan memilih jalan ini: `cad_codegen.py` 50 KB dan setiap penyuntingan berkas sebesar itu mahal dan
rawan; aturan yang sama tetap sampai ke model lewat satu tempat yang sudah diuji.

## Part hilang

`missing_part.interfaces_needed` adalah daftar yang **harus diukur engineer** (jarak lubang baut,
diameter poros, lebar dudukan). `/vision/job` menolak memulai desain selama ada item yang belum masuk
`confirmed_dims` — foto tidak cukup untuk ukuran pas. `loads_guess` dibersihkan dari semua angka
(`loads_numbers_removed: true`), karena beban hanya boleh datang dari engineer.

## Kemiripan bentuk (V3)

`similarity()` meraster dua siluet (256×256 bawaan, `size` bisa diminta 64–512), menghitung IoU, lalu
mencari blok ketidakcocokan terbesar pada grid 32×32 (4-terhubung) untuk dilaporkan sebagai kotak
persen. Angkanya diukur dari outline yang dikirim, bukan diklaim AI, dan jawabannya selalu menyebut
"silhouette comparison; never a 3D claim".

## Yang diuji (angka sebenarnya)

```
PYTHONPATH=$(pwd) python -m pytest -q tests/test_vision_read.py tests/test_vision_similarity.py \
  tests/test_routes.py tests/test_ai_budget.py tests/test_design_job.py tests/test_shape_objectives.py \
  tests/test_cad_codegen.py
-> 93 passed (vision: 31)
```

Isi tes vision: angka tertulis masuk dan yang karangan dibuang; foto tanpa acuan → skala tidak diset +
label + job diblokir; acuan engineer → label "Scale from: overall width 120 mm (engineer)"; EXIF/GPS
hilang sebelum gambar dikirim; chunk teks PNG hilang; > 10 MB ditolak; bukan PNG/JPEG ditolak; gambar
> 1600 px tanpa Pillow ditolak; mode asing ditolak; `confirmed_dims` sampai ke job; part hilang
diblokir sampai ukuran sambungan diisi (lalu tetap butuh skala); `region` sampai ke prompt; tafsiran
yang dipilih engineer sampai ke job; **kemiripan**: dua siluet identik 100%, lingkaran di dalam persegi
78,5% (π/4), segitiga 50% dari kotak pembatasnya, dua kotak terpisah 0%, bitmap 128×128 diterima dan
disamakan skalanya, poligon dan bitmap bentuk sama menghasilkan angka yang sama.

## Yang **belum** diuji (jujur)

- Tidak ada AI gambar sungguhan: `llm.call_vision` diganti jawaban tiruan. Kualitas prompt dan mutu
  pembacaan sketsa/foto nyata belum pernah dilihat sama sekali.
- Tidak ada foto atau sketsa sungguhan; semua gambar di tes dibuat byte-per-byte di dalam tes.
- Jalur Pillow (resize) tidak pernah jalan di sini karena Pillow tidak terpasang; yang teruji adalah
  jalur tanpa Pillow beserta penolakannya.
- `POST /vision/job` di tes memakai `design_job.submit` tiruan: job desain sungguhan dengan
  `context.vision` belum pernah dijalankan sampai selesai (butuh kunci AI).
- `/health` belum menyebut rute vision (itu akan menulis `main.py` kedua kalinya dari sini). Rute tetap
  dapat ditemukan lewat `GET /vision` dan `http_ext`.
- Commit pertama cabang ini menulis `vision.py` dengan isi placeholder karena kesalahan saya; commit
  berikutnya (`fix(vision)`) menulis berkas yang sebenarnya. Yang diuji adalah isi terakhir.
