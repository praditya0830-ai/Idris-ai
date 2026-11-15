# Tahap A — kartu varian di drawer Studio

Catatan untuk tim UI dan tim solver. Pasangan dari
[`Idris-studio/docs/studio-variants-drawer.patch`](https://github.com/praditya0830-ai/Idris-studio/blob/main/docs/studio-variants-drawer.patch).

## Yang ternyata sudah ada

`/generate-variants` dan `/edit-variant` **sudah tersambung** di Studio sebelum perubahan ini:

| Pemanggil | Lokasi | Endpoint |
|---|---|---|
| `run()` | `index.html` baris 2262 | `POST /generate-variants` |
| `editVariant()` | `index.html` baris 3786 | `POST /edit-variant` (termasuk `allowed_range`, `diff`, `before`/`after`) |

Ditambah `editCard()` (slider per parameter) dan drag-handle 3D yang keduanya memakai `/edit-variant`.
Jadi tidak ada kabel yang perlu dipasang ulang di sana.

## Yang benar-benar kosong

Drawer bawah (`#simBody`, tab *Gear check*) hanya menampilkan "select a variant to see its results"
ketika varian sudah dihitung tetapi belum ada yang dipilih. Itu terlihat seperti jalan buntu.

## Perubahan

`renderSim()` sekarang punya tiga keadaan:

1. **Belum ada studi** — tetap pesan kosong seperti sebelumnya (tidak berubah).
2. **Ada varian, belum ada yang dipilih** — grid kartu, satu kartu per varian, memakai helper
   `vcard()` dan CSS `.sim-grid` / `.rcard` yang sudah ada (tanpa CSS baru). Tiap kartu memuat
   bar Physics/Design, fakta desain, pil PASS/FAIL dari FoS minimum, serta penanda
   `paling seimbang` dan `desain terpilih`. Klik kartu → `selectVariant(i)`.
3. **Ada varian terpilih** — persis seperti sebelumnya.

Ukuran perubahan: 23 baris ditambah, 2 dibuang, seluruhnya di dalam `renderSim()`.

## Cara menerapkan

Repo `Idris-studio` belum bisa menerima file 833 KB lewat API, jadi patch diterapkan dengan salah satu:

- `git apply docs/studio-variants-drawer.patch` dari root repo `Idris-studio`, atau
- tempel manual: cari `if (!v) { sub.textContent = state.variants.length` di `index.html`,
ganti dua baris itu dengan blok di dalam patch.

Commit message yang dipakai: `feat(studio): drawer varian menampilkan kartu tiap varian`.

## Status verifikasi

| Cek | Hasil |
|---|---|
| `node --check` pada 4 blok `<script>` inline | lolos |
| Halaman dimuat di browser (port 3000) | boot normal, "Engine online", solver 0.35.0, tanpa error JS |
| Cabang "belum ada studi" | terbukti live di DOM (`dockSub` = "no study yet") |
| Cabang kartu varian | **belum terbukti** — butuh satu pencarian desain nyata; alur "Gearbox demos" tidak terbuka saat diotomasi |
