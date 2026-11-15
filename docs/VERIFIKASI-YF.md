# Verifikasi YF / Y_Lewis — hasil investigasi 3 kegagalan `tests/test_validation.py`

Status: **rumus tidak diubah**. Tiga baris benchmark yang salah, bukan `physics/toothform.py`.
Diperiksa 15 November 2026.

## Gejala awal

`python3 tests/test_validation.py` → 16 PASS, 3 FAIL:

| Uji | Hasil | Referensi | Selisih |
|---|---|---|---|
| `YF z=400` batas rack (analitik) | 2,097 | 2,229 | −5,95 % |
| `1/Y_Lewis vs YF, z=17` | 2,959 | 3,300 | −10,35 % |
| `1/Y_Lewis vs YF, z=24` | 2,661 | 2,967 | −10,33 % |

## Bukti: implementasi geometri benar

`tooth_form()` (trochoid hasil generate rack) dibandingkan dengan rumus analitik ISO 6336-3
(`iso_closed_form()` di modul yang sama):

| z | YF geometri | YF ISO | selisih |
|---|---|---|---|
| 16 | 3,0279 | 3,0263 | +0,05 % |
| 17 | 2,9589 | 2,9575 | +0,05 % |
| 24 | 2,6608 | 2,6605 | +0,01 % |
| 50 | 2,3338 | 2,3330 | +0,04 % |
| 100 | 2,1958 | 2,1953 | +0,02 % |
| 400 | 2,0966 | 2,0959 | +0,03 % |

Konvergensi terhadap jumlah titik fillet juga sehat: z=400 memberi 2,09799 (n=200),
2,09658 (n=600), 2,09592 (n=8000) — default n=600 berada 0,03 % dari nilai konvergen.

## Temuan 1 — dua baris membandingkan besaran yang berbeda (salah definisi)

Baris `1/Y_Lewis vs YF` berlabel "same physics, tabulated vs computed". Premis ini keliru:

- **Y_Lewis** (Shigley tabel 14-2) memakai balok parabola dengan beban di ujung gigi, tanpa
  konstruksi garis singgung 30° dan tanpa proyeksi sudut beban.
- **Y_F** (ISO 6336-3) memakai penampang kritis di titik singgung 30° dan lengan momen
  `hFe` yang diukur tegak lurus garis kerja beban.

Keduanya bukan besaran yang sama, jadi selisih 10 % di z kecil adalah perilaku yang benar.
Rasio YF / (1/Y_Lewis) mengecil sesuai teori:

| z | 17 | 24 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| rasio | 0,896 | 0,897 | 0,954 | 0,982 | 0,988 | 1,006 |

Konvergen ke 1 saat z → ∞. Uji dengan toleransi 8 % di z=17 tidak bisa dipenuhi oleh YF ISO
yang benar mana pun — toleransi itu tidak salah longgar, referensinya yang salah besaran.

## Temuan 2 — konstanta batas rack di dalam uji salah hitung

Rumus tangan di dalam uji mengukur tinggi penampang dari ujung gigi
(`h_rack = 1 − (−H_F0 + ρ + ρ sin(−30°)) = 2,060`), bukan jarak tegak lurus dari penampang
kritis ke garis kerja beban seperti definisi ISO. Akibatnya `hFe` kelebihan sekitar 8 %
(2,060 vs 1,903) dan YF batas rack jadi 2,229 alih-alih ~2,096 — deviasi 6 % yang muncul
sebagai "kegagalan rumus". `sFn` juga sedikit meleset (2,355 vs 2,329).

## Perbaikan

`tests/test_validation.py`:

1. Batas rack kini diuji terhadap `iso_closed_form(400)` dengan toleransi **0,5 %** (lebih ketat
   dari 3 % sebelumnya) — referensi yang sama dengan blok lain.
2. Ditambah `YF z=17/24/50 vs ISO closed form`, toleransi **0,5 %**, menggantikan perbandingan
   terhadap Y_Lewis.
3. Hubungan Lewis↔ISO diuji sebagai **rentang** yang bermakna secara fisika: rasio 0,85–0,92 di
   z=17, dan 0,98–1,02 di z=400 (konvergensi di limit rack). Helper `check_range()` ditambahkan.
4. Penghitung kegagalan diperbaiki: `bad` dulu hanya menjumlahkan blok pertama dan 8 baris
   terakhir, sehingga baris di tengah (termasuk dua `Z_eps`) tidak pernah memengaruhi exit code.
   Sekarang seluruh baris dihitung dan totalnya dicetak.

Hasil: `tests/test_validation.py` → **seluruh benchmark PASS**; 30/30 skrip non-jaringan di
`tests/` lolos. Tidak ada perubahan pada `physics/`, `solvers/`, atau angka desain apa pun.
