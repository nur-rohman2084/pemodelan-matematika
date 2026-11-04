# Audit konsistensi konfigurasi RDA — u100/v100, September 2026

## Konfigurasi standar

- Angin ERA5: u100 (parameter GRIB 246) dan v100 (parameter GRIB 247); tidak menggunakan u10/v10 sebagai input eksperimen standar.
- Sumber angin: `data_1.grib`, hourly, 1–30 September 2026.
- Periode simulasi: 1 September 2026 00:00 UTC sampai 30 September 2026 23:00 UTC (30 x 24 = 720 langkah per jam).
- Kelembapan validasi: ERA5 `swvl1` (parameter 39), `data_c11eca.grib`, periode September.
- Hotspot: VIIRS NOAA-20, difilter untuk periode September, confidence Medium/High dan domain model.
- Domain validasi: lintang -3 sampai -1 derajat, bujur 111 sampai 116 derajat.
- Grid validasi: 0,03 derajat.

## Berkas kode

- `rda_karhutla_lengkap.ipynb`: alur utama, termasuk eksperimen sintetis dan validasi v2. Eksperimen sintetis memakai `u100/v100`; validasi v2 mengkalibrasi koefisien adveksi `kappa` bersama parameter RDA.
- `rda_karhutla.py`: program simulasi numerik mandiri; default input dan periode diarahkan ke u100/September.
- `validasi_hotspot.py`: skrip validasi dasar terpisah; juga diarahkan ke u100 dan 30 hari, tetapi bukan implementasi v2 yang memakai kelembapan dan suku penyalaan nonlokal.

## Catatan penting sebelum memakai angka dalam artikel

1. Hasil di folder `hasil/` pada commit sebelumnya dihitung dengan konfigurasi lama (validasi 29 hari; eksperimen sintetis menggunakan u10 dan tanggal Agustus). Angka, tabel, dan gambar lama **belum menjadi hasil konfigurasi baru**.
2. Jalankan ulang notebook utama dari awal sampai akhir dengan seluruh berkas masukan tersedia. Regenerasikan `hasil/hasil_sintetis.json`, `hasil/hasil_validasi.json`, `hasil/kalibrasi.csv`, `hasil/tabel_metrik.csv`, dan seluruh gambar sebelum angka atau gambar dimasukkan ke artikel.
3. Skrip `validasi_hotspot.py` adalah eksperimen validasi dasar tanpa kelembapan tanah dan tanpa suku penyalaan baru. Jangan mencampur metriknya dengan hasil validasi v2 dari notebook.
4. Koefisien adveksi angin `kappa` sekarang menjadi bagian grid search validasi; jangan menggunakan hasil kalibrasi lama yang dibuat saat `kappa` dipatok nol.\n5. Pada notebook utama, mask gambut validasi masih seragam (`G=1)) karena CSV gambut yang tersedia hanya berisi atribut tanpa geometri. Eksperimen sintetis memakai mask gambut sintetis. Keduanya tidak boleh ditulis seolah-olah memakai peta gambut aktual.
6. Hotspot VIIRS merupakan deteksi anomali termal, bukan batas langsung luas area terbakar. Istilah luas observasi perlu dijelaskan sebagai luas sel hotspot kumulatif setelah rasterisasi dan dilasi satu sel.
7. Model ini adalah model numerik berbasis persamaan RDA yang diselesaikan dengan finite difference (Euler eksplisit, adveksi upwind dan difusi beda pusat), bukan model machine learning.

## Pemeriksaan waktu

Dengan `ndays=30`, rentang indeks jam yang dipakai adalah 0–719. Data hourly harus mencakup timestamp 1 September 00:00 UTC hingga 30 September 23:00 UTC. Pastikan file ERA5 dan hotspot benar-benar memiliki cakupan tersebut sebelum menjalankan eksperimen.
