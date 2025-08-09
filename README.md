## Fitur
- Streaming batch dengan multiprocessing, hemat memori.
- Checkpoint otomatis untuk resume proses.
- Animasi progress bar interaktif dengan persentase, estimasi waktu, dan kecepatan.
- Reset checkpoint otomatis setelah seed phrase ditemukan.
- Output hasil dengan format rapi.
---
## Persyaratan
- Python 3.8+
- Paket Python berikut (install via pip):
```
git clone https://github.com/erwindobp98/recovery.git
cd recovery
```
```
pip install mnemonic bip-utils eth-account colorama
```
## Cara Menjalankan
1. Siapkan partial phrase di dalam file recover.py pada variabel partial_phrase
Gunakan tanda ? untuk kata yang hilang, contoh:
```
partial_phrase = "also naive cream notice lounge mask olympic grocery slush arrive spoil ?"
```
2. Masukkan target address di variabel target_address (harus dalam lowercase), contoh:
```
target_address = "0x650b26e4e63abb7131424b46a68c6503c332bae1"
```
3. Jalankan skrip dengan perintah:
```
python recover.py
```
4. Skrip akan menampilkan progress interaktif di terminal dan secara otomatis menyimpan checkpoint ke file recover_checkpoint.json.
5. Jika seed phrase ditemukan, skrip akan menampilkan hasil lengkap dan menghapus checkpoint agar run berikutnya mulai dari awal.
6. Contoh output
```
=== OPTIMIZED BIP39 RECOVERY TOOL ===
[INFO] CPU: 2 logical, 1 physical cores
[INFO] RAM: 5.9GB available / 7.8GB total
[AUTO-BENCHMARK] Finding optimal configuration...
[BENCHMARK] Testing 1 workers, batch 4096...
[BENCHMARK] Rate: 14970.6 combinations/sec
[BENCHMARK] Testing 2 workers, batch 2048...
[BENCHMARK] Rate: 42475.7 combinations/sec
[BENCHMARK] Testing 2 workers, batch 8192...
[BENCHMARK] Rate: 35922.1 combinations/sec
[AUTO-BENCHMARK] Optimal: 2 workers, batch 2048 (42475.7/sec)
[CONFIG] Workers: 2, Batch: 2048, Pipeline: 4
[INFO] Missing positions: [2, 11]
[INFO] Total combinations: 4,194,304
[INFO] Starting from offset 0, checked: 0
[███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  19.82% │ 831,488/4,194,304 │ 3066.3/s │ ⏱️4m31s │ ETA 18m16s │ 👥2 
============================================================
[COMPLETE] Processed 835,486 combinations in 4m33s
[PERFORMANCE] Average rate: 3054.1 combinations/sec

 SUCCESS: WALLET RECOVERED! 
Seed phrase : also naive cream notice lounge mask olympic grocery slush arrive spoil victory
Address     : 0xf7c1e38dbc725a69d18922a1ce0c3c0ae62c154d
Private key : 307d3213389afcf397a13f610c400545e466eccfb71b134c720679daea34966d

⚠️  IMMEDIATELY verify this on a secure/offline machine!
