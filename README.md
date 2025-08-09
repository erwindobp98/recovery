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
https://github.com/erwindobp98/recovery.git
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
[INFO] Missing positions: [11]
[INFO] Kandidat per posisi: [2048]
[INFO] Total kombinasi: 2,048
[INFO] Resuming from offset 0, checked so far 0
[############################  ]  95.21% checked=1,950/2,048 rate=3253.2/s elapsed=0s ETA=0s                   
[FOUND] Seed phrase ditemukan!
[DONE] elapsed 0.6s - checked 1,950
Seed phrase : also naive cream notice lounge mask olympic grocery slush arrive spoil victory
Address     : 0xf7c1e38dbc725a69d18922a1ce0c3c0ae62c154d
Private key : 307d3213389afcf397a13f610c400545e466eccfb71b134c720679daea34966d

*** Immediately verify manually on an offline/secure machine.
