# BIP39 Seed Phrase Recovery
Skrip ini digunakan untuk melakukan **multiprocessing streaming recovery** frase seed BIP39 dengan hingga 4 kata yang hilang.  
Cocok untuk mencoba kombinasi kata hilang tanpa perlu menyimpan daftar kombinasi besar sekaligus.
---
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

```bash
pip install mnemonic bip-utils eth-account colorama
```
## Cara Menjalankan
1. Siapkan partial phrase di dalam file recover.py pada variabel partial_phrase
Gunakan tanda ? untuk kata yang hilang, contoh:
```
partial_phrase = "stage lift submit ? shoe manual strike ? play ? phrase ?"
```
2. Masukkan target address di variabel target_address (harus dalam lowercase):
```
target_address = "0x650b26e4e63abb7131424b46a68c6503c332bae1"
```
3. Jalankan skrip dengan perintah:
```
python recover.py
```
4. Skrip akan menampilkan progress interaktif di terminal dan secara otomatis menyimpan checkpoint ke file recover_checkpoint.json.
5. Jika seed phrase ditemukan, skrip akan menampilkan hasil lengkap dan menghapus checkpoint agar run berikutnya mulai dari awal.
