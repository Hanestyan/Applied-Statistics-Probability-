from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# A. PERSIAPAN DATA (Histori Email)
email_histori = [
    "promo diskon free uang",        # Spam
    "rapat tim besok pagi",          # Aman
    "klaim hadiah uang tunai free",  # Spam
    "revisi laporan bulanan"         # Aman
]
label_histori = ["Spam", "Aman", "Spam", "Aman"]

# Komputer tidak mengerti teks, jadi kita ubah kata-kata menjadi matriks angka (frekuensi kemunculan)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(email_histori)

# B. MELATIH MODEL (Menghitung Prior & Likelihood)
model_bayes = MultinomialNB()
# Saat .fit() dipanggil, mesin secara otomatis menghitung P(Spam), P("free"|Spam), P("uang"|Spam), dst.
model_bayes.fit(X_train, label_histori)

# C. PENGUJIAN (Inference)
email_baru = ["klaim free diskon untuk rapat besok"]

# Ubah email baru ke format angka yang sama
X_test = vectorizer.transform(email_baru)

# Model menebak berdasarkan probabilitas tertinggi
tebakan = model_bayes.predict(X_test)
probabilitas = model_bayes.predict_proba(X_test)

print("Email Baru:", email_baru[0])
print("Tebakan AI:", tebakan[0])
print(f"Keyakinan Model (Probabilitas): Aman = {probabilitas[0][0]*100:.1f}%, Spam = {probabilitas[0][1]*100:.1f}%")