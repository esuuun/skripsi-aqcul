import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === KONFIGURASI ===
DATASET_FILE = os.path.join("dataset", "phishing_email.csv") 
MODEL_DIR = "models"

# Buat folder models kalau belum ada
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Fitur yang akan dipakai (Wajib konsisten sama main.py)
FEATURE_NAMES = [
    "n_urls", "n_domains", "n_attachments", "has_password_word", 
    "min_domain_age_days", "count_suspicious_tld", "is_spf_fail"
]

def extract_features_from_text(df):
    """
    Mengubah teks email mentah dari CSV menjadi fitur angka.
    """
    print("⚙️  Sedang mengekstrak fitur dari kolom 'text_combined'...")
    
    # --- 1. FITUR DARI TEKS ASLI ---
    
    # Pastikan kolom text dibaca sebagai string
    df['text_combined'] = df['text_combined'].astype(str)
    
    # Hitung jumlah URL (http/https)
    df['n_urls'] = df['text_combined'].apply(lambda x: len(re.findall(r'https?://', x)))
    
    # Asumsi sementara: jumlah domain = jumlah URL
    df['n_domains'] = df['n_urls']
    
    # Cek keyword sensitif (password, login, bank, dll)
    keywords = ['password', 'verify', 'account', 'bank', 'urgent', 'login', 'security', 'alert']
    pattern = '|'.join(keywords)
    df['has_password_word'] = df['text_combined'].apply(
        lambda x: 1 if re.search(pattern, x, re.IGNORECASE) else 0
    )
    
    # --- 2. TARGET / LABEL ---
    # Di dataset kamu labelnya sudah angka (0/1), jadi langsung pakai.
    # Asumsi: 1 = Phishing, 0 = Aman.
    y = df['label'].values
    n_samples = len(df)
    
    # --- 3. SIMULASI FITUR OSINT (PENTING!) ---
    # Karena CSV cuma berisi teks (gak ada data WHOIS domain age),
    # kita harus "Pura-pura" generate data OSINT berdasarkan labelnya.
    # Tujuannya: Supaya model BELAJAR bahwa "Domain Muda = Bahaya".
    
    print("⚠️  Mensimulasikan data WHOIS & SPF untuk keperluan training...")
    
    # Logika: Kalau Phishing (1), umur domain dibuat muda (0-60 hari)
    # Kalau Aman (0), umur domain dibuat tua (> 100 hari)
    df['min_domain_age_days'] = np.where(
        y == 1,
        np.random.randint(0, 60, size=n_samples),     # Phishing
        np.random.randint(100, 3650, size=n_samples) # Aman
    )
    
    # Logika: Phishing sering pakai TLD aneh (.xyz, .top)
    df['count_suspicious_tld'] = np.where(
        y == 1, 
        np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]), 
        0
    )
    
    # Logika: Phishing sering gagal SPF
    df['is_spf_fail'] = np.where(
        y == 1, 
        np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]), 
        0
    )
    
    # Set attachment 0 karena tidak ada info di CSV
    df['n_attachments'] = 0 
    
    return df[FEATURE_NAMES], df['label']

def train():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ ERROR: File {DATASET_FILE} gak ketemu! Pastikan ada di folder yang sama.")
        return

    print(f"📂 Loading dataset: {DATASET_FILE}")
    try:
        # Load CSV
        df = pd.read_csv(DATASET_FILE)
    except Exception as e:
        print(f"❌ Gagal baca CSV: {e}")
        return

    # Buang baris yang text-nya kosong/NaN
    df = df.dropna(subset=['text_combined', 'label'])
    
    print(f"📊 Total Data: {len(df)} baris")
    
    # Ekstrak Fitur
    X, y = extract_features_from_text(df)
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n🚀 Mulai Training XGBoost...")
    # Setup Model
    model = xgb.XGBClassifier(
        n_estimators=200,    # Jumlah 'pohon' keputusan
        max_depth=6,         # Kedalaman pohon
        learning_rate=0.1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Cek Hasil
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n✅ Training Selesai! Akurasi: {acc*100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, preds))
    
    # Simpan Model
    model.save_model(os.path.join(MODEL_DIR, "xgb_phishing.json"))
    
    # Simpan Explainer (SHAP)
    explainer = shap.TreeExplainer(model)
    with open(os.path.join(MODEL_DIR, "shap_explainer.pkl"), 'wb') as f:
        pickle.dump(explainer, f)
        
    print("🎉 File model & explainer berhasil disimpan! Siap dijalankan di main.py.")

if __name__ == "__main__":
    train()