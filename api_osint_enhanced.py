"""
FastAPI Server untuk Phishing Detection dengan OSINT_ENHANCED Model
Menerima email dari N8N, extract features, run OSINT tools, predict
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import subprocess
import re
import json
import socket
import nmap
import shap
from typing import List, Optional
from datetime import datetime
from urllib.parse import urlparse
import whois as python_whois

app = FastAPI(title="Phishing Detection API - OSINT Enhanced")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# LOAD MODELS & VECTORIZER
# ========================================
text_model = None
osint_model = None
tfidf_vectorizer = None
shap_explainer = None

@app.on_event("startup")
def load_models():
    global text_model, osint_model, tfidf_vectorizer, shap_explainer
    
    print("🚀 Loading models...")
    
    try:
        # Load TEXT_ONLY model
        text_model = xgb.Booster()
        text_model.load_model('models/xgb_text_only.json')
        print("✅ TEXT_ONLY model loaded")
    except Exception as e:
        print(f"⚠️  TEXT_ONLY model not found: {e}")
    
    try:
        # Load OSINT_ENHANCED model
        osint_model = xgb.Booster()
        osint_model.load_model('models/xgb_osint_enhanced.json')
        print("✅ OSINT_ENHANCED model loaded")
    except Exception as e:
        print(f"⚠️  OSINT_ENHANCED model not found: {e}")
    
    try:
        # Load TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print("✅ TF-IDF vectorizer loaded")
    except Exception as e:
        print(f"⚠️  TF-IDF vectorizer not found: {e}")
    
    try:
        # Load SHAP explainer
        with open('models/shap_explainer.pkl', 'rb') as f:
            shap_explainer = pickle.load(f)
        print("✅ SHAP explainer loaded")
    except Exception as e:
        print(f"⚠️  SHAP explainer not found: {e}")
        print("   SHAP analysis will be skipped in predictions")
    
    print("🎉 API ready!\n")


# ========================================
# REQUEST MODELS
# ========================================
class SimpleRequest(BaseModel):
    email_text: str

class EmailRequest(BaseModel):
    email_text: str  # Only need raw email text!


# ========================================
# HELPER FUNCTIONS
# ========================================

def extract_urls_from_text(text: str):
    """Extract URLs from email text."""
    if not text:
        return []
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, str(text))

def extract_domains_from_urls(urls):
    """Extract domains from URLs."""
    domains = []
    for url in urls:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain:
                domains.append(domain)
        except:
            pass
    return list(set(domains))

def detect_interesting_url(text: str) -> int:
    """Detect suspicious URL patterns in email text"""
    urls = extract_urls_from_text(text)
    
    suspicious_patterns = [
        r'login', r'verify', r'update', r'secure', r'account',
        r'confirm', r'banking', r'paypal', r'signin', r'password'
    ]
    
    for url in urls:
        url_lower = url.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, url_lower):
                return 1
    
    return 0


# ========================================
# OSINT TOOLS FUNCTIONS
# ========================================

def extract_whois_features(domain: str) -> dict:
    """Extract WHOIS features: domain_age_days, has_registrar"""
    try:
        w = python_whois.whois(domain)
        
        # Domain age
        domain_age_days = 3650  # Default: 10 years (old domain)
        if w.creation_date:
            created = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
            if created:
                domain_age_days = (datetime.now() - created).days
        
        # Has registrar
        has_registrar = 1 if w.registrar and w.registrar != "Unknown" else 0
        
        return {
            'domain_age_days': domain_age_days,
            'has_registrar': has_registrar
        }
    except Exception as e:
        print(f"⚠️ WHOIS failed for {domain}: {e}")
        return {
            'domain_age_days': 3650,  # Assume old (safe)
            'has_registrar': 0
        }


def extract_nmap_features(domain: str, timeout: int = 10) -> dict:
    """
    Extract Nmap features: host_up, common_web_ports_open, open_ports_count, 
    filtered_ports_count, https_supported, latency, scan_duration
    """
    try:
        # Resolve domain to IP
        ip = socket.gethostbyname(domain)
        
        # Run Nmap scan (common web ports only for speed)
        cmd = f'nmap -Pn -p 80,443,8080,8443 --host-timeout {timeout}s {ip}'
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout+5
        )
        
        output = result.stdout
        
        # Parse results
        host_up = 1 if 'Host is up' in output else 0
        
        # Count open ports
        open_ports = re.findall(r'(\d+)/tcp\s+open', output)
        open_ports_count = len(open_ports)
        
        # Common web ports open (80, 443, 8080, 8443)
        common_web_ports = ['80', '443', '8080', '8443']
        common_web_ports_open = sum(1 for port in open_ports if port in common_web_ports)
        
        # Filtered ports
        filtered_ports = re.findall(r'(\d+)/tcp\s+filtered', output)
        filtered_ports_count = len(filtered_ports)
        
        # HTTPS supported (443 open)
        https_supported = 1 if '443' in open_ports else 0
        
        # Latency (extract from output)
        latency_match = re.search(r'latency\):\s+([\d.]+)ms', output)
        latency = float(latency_match.group(1)) if latency_match else 50.0
        
        # Scan duration (extract from Nmap output)
        duration_match = re.search(r'done.*in\s+([\d.]+)\s+seconds', output)
        scan_duration = float(duration_match.group(1)) if duration_match else 5.0
        
        return {
            'host_up': host_up,
            'common_web_ports_open': common_web_ports_open,
            'open_ports_count': open_ports_count,
            'filtered_ports_count': filtered_ports_count,
            'https_supported': https_supported,
            'latency': latency,
            'scan_duration': scan_duration
        }
        
    except subprocess.TimeoutExpired:
        print(f"⚠️ Nmap timeout for {domain}")
        return default_nmap_features()
    except Exception as e:
        print(f"⚠️ Nmap failed for {domain}: {e}")
        return default_nmap_features()


def default_nmap_features() -> dict:
    """Default Nmap features when scan fails"""
    return {
        'host_up': 0,
        'common_web_ports_open': 0,
        'open_ports_count': 0,
        'filtered_ports_count': 0,
        'https_supported': 0,
        'latency': 100.0,
        'scan_duration': 10.0
    }


def extract_theharvester_features(domain: str, timeout: int = 15) -> dict:
    """
    Extract theHarvester features: alternate_ip_count, asn_found, host_found, ip_found
    """
    try:
        # Run theHarvester (simple DNS/IP lookup only, no API keys)
        cmd = f'theHarvester -d {domain} -b dnsdumpster -l 10'
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout + result.stderr
        
        # Parse results
        # Count IPs found
        ips = re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', output)
        alternate_ip_count = len(set(ips)) if ips else 0
        
        # ASN found
        asn_found = 1 if re.search(r'AS\d+', output) else 0
        
        # Host found
        host_found = 1 if 'Hosts found' in output or len(ips) > 0 else 0
        
        # IP found
        ip_found = 1 if alternate_ip_count > 0 else 0
        
        return {
            'alternate_ip_count': alternate_ip_count,
            'asn_found': asn_found,
            'host_found': host_found,
            'ip_found': ip_found
        }
        
    except subprocess.TimeoutExpired:
        print(f"⚠️ theHarvester timeout for {domain}")
        return default_harvester_features()
    except Exception as e:
        print(f"⚠️ theHarvester failed for {domain}: {e}")
        return default_harvester_features()


def default_harvester_features() -> dict:
    """Default theHarvester features when scan fails"""
    return {
        'alternate_ip_count': 0,
        'asn_found': 0,
        'host_found': 0,
        'ip_found': 0
    }


def detect_interesting_url(domains: List[str]) -> int:
    """Detect if URL contains suspicious patterns"""
    suspicious_patterns = [
        'login', 'verify', 'update', 'secure', 'account', 
        'confirm', 'banking', 'paypal', 'signin', 'suspend'
    ]
    
    for domain in domains:
        domain_lower = domain.lower()
        if any(pattern in domain_lower for pattern in suspicious_patterns):
            return 1
    
    return 0


# ========================================
# PREDICTION ENDPOINT
# ========================================

@app.post("/predict_osint_enhanced")
async def predict_osint_enhanced(request: EmailRequest):
    """
    Endpoint untuk prediksi phishing dengan OSINT_ENHANCED model (173 features)
    
    Input: email_text (raw email content)
    API akan extract:
    - Manual text: 9 features
    - TF-IDF: 150 features
    - OSINT: 14 features (WHOIS, Nmap, theHarvester)
    
    Total: 173 features
    """
    
    if osint_model is None or tfidf_vectorizer is None:
        raise HTTPException(status_code=500, detail="OSINT model or TF-IDF vectorizer not loaded")
    
    try:
        print(f"\n{'='*70}")
        print(f"🔍 OSINT_ENHANCED PREDICTION")
        print(f"{'='*70}")
        
        # 1. Extract manual text features (9) - dari email_text
        print("📝 Extracting manual text features from email...")
        
        urls = extract_urls_from_text(request.email_text)
        domains = extract_domains_from_urls(urls)
        
        # Manual features
        n_urls = len(urls)
        n_domains = len(domains)
        
        # IP count
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        n_ips = len(re.findall(ip_pattern, request.email_text))
        
        # HTTPS/HTTP
        has_https = 1 if any('https://' in url for url in urls) else 0
        has_http = 1 if any('http://' in url and 'https://' not in url for url in urls) else 0
        
        # Suspicious keywords
        suspicious_keywords = [
            'urgent', 'verify', 'account', 'suspended', 'click', 'confirm',
            'password', 'update', 'secure', 'bank', 'login', 'credit',
            'ssn', 'social security', 'expire', 'limited time', 'act now',
            'congratulations', 'winner', 'prize', 'claim', 'refund'
        ]
        n_suspicious_keywords = sum(1 for kw in suspicious_keywords if kw in request.email_text.lower())
        
        # Email length
        email_length = len(request.email_text)
        
        # Attachment mentions
        attachment_keywords = ['attachment', 'attached', 'file', 'download', 'pdf', 'doc', 'zip']
        n_attachments = sum(1 for kw in attachment_keywords if kw in request.email_text.lower())
        
        # Suspicious TLD count
        suspicious_tlds = ['.xyz', '.top', '.club', '.work', '.download', '.loan', 
                           '.win', '.bid', '.click', '.stream', '.gq', '.cf', '.ml']
        count_susp_tld = 0
        for domain in domains:
            for tld in suspicious_tlds:
                if domain.endswith(tld):
                    count_susp_tld += 1
                    break
        
        manual_features = [
            n_urls,
            n_domains,
            n_ips,
            has_https,
            has_http,
            n_suspicious_keywords,
            email_length,
            n_attachments,
            count_susp_tld
        ]
        
        print(f"✅ Manual features: {manual_features}")
        
        # 2. TF-IDF features (150)
        print("📊 Extracting TF-IDF features...")
        tfidf_matrix = tfidf_vectorizer.transform([request.email_text])
        tfidf_features = tfidf_matrix.toarray()[0].tolist()
        print(f"✅ TF-IDF features extracted: {len(tfidf_features)} features")
        
        # 3. OSINT features (14) - run tools
        domain = domains[0] if domains else None
        
        if domain:
            print(f"🔍 Running OSINT tools for domain: {domain}")
            
            # WHOIS (2 features)
            whois_features = extract_whois_features(domain)
            print(f"   ✓ WHOIS: age={whois_features['domain_age_days']} days")
            
            # Nmap (7 features)
            nmap_features = extract_nmap_features(domain)
            print(f"   ✓ Nmap: host_up={nmap_features['host_up']}, ports={nmap_features['open_ports_count']}")
            
            # theHarvester (4 features)
            harvester_features = extract_theharvester_features(domain)
            print(f"   ✓ theHarvester: IPs={harvester_features['alternate_ip_count']}")
            
            # Interesting URL (1 feature)
            interesting_url = detect_interesting_url(request.email_text)
            
        else:
            # No domain - use defaults
            whois_features = {'domain_age_days': 3650, 'has_registrar': 0}
            nmap_features = default_nmap_features()
            harvester_features = default_harvester_features()
            interesting_url = 0
        
        # Combine OSINT features (URUTAN HARUS SAMA DENGAN TRAINING!)
        # Order: domain_age_days, host_up, common_web_ports_open, open_ports_count,
        #        filtered_ports_count, https_supported, latency, scan_duration,
        #        alternate_ip_count, asn_found, host_found, ip_found,
        #        interesting_url, has_registrar (14 features)
        osint_features = [
            whois_features['domain_age_days'],
            nmap_features['host_up'],
            nmap_features['common_web_ports_open'],
            nmap_features['open_ports_count'],
            nmap_features['filtered_ports_count'],
            nmap_features['https_supported'],
            nmap_features['latency'],
            nmap_features['scan_duration'],
            harvester_features['alternate_ip_count'],
            harvester_features['asn_found'],
            harvester_features['host_found'],
            harvester_features['ip_found'],
            interesting_url,
            whois_features['has_registrar']  # LAST!
        ]
        
        print(f"✅ OSINT features: {osint_features}")
        
        # 4. Combine ALL features: 9 + 150 + 14 = 173
        # Build feature names
        manual_feature_names = [
            'n_urls', 'n_domains', 'n_ips', 'has_https', 'has_http',
            'n_suspicious_keywords', 'email_length', 'n_attachments', 'count_suspicious_tld'
        ]
        
        # TF-IDF feature names (dari vectorizer)
        tfidf_feature_names = [f'tfidf_{word}' for word in tfidf_vectorizer.get_feature_names_out()]
        
        # OSINT feature names
        osint_feature_names = [
            'domain_age_days', 'host_up', 'common_web_ports_open', 'open_ports_count',
            'filtered_ports_count', 'https_supported', 'latency', 'scan_duration',
            'alternate_ip_count', 'asn_found', 'host_found', 'ip_found',
            'interesting_url', 'has_registrar'
        ]
        
        # Combine feature names
        all_feature_names = manual_feature_names + tfidf_feature_names + osint_feature_names
        
        # Combine feature values
        all_features = manual_features + tfidf_features + osint_features
        
        print(f"✅ Total features generated: {len(all_features)}")
        
        # 5. Get expected feature names from the model
        model_feature_names = osint_model.feature_names
        
        if model_feature_names is None:
            raise HTTPException(
                status_code=500,
                detail="Model does not have feature_names. Cannot validate features."
            )
        
        print(f"✅ Model expects {len(model_feature_names)} features")
        
        # 6. Validate that TF-IDF vectorizer matches the model
        # Check if feature names match (order doesn't matter yet)
        expected_features_set = set(model_feature_names)
        generated_features_set = set(all_feature_names)
        
        missing_features = expected_features_set - generated_features_set
        extra_features = generated_features_set - expected_features_set
        
        if missing_features or extra_features:
            error_msg = "❌ FEATURE MISMATCH ERROR\n\n"
            error_msg += "TF-IDF vectorizer yang di-load TIDAK SAMA dengan yang digunakan saat training model!\n\n"
            
            if missing_features:
                error_msg += f"Missing features (model expects but not generated): {len(missing_features)}\n"
                error_msg += f"Examples: {list(missing_features)[:10]}\n\n"
            
            if extra_features:
                error_msg += f"Extra features (generated but model doesn't expect): {len(extra_features)}\n"
                error_msg += f"Examples: {list(extra_features)[:10]}\n\n"
            
            error_msg += "SOLUSI:\n"
            error_msg += "1. Gunakan tfidf_vectorizer.pkl yang SAMA dengan yang digunakan saat training\n"
            error_msg += "2. Atau re-train model dengan vectorizer yang sekarang di-load\n"
            error_msg += "3. Pastikan file models/tfidf_vectorizer.pkl adalah file yang benar dari training"
            
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        print("✅ Feature names match! Reordering to match model...")
        
        # 7. Create DataFrame with features in the correct order
        # Reorder to match model's expected order
        X_df = pd.DataFrame([all_features], columns=all_feature_names)
        X_df = X_df[model_feature_names]  # Reorder columns to match model
        
        print(f"✅ DataFrame shape: {X_df.shape}")
        print(f"✅ First 5 columns: {list(X_df.columns[:5])}")
        print(f"✅ Last 5 columns: {list(X_df.columns[-5:])}")
        
        # Verify feature count
        if X_df.shape[1] != 173:
            raise HTTPException(
                status_code=500, 
                detail=f"Feature count mismatch: got {X_df.shape[1]}, expected 173"
            )
        
        # Create DMatrix from DataFrame (will preserve column names)
        dmatrix = xgb.DMatrix(X_df)
        
        # 7. Predict
        prediction = osint_model.predict(dmatrix)[0]
        
        # 8. Compute SHAP values (SKIP for now - too slow in production)
        # SHAP computation dapat memakan waktu 5-30 detik per prediction
        # Untuk production, lebih baik di-disable atau di-cache
        analisis_shap = {
            "info": "SHAP analysis disabled for faster response",
            "top_features": []
        }
        
        # Uncomment below to enable SHAP (WARNING: SLOW!)
        # if shap_explainer is not None:
        #     try:
        #         print("📊 Computing SHAP values...")
        #         shap_values = shap_explainer.shap_values(X_df)
        #         
        #         if isinstance(shap_values, list):
        #             shap_values = shap_values[1]
        #         
        #         feature_importance = pd.DataFrame({
        #             'feature': X_df.columns,
        #             'shap_value': shap_values[0]
        #         }).sort_values('shap_value', key=abs, ascending=False)
        #         
        #         top_features = feature_importance.head(5)
        #         
        #         analisis_shap = {
        #             "top_features": [
        #                 {
        #                     "feature": row['feature'],
        #                     "impact": float(row['shap_value']),
        #                     "direction": "phishing" if row['shap_value'] > 0 else "legitimate"
        #                 }
        #                 for _, row in top_features.iterrows()
        #             ]
        #         }
        #         print(f"   ✓ SHAP complete")
        #         
        #     except Exception as e:
        #         print(f"⚠️ SHAP computation failed: {e}")
        #         analisis_shap = {"error": str(e)}
        
        # 9. Result
        label = "PHISHING" if prediction > 0.5 else "LEGITIMATE"
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        
        return {
            "label": label,
            "score_bahaya": float(prediction),
            "confidence": confidence,
            "features_used": 173,
            "analisis_shap": analisis_shap,
            "osint_data": {
                "domain": domain,
                "domain_age_days": whois_features['domain_age_days'],
                "host_up": nmap_features['host_up'],
                "https_supported": nmap_features['https_supported'],
                "alternate_ips": harvester_features['alternate_ip_count']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
async def root():
    return {
        "service": "Phishing Detection API - OSINT Enhanced",
        "model": "XGBoost + OSINT (173 features)",
        "version": "1.0",
        "endpoints": [
            "/predict_osint_enhanced"
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tfidf_loaded": tfidf_vectorizer is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
