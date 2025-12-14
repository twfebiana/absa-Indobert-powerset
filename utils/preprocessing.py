import re
import pandas as pd

def load_slang_dict(path="utils/colloquial-indonesian-lexicon.csv"):
    try:
        df = pd.read_csv(path)
        return dict(zip(df["slang"].astype(str), df["formal"].astype(str)))
    except Exception:
        return {}

SLANG_DICT = load_slang_dict()

def clean_text(text: str) -> str:
    text = str(text)
    # 1. Hilangkan spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    # 2. Hilangkan URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # 3. Hilangkan mention dan hashtag
    text = re.sub(r'@\w+|#\w+', '', text)
    # 4. Normalisasi x2 -> x-x 
    text = re.sub(r'\b([a-zA-Z]+)(?:2|²|\^2)(?=\b)', r'\1-\1', text)
    # 5. Hanya simpan karakter umum
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    # 6. Kurangi huruf berulang (aaa -> aa)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text

def normalize_slang(text: str) -> str:
    words = text.lower().split()
    new_words = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(new_words)

def preprocess_for_inference(text: str) -> str:
    t = clean_text(text)
    t = normalize_slang(t)
    return t
