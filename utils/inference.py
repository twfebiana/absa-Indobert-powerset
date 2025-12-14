# utils/inference.py
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocessing import preprocess_for_inference


MODEL_DIR = "model/best_model_final"
TOKENIZER_NAME = MODEL_DIR 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("utils/label_maps.json", "r", encoding="utf-8") as f:
    maps = json.load(f)
label2id = maps.get("label2id", {})
id2label = {int(k): v for k,v in maps.get("id2label", {}).items()}
binary_map = maps.get("binary_map", {}) 

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

def decode_binary_combo(class_name):
    return binary_map.get(class_name, None)

def map_binary_to_aspects(binary_string):
    aspects = {}
    if binary_string is None:
        return aspects
    keys = ["kpms_pos","kpms_neg","fi_pos","fi_neg","wt_pos","wt_neg","bl_pos","bl_neg"]
    for k, bit in zip(keys, list(binary_string)):
        aspects[k] = int(bit)
    aspect_summary = {}
    for a in ["kpms","fi","wt","bl"]:
        pos = aspects.get(f"{a}_pos", 0)
        neg = aspects.get(f"{a}_neg", 0)
        if pos==1 and neg==0:
            aspect_summary[a] = 1
        elif neg==1 and pos==0:
            aspect_summary[a] = -1
        elif pos==1 and neg==1:
            aspect_summary[a] = 0 
        else:
            aspect_summary[a] = 0
    return aspect_summary

def predict_single(text):
    clean = preprocess_for_inference(text)
    enc = tokenizer(clean, return_tensors="pt", truncation=True, padding=True, max_length=128)
    enc = {k:v.to(DEVICE) for k,v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        pred_id = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
    class_name = id2label.get(pred_id, str(pred_id))
    binary = decode_binary_combo(class_name)
    aspects = map_binary_to_aspects(binary)
    # compute softmax probs
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
    return {
        "clean_text": clean,
        "class": class_name,
        "binary": binary,
        "aspects": aspects,
        "probs": probs,
        "pred_id": pred_id
    }

def predict_batch_dataframe(df, text_col="ulasan", progress=False):
    results = []
    for i, t in enumerate(df[text_col].astype(str).tolist()):
        res = predict_single(t)
        flat = {
            "ulasan": t,
            "clean_text": res["clean_text"],
            "pred_class": res["class"],
            "pred_binary": res["binary"],
            "pred_id": res["pred_id"],
            "probs": res["probs"]
        }
        for k,v in res["aspects"].items():
            flat[f"aspect_{k}"] = v
        results.append(flat)
    return pd.DataFrame(results)
