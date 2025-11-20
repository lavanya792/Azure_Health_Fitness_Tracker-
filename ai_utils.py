# ai_utils.py
import os, json, requests
from typing import List, Tuple, Dict, Optional
import pandas as pd
from fuzzywuzzy import process

# -------------------------
# JSON helpers
# -------------------------
def safe_load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            txt = f.read().strip()
            if not txt:
                return {}
            return json.loads(txt)
    except Exception:
        return {}

def safe_save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# -------------------------
# Azure Language: KeyPhrases & Sentiment
# -------------------------
def azure_keyphrases(text: str, endpoint: str, key: str) -> List[str]:
    if not text or not endpoint or not key:
        return []
    url = endpoint.rstrip("/") + "/text/analytics/v3.0/keyPhrases"
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}
    body = {"documents": [{"id":"1", "language":"en", "text": text}]}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=10)
        r.raise_for_status()
        j = r.json()
        return j.get("documents", [{}])[0].get("keyPhrases", [])
    except Exception:
        return []

def azure_sentiment_analysis(text: str, endpoint: str, key: str) -> Dict:
    """
    Returns dict: {'sentiment':'positive'/'neutral'/'negative', 'confidenceScores': {...}}
    """
    if not text or not endpoint or not key:
        return {}
    url = endpoint.rstrip("/") + "/text/analytics/v3.0/sentiment"
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}
    body = {"documents": [{"id":"1","language":"en","text": text}]}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=10)
        r.raise_for_status()
        j = r.json()
        doc = j.get("documents", [{}])[0]
        return {"sentiment": doc.get("sentiment"), "confidenceScores": doc.get("confidenceScores", {})}
    except Exception:
        return {}

# -------------------------
# Azure Vision analyze (tags & description)
# -------------------------
def azure_vision_analyze_image(image_bytes: bytes, endpoint: str, key: str) -> Dict:
    if not image_bytes or not endpoint or not key:
        return {"error": "Missing image or keys."}
    url = endpoint.rstrip("/") + "/vision/v3.2/analyze"
    params = {"visualFeatures":"Tags,Description,Objects","language":"en"}
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type":"application/octet-stream"}
    try:
        r = requests.post(url, headers=headers, params=params, data=image_bytes, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Fuzzy matching helpers
# -------------------------
def find_closest_foods_from_tags(tags: List[str], df_food: pd.DataFrame, top_n: int = 6, score_cutoff: int = 55) -> List[Tuple[str,int]]:
    if df_food is None or df_food.empty or not tags:
        return []
    dish_list = df_food['Dish Name'].astype(str).tolist()
    candidates = {}
    for t in tags:
        if not t:
            continue
        match = process.extractOne(t, dish_list, score_cutoff=score_cutoff)
        if match:
            name, score = match[0], match[1]
            candidates[name] = max(candidates.get(name, 0), score)
    sorted_matches = sorted(candidates.items(), key=lambda x: -x[1])[:top_n]
    return sorted_matches

# -------------------------
# Nutrition calculation
# -------------------------
def calculate_nutrition_for_servings(df_food: pd.DataFrame, dish_name: str, servings: float) -> Optional[dict]:
    if df_food is None or df_food.empty:
        return None
    dish_name = str(dish_name).strip()
    if not dish_name:
        return None
    # exact case-insensitive match
    matches = df_food[df_food['Dish Name'].str.lower() == dish_name.lower()]
    if matches.empty:
        # fuzzy fallback
        match = process.extractOne(dish_name, df_food['Dish Name'].astype(str).tolist(), score_cutoff=50)
        if not match:
            return None
        dish_name = match[0]
        matches = df_food[df_food['Dish Name'] == dish_name]
        if matches.empty:
            return None
    row = matches.iloc[0]
    def _val(col):
        try:
            return float(row.get(col, 0) or 0)
        except Exception:
            return 0.0
    # assume dataset nutrition values are per serving (100g) — user uses servings accordingly
    calories = _val('Calories (kcal)') * servings
    protein = _val('Protein (g)') * servings
    carbs = _val('Carbohydrates (g)') * servings
    fats = _val('Fats (g)') * servings
    return {
        "dish": dish_name,
        "servings": servings,
        "calories": round(calories, 2),
        "protein_g": round(protein, 2),
        "carbs_g": round(carbs, 2),
        "fat_g": round(fats, 2)
    }
