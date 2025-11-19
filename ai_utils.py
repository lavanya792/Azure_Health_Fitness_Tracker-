# ai_utils.py
import os
import json
import requests
from typing import List, Dict
from io import BytesIO
from PIL import Image
import pandas as pd
from fuzzywuzzy import process

def safe_load_json(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)
        return {}
    with open(path, "r") as f:
        txt = f.read().strip()
        if not txt:
            return {}
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return {}

def safe_save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# Azure Language (Key Phrases)
def azure_keyphrases(text: str, endpoint: str, key: str) -> List[str]:
    if not text:
        return []
    url = endpoint.rstrip("/") + "/text/analytics/v3.0/keyPhrases"
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}
    body = {"documents": [{"id": "1", "language": "en", "text": text}]}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("documents", [{}])[0].get("keyPhrases", [])
    except Exception:
        return []

# Azure Computer Vision analyze
def azure_vision_analyze_image(image_bytes: bytes, endpoint: str, key: str) -> Dict:
    url = endpoint.rstrip("/") + "/vision/v3.2/analyze"
    params = {"visualFeatures": "Tags,Description,Objects", "language": "en"}
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"}
    try:
        r = requests.post(url, headers=headers, params=params, data=image_bytes, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Map vision tags/descriptions to dataset dish names using fuzzy matching
def find_closest_foods_from_tags(tags: List[str], df_food: pd.DataFrame, top_n=5, score_cutoff=60):
    if df_food is None or df_food.empty:
        return []
    dish_list = df_food['Dish Name'].astype(str).tolist()
    candidates = {}
    for t in tags:
        match = process.extractOne(t, dish_list, score_cutoff=score_cutoff)
        if match:
            name, score = match[0], match[1]
            if name in candidates:
                candidates[name] = max(candidates[name], score)
            else:
                candidates[name] = score
    sorted_matches = sorted(candidates.items(), key=lambda x: -x[1])[:top_n]
    return sorted_matches

# Nutrition calculation (dataset expected to have per-100g values)
def calculate_nutrition_for_servings(df_food: pd.DataFrame, dish_name: str, servings: float):
    dish_row = df_food[df_food['Dish Name'].str.lower() == dish_name.lower()]
    if dish_row.empty:
        # try fuzzy match
        match = process.extractOne(dish_name, df_food['Dish Name'].astype(str).tolist(), score_cutoff=50)
        if not match:
            return None
        dish_name = match[0]
        dish_row = df_food[df_food['Dish Name'] == dish_name]
    row = dish_row.iloc[0]
    # assume columns exist as 'Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fats (g)'
    calories = float(row.get('Calories (kcal)', 0)) * servings
    protein = float(row.get('Protein (g)', 0)) * servings
    carbs = float(row.get('Carbohydrates (g)', 0)) * servings
    fats = float(row.get('Fats (g)', 0)) * servings
    return {"dish": dish_name, "servings": servings, "calories": round(calories,2),
            "protein_g": round(protein,2), "carbs_g": round(carbs,2), "fat_g": round(fats,2)}