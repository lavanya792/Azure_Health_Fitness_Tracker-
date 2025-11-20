# app.py
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import os
import matplotlib.pyplot as plt
import requests  # used for translator fallback
from typing import Tuple, Dict

from ai_utils import (
    safe_load_json,
    safe_save_json,
    azure_keyphrases,
    azure_sentiment_analysis,
    azure_vision_analyze_image,
    find_closest_foods_from_tags,
    calculate_nutrition_for_servings,
)

st.set_page_config(page_title="AI-Enabled Health Tracker", layout="wide", initial_sidebar_state="expanded")

# ---- paths ----
DATASET_PATH = "Indian_Food_Nutrition_Processed.csv"
USERS_PATH = "users.json"
PROFILES_PATH = "profiles.json"
LOG_CSV = "all_user_activity_log.csv"

# -------- helpers (daily & weekly stats) ----------
def generate_daily_stats(username: str, log_csv_path: str) -> Tuple[Dict, str]:
    if not os.path.exists(log_csv_path):
        return None, "No logs found."
    df = pd.read_csv(log_csv_path)
    if 'username' not in df.columns or 'date' not in df.columns:
        return None, "Log file format incorrect."
    today = datetime.now().strftime("%Y-%m-%d")
    df_user_today = df[(df['username'] == username) & (df['date'] == today)]
    if df_user_today.empty:
        return None, "No entries logged today."
    total_steps = int(df_user_today['steps'].sum())
    total_calories = float(df_user_today['calories'].sum())
    protein = float(df_user_today['protein_g'].sum()) if 'protein_g' in df_user_today.columns else 0.0
    fats = float(df_user_today['fat_g'].sum()) if 'fat_g' in df_user_today.columns else 0.0
    carbs = float(df_user_today['carbs_g'].sum()) if 'carbs_g' in df_user_today.columns else 0.0
    stats = {
        "date": today,
        "steps": total_steps,
        "calories": round(total_calories,2),
        "protein": round(protein,2),
        "fats": round(fats,2),
        "carbs": round(carbs,2)
    }
    return stats, "Daily stats generated"

def generate_weekly_summary(username: str, log_csv_path: str) -> Tuple[Dict, pd.DataFrame]:
    if not os.path.exists(log_csv_path):
        return None, "No logs found."
    df = pd.read_csv(log_csv_path, parse_dates=['date'])
    if 'username' not in df.columns:
        return None, "Log file format incorrect."
    today = pd.to_datetime(datetime.now().date())
    week_start = today - pd.Timedelta(days=6)
    df_user = df[df['username'] == username].copy()
    if df_user.empty:
        return None, "No logs for user."
    df_user['date'] = pd.to_datetime(df_user['date'])
    df_week = df_user[(df_user['date'] >= week_start) & (df_user['date'] <= today)]
    if df_week.empty:
        return None, "No entries in the last 7 days."
    totals = {
        "total_steps": int(df_week['steps'].sum()),
        "avg_steps": int(df_week['steps'].mean()),
        "total_cal": float(df_week['calories'].sum()),
        "avg_cal": float(df_week['calories'].mean()),
        "protein_g": float(df_week['protein_g'].sum()) if 'protein_g' in df_week.columns else 0.0,
        "fat_g": float(df_week['fat_g'].sum()) if 'fat_g' in df_week.columns else 0.0,
        "carbs_g": float(df_week['carbs_g'].sum()) if 'carbs_g' in df_week.columns else 0.0,
        "days_logged": int(df_week.shape[0])
    }
    return totals, df_week

def daily_suggestions(profile: dict, stats: dict):
    suggestions = []
    if not stats:
        return ["No data logged today. Add your meals and steps first."]
    goal_steps = profile.get("steps_goal", 8000)
    if stats['steps'] < goal_steps:
        suggestions.append(f"Steps low ({stats['steps']} < {goal_steps}). Try a 15-min brisk walk.")
    else:
        suggestions.append("Step goal reached — great job!")
    goal_cal = profile.get("calorie_goal", 2000)
    if stats['calories'] > goal_cal + 200:
        suggestions.append("Calories are higher than your goal — prefer a lighter dinner.")
    elif stats['calories'] < goal_cal - 200:
        suggestions.append("Calories are lower than your goal — add a nutritious snack.")
    else:
        suggestions.append("Calories are within the target range.")
    if stats['protein'] < 40:
        suggestions.append("Protein low — include paneer, dal, yogurt or nuts.")
    if stats['fats'] > 80:
        suggestions.append("High fat intake — avoid fried foods next meal.")
    return suggestions

# ---- load dataset ----
@st.cache_data
def load_food_dataset(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

df_food = load_food_dataset(DATASET_PATH)
if df_food.empty:
    st.warning(f"Dataset not found at '{DATASET_PATH}'. Place the CSV in this path or update DATASET_PATH.")

# ---- load JSONs ----
users = safe_load_json(USERS_PATH)
profiles = safe_load_json(PROFILES_PATH)

def hash_password(pw: str):
    import hashlib
    return hashlib.sha256(pw.encode()).hexdigest()

def save_users():
    safe_save_json(USERS_PATH, users)

def save_profiles():
    safe_save_json(PROFILES_PATH, profiles)

def append_log(entry: dict):
    cols = ['username','date','steps','calories','protein_g','fat_g','carbs_g','notes']
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=cols).to_csv(LOG_CSV, index=False)
    df = pd.read_csv(LOG_CSV)
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(LOG_CSV, index=False)

# ---- session init ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

# ---- Apple-like CSS (clean, airy) ----
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "SF Pro Text", Helvetica, Arial, sans-serif;
}

/* Main title */
.main-title {
    color: #1C1C1E; 
    font-size: 28px;
    font-weight: 600;
    padding-bottom: 6px;
}

/* Card containers (macOS panel style) */
.card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 18px 20px;
    border-radius: 16px;
    border: 1px solid rgba(230,230,230,0.5);
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

/* Input elements */
input, select, textarea {
    border-radius: 10px !important;
    border: 1px solid #D1D1D6 !important;
    background: #F2F2F7 !important;
    padding: 8px 10px !important;
    font-size: 15px !important;
    color: #1C1C1E !important;
}

/* Buttons — macOS style */
.stButton > button {
    background-color: #007AFF !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    border: none !important;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    background-color: #005BBB !important;
}

/* Sidebar Apple look */
section[data-testid="stSidebar"] {
    background-color: #F2F2F7;
    border-right: 1px solid #E5E5EA;
}

.stSidebar input, .stSidebar select {
    background: white !important;
}

/* Metrics card (Steps/Calories) */
[data-testid="stMetricValue"] {
    color: #111;
    font-weight: 600;
}

[data-testid="stMetricLabel"] {
    color: #8E8E93;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

</style>
""", unsafe_allow_html=True)
 

# ---- sidebar ----
st.sidebar.markdown("<h3 style='color:var(--accent)'>Account</h3>", unsafe_allow_html=True)
menu = st.sidebar.selectbox("", ["Login","Signup","Dashboard","Weekly Summary","AI Insights","Profile"])

# small helper: metrics card (styled)
def metrics_card(k, v, delta=None):
    st.metric(label=k, value=v, delta=delta)

# -------- SIGNUP --------
if menu == "Signup":
    st.markdown("<div class='main-title'>Create account</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Secure and personal — your profile stays private.</div>", unsafe_allow_html=True)
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Create account"):
        if not new_user or not new_pass:
            st.warning("Enter username and password")
        elif new_user in users:
            st.error("User exists")
        else:
            users[new_user] = hash_password(new_pass)
            profiles[new_user] = {"age":None,"weight":None,"height":None,"activity":"Medium",
                                  "steps_goal":8000,"calorie_goal":2000,"diet":"vegetarian"}
            save_users(); save_profiles()
            st.success("Account created. Please login.")

# -------- LOGIN --------
elif menu == "Login":
    st.markdown("<div class='main-title'>Welcome back</div>", unsafe_allow_html=True)
    u = st.text_input("Username", key="login_user")
    p = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        if u in users and users[u] == hash_password(p):
            st.session_state.logged_in = True
            st.session_state.user = u
            st.success(f"Welcome back, {u}!")
        else:
            st.error("Incorrect credentials")

# -------- PROFILE --------
elif menu == "Profile":
    if not st.session_state.logged_in:
        st.info("Please login first.")
    else:
        u = st.session_state.user
        st.markdown("<div class='main-title'>Profile & Goals</div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>Update your personal info and daily goals.</div>", unsafe_allow_html=True)
        prof = profiles.get(u, {})
        prof['age'] = st.number_input("Age", min_value=1, max_value=120, value=prof.get('age') or 20)
        prof['weight'] = st.number_input("Weight (kg)", min_value=20, max_value=300, value=prof.get('weight') or 60)
        prof['height'] = st.number_input("Height (cm)", min_value=100, max_value=230, value=prof.get('height') or 160)
        prof['activity'] = st.selectbox("Activity level", ["Low","Medium","High"], index=["Low","Medium","High"].index(prof.get('activity',"Medium")))
        prof['steps_goal'] = st.number_input("Daily steps goal", 1000, 50000, value=prof.get('steps_goal',8000), step=500)
        prof['calorie_goal'] = st.number_input("Daily calorie goal", 800, 5000, value=prof.get('calorie_goal',2000), step=50)
        prof['diet'] = st.selectbox("Diet preference", ["vegetarian","non-vegetarian","vegan"], index=["vegetarian","non-vegetarian","vegan"].index(prof.get('diet',"vegetarian")))
        if st.button("Save Profile"):
            profiles[u] = prof
            save_profiles()
            st.success("Profile saved.")

# -------- DASHBOARD --------
elif menu == "Dashboard":
    if not st.session_state.logged_in:
        st.info("Please login first.")
    else:
        u = st.session_state.user
        st.markdown("<div class='main-title'>Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Track meals, steps and get daily suggestions</div>", unsafe_allow_html=True)

        # compute quick stats & show daily suggestions automatically (if present)
        stats, _ = generate_daily_stats(u, LOG_CSV)
        prof = profiles.get(u, {})
        if stats:
            suggs = daily_suggestions(prof, stats)
            with st.container():
                st.markdown("<div class='card'><b>Today's suggestions</b></div>", unsafe_allow_html=True)
                for s in suggs:
                    st.write("• " + s)
        else:
            st.info("No entries for today yet — log a meal below.")

        # top metrics
        col1, col2, col3 = st.columns(3)
        if os.path.exists(LOG_CSV):
            df_all = pd.read_csv(LOG_CSV)
            df_user = df_all[df_all['username']==u]
            total_cal = round(df_user['calories'].sum(),1) if not df_user.empty else 0
            total_steps = int(df_user['steps'].sum()) if not df_user.empty else 0
            days_logged = int(df_user.shape[0]) if not df_user.empty else 0
        else:
            total_cal = 0; total_steps = 0; days_logged = 0
        with col1:
            metrics_card("Total Calories", f"{total_cal} kcal")
        with col2:
            metrics_card("Total Steps", f"{total_steps}")
        with col3:
            metrics_card("Days Logged", f"{days_logged}")

        st.markdown("### Log today's activity")
        left, right = st.columns([2,1])
        with left:
            st.markdown("#### Meals")
            b1 = st.text_input("Breakfast (name):", key="b1")
            s1 = st.number_input("Servings (1 serve = 100g)", min_value=0.25, max_value=10.0, value=1.0, step=0.25, key="s1")
            b2 = st.text_input("Lunch (name):", key="b2")
            s2 = st.number_input("Servings", min_value=0.25, max_value=10.0, value=1.0, step=0.25, key="s2")
            b3 = st.text_input("Dinner (name):", key="b3")
            s3 = st.number_input("Servings", min_value=0.25, max_value=10.0, value=1.0, step=0.25, key="s3")
            notes = st.text_area("Notes (optional)")
        with right:
            st.markdown("#### Activity")
            steps = st.number_input("Steps today", min_value=0, max_value=200000, value=0, step=100)
            workout = st.selectbox("Workout type", ["None","Yoga","Walk","Run","Gym"])
            intensity = st.selectbox("Intensity", ["Low","Medium","High"])
            if st.button("Compute & Save"):
                total_cal = total_prot = total_fat = total_carbs = 0.0
                entries = [(b1,s1),(b2,s2),(b3,s3)]
                for dish, sv in entries:
                    if dish and str(dish).strip().lower()!="none":
                        nut = calculate_nutrition_for_servings(df_food, dish, float(sv))
                        if nut:
                            total_cal += nut['calories']
                            total_prot += nut['protein_g']
                            total_fat += nut['fat_g']
                            total_carbs += nut['carbs_g']
                entry = {
                    "username": u,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "steps": int(steps),
                    "calories": round(total_cal,2),
                    "protein_g": round(total_prot,2),
                    "fat_g": round(total_fat,2),
                    "carbs_g": round(total_carbs,2),
                    "notes": notes
                }
                append_log(entry)
                st.success("Saved today's entry.")

        # recent table & charts
        if os.path.exists(LOG_CSV):
            df_all = pd.read_csv(LOG_CSV)
            df_user = df_all[df_all['username']==u]
            if not df_user.empty:
                last = df_user.tail(7)
                st.markdown("#### Recent entries")
                st.dataframe(last[['date','steps','calories','protein_g','fat_g','carbs_g']].reset_index(drop=True), height=220)
                # charts
                fig, ax = plt.subplots(1,2, figsize=(10,3), constrained_layout=True)
                last.plot(x='date', y='steps', ax=ax[0], marker='o', title='Steps (last entries)')
                last.plot(x='date', y='calories', ax=ax[1], marker='o', title='Calories (last entries)')
                ax[0].tick_params(axis='x', rotation=30)
                ax[1].tick_params(axis='x', rotation=30)
                st.pyplot(fig)
            else:
                st.info("No logs yet. Add today's entry!")

# -------- WEEKLY SUMMARY PAGE --------
elif menu == "Weekly Summary":
    if not st.session_state.logged_in:
        st.info("Please login first.")
    else:
        u = st.session_state.user
        st.markdown("<div class='main-title'>Weekly Summary</div>", unsafe_allow_html=True)
        totals, df_week = generate_weekly_summary(u, LOG_CSV)
        if not totals:
            st.info("No weekly data available yet.")
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write(f"Logged days: **{totals['days_logged']}/7**")
            st.write(f"Total steps: **{totals['total_steps']}** (avg **{totals['avg_steps']}** /day)")
            st.write(f"Total calories: **{round(totals['total_cal'],1)} kcal** (avg **{round(totals['avg_cal'],1)} kcal/day)")
            st.write(f"Week macronutrients — Protein: **{round(totals['protein_g'],1)} g**, Fat: **{round(totals['fat_g'],1)} g**, Carbs: **{round(totals['carbs_g'],1)} g**")
            st.markdown("</div>", unsafe_allow_html=True)

            # charts: steps and calories daywise
            df_plot = df_week.groupby(df_week['date'].dt.date).agg({'steps':'sum','calories':'sum'}).reset_index()
            fig, ax = plt.subplots(1,2, figsize=(11,3), constrained_layout=True)
            ax[0].bar(df_plot['date'].astype(str), df_plot['steps'])
            ax[0].set_title("Daily Steps (week)")
            ax[0].tick_params(axis='x', rotation=30)
            ax[1].bar(df_plot['date'].astype(str), df_plot['calories'])
            ax[1].set_title("Daily Calories (week)")
            ax[1].tick_params(axis='x', rotation=30)
            st.pyplot(fig)

# -------- AI INSIGHTS (translate, sentiment, photo analyze) --------
elif menu == "AI Insights":
    if not st.session_state.logged_in:
        st.info("Please login first.")
    else:
        u = st.session_state.user
        st.markdown("<div class='main-title'>AI Insights</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Translate text, analyze sentiment, and use the photo analyzer.</div>", unsafe_allow_html=True)

        # Text translate + sentiment
        st.subheader("Translate & Sentiment")
        text = st.text_area("Enter text to translate / analyze (e.g., notes about meals):", height=120)
        target_lang = st.selectbox("Translate to:", ["hi","en","mr","ta","bn","gu"], index=0)
        if st.button("Translate & Analyze"):
            lang_endpoint = st.secrets.get("AZURE_LANG_ENDPOINT","")
            lang_key = st.secrets.get("AZURE_LANG_KEY","")
            if not lang_endpoint or not lang_key:
                st.error("Azure Language keys not found in secrets.")
            else:
                phrases = azure_keyphrases(text, lang_endpoint, lang_key)
                sent = azure_sentiment_analysis(text, lang_endpoint, lang_key)
                st.write("Key phrases:", phrases or "—")
                st.write("Sentiment:", sent.get("sentiment"))
                st.write("Confidence:", sent.get("confidenceScores"))
                # Translate (best-effort using Microsoft Translator endpoint)
                try:
                    trans_url = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=" + target_lang
                    headers = {"Ocp-Apim-Subscription-Key": lang_key, "Content-Type":"application/json"}
                    body = [{"Text": text}]
                    r = requests.post(trans_url, headers=headers, json=body, timeout=10)
                    r.raise_for_status()
                    trans = r.json()[0]['translations'][0]['text']
                    st.write("Translation:", trans)
                except Exception:
                    st.info("Translator API unavailable or not configured for this key.")

        st.markdown("---")

        # Photo analyzer (vision)
        st.subheader("📷 Food Photo Analyzer")
        uploaded = st.file_uploader("Upload a food photo", type=["jpg","jpeg","png"])
        if uploaded is not None:
            img_bytes = uploaded.read()
            vision_endpoint = st.secrets.get("AZURE_VISION_ENDPOINT","")
            vision_key = st.secrets.get("AZURE_VISION_KEY","")
            if not vision_endpoint or not vision_key:
                st.error("Azure Vision keys not found in secrets.")
            else:
                with st.spinner("Analyzing image with Azure Vision..."):
                    result = azure_vision_analyze_image(img_bytes, vision_endpoint, vision_key)
                if "error" in result:
                    st.error("Vision API error: " + str(result.get("error")))
                else:
                    st.json(result)
                    tags = []
                    if "tags" in result:
                        tags += [t['name'] for t in result['tags']]
                    if "description" in result and result['description'].get('tags'):
                        tags += result['description']['tags']
                    tags = list(set(tags))
                    st.write("Detected tags:", tags or "—")
                    matches = find_closest_foods_from_tags(tags, df_food)
                    st.write("Closest matches from dataset:", matches or "—")
                    if matches:
                        chosen = st.selectbox("Choose dish to log", [m[0] for m in matches], key="photo_choice")
                        sers = st.number_input("Servings (1 serve=100g)", value=1.0, min_value=0.25, max_value=10.0, step=0.25, key="photo_sers")
                        if st.button("Add detected dish to today's entry"):
                            nut = calculate_nutrition_for_servings(df_food, chosen, sers)
                            if nut:
                                entry = {
                                    "username": u,
                                    "date": datetime.now().strftime("%Y-%m-%d"),
                                    "steps": 0,
                                    "calories": nut['calories'],
                                    "protein_g": nut['protein_g'],
                                    "fat_g": nut['fat_g'],
                                    "carbs_g": nut['carbs_g'],
                                    "notes": f"Logged via image: {chosen}"
                                }
                                append_log(entry)
                                st.success("Added to your log.")

# ------------------ end UI ------------------
