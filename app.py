# app.py
import streamlit as st
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt

from ai_utils import (
    safe_load_json,
    safe_save_json,
    azure_keyphrases,
    azure_vision_analyze_image,
    find_closest_foods_from_tags,
    calculate_nutrition_for_servings,
)

st.set_page_config(page_title="Health & Nutrition Tracker", layout="wide")

# file paths
DATASET_PATH = "Indian_Food_Nutrition_Processed.csv"
USERS_PATH = "users.json"
PROFILES_PATH = "profiles.json"
LOG_CSV = "all_user_activity_log.csv"

# load dataset
@st.cache_data
def load_food_dataset(path):
    if not os.path.exists(path):
        st.error("Dataset not found: " + path)
        return pd.DataFrame()
    return pd.read_csv(path)

df_food = load_food_dataset(DATASET_PATH)

# load jsons
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

# session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

st.sidebar.title("Account")
menu = st.sidebar.selectbox("Menu", ["Login","Signup","Dashboard","AI Insights","Profile"])

# -------------------------
# AUTH: Signup
# -------------------------
if menu == "Signup":
    st.header("Create Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Create"):
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

# -------------------------
# AUTH: Login
# -------------------------
elif menu == "Login":
    st.header("Login")
    u = st.text_input("Username", key="login_user")
    p = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        if u in users and users[u] == hash_password(p):
            st.session_state.logged_in = True
            st.session_state.user = u
            st.success(f"Welcome back, {u}")
        else:
            st.error("Incorrect credentials")

# -------------------------
# PROFILE
# -------------------------
elif menu == "Profile":
    if not st.session_state.logged_in:
        st.info("Please login first.")
    else:
        u = st.session_state.user
        st.header("Profile & Goals")
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

# -------------------------
# DASHBOARD
# -------------------------
elif menu == "Dashboard":
    if not st.session_state.logged_in:
        st.info("Please login first.")
    else:
        u = st.session_state.user
        st.title("Dashboard")
        st.subheader("Log today's activity")
        col1, col2 = st.columns([2,1])
        with col1:
            st.write("Meals")
            b1 = st.text_input("Breakfast (type name or choose):", key="b1")
            s1 = st.number_input("Servings (1 serve = 100g)", min_value=0.25, max_value=10.0, value=1.0, step=0.25, key="s1")
            b2 = st.text_input("Lunch (type name or choose):", key="b2")
            s2 = st.number_input("Servings", min_value=0.25, max_value=10.0, value=1.0, step=0.25, key="s2")
            b3 = st.text_input("Dinner (type name or choose):", key="b3")
            s3 = st.number_input("Servings", min_value=0.25, max_value=10.0, value=1.0, step=0.25, key="s3")
            notes = st.text_area("Notes (optional)")
        with col2:
            steps = st.number_input("Steps today", min_value=0, max_value=200000, value=0, step=100)
            workout = st.selectbox("Workout type", ["None","Yoga","Walk","Run","Gym"])
            intensity = st.selectbox("Intensity", ["Low","Medium","High"])
            if st.button("Compute & Save"):
                total_cal = 0.0; total_prot=0.0; total_fat=0.0; total_carbs=0.0
                entries = [(b1,s1),(b2,s2),(b3,s3)]
                for dish, sv in entries:
                    if dish and str(dish).strip().lower()!="none":
                        nut = calculate_nutrition_for_servings(df_food, str(dish), float(sv))
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
        # show today's summary
        if os.path.exists(LOG_CSV):
            df_all = pd.read_csv(LOG_CSV)
            df_user = df_all[df_all['username']==u]
            if not df_user.empty:
                last = df_user.tail(7)
                st.subheader("Last entries")
                st.dataframe(last[['date','steps','calories','protein_g','fat_g','carbs_g']].reset_index(drop=True))
                fig, ax = plt.subplots(1,2, figsize=(10,3))
                last.plot(x='date', y='steps', ax=ax[0], marker='o', title='Steps (last entries)')
                last.plot(x='date', y='calories', ax=ax[1], marker='o', title='Calories (last entries)')
                st.pyplot(fig)
            else:
                st.info("No logs yet. Add today's entry!")

# -------------------------
# AI INSIGHTS (Daily)
# -------------------------
elif menu == "AI Insights":
    if not st.session_state.logged_in:
        st.info("Please login first.")
    else:
        u = st.session_state.user
        st.title("AI Insights")
        st.write("This page uses Azure AI services for food detection, meal analysis, and generating daily reports.")

        # DAILY AI REPORT
        st.subheader("📅 Daily AI-Generated Report")
        if st.button("Generate Daily Report"):
            stats, msg = generate_daily_stats(u, LOG_CSV)
            if not stats:
                st.info(msg)
            else:
                st.success("Daily Report:")
                st.write(f"### 📊 Stats for {stats['date']}")
                st.write(f"- **Calories:** {stats['calories']} kcal")
                st.write(f"- **Steps:** {stats['steps']}")
                st.write(f("- **Protein:** {stats['protein']} g"))
                st.write(f("- **Carbs:** {stats['carbs']} g"))
                st.write(f"- **Fats:** {stats['fats']} g")

        st.markdown("---")

        # DAILY DIET SUGGESTIONS
        st.subheader("🥗 AI Daily Diet & Fitness Suggestions")
        if st.button("Get Daily Suggestions"):
            profile = profiles.get(u, {})
            stats, msg = generate_daily_stats(u, LOG_CSV)
            if not stats:
                st.info(msg)
            else:
                sug = daily_suggestions(profile, stats)
                st.write("### 🔸 Personalized Daily Advice:")
                for s in sug:
                    st.write("• " + s)

        st.markdown("---")

        # FOOD PHOTO ANALYZER (Azure Vision)
        st.subheader("📷 Food Photo Analyzer (Azure Vision)")
        uploaded = st.file_uploader("Upload a food image", type=["jpg","jpeg","png"])
        if uploaded is not None:
            img_bytes = uploaded.read()
            vision_endpoint = st.secrets.get("AZURE_VISION_ENDPOINT", "")
            vision_key = st.secrets.get("AZURE_VISION_KEY", "")
            if not vision_endpoint or not vision_key:
                st.error("Azure Vision keys not set in secrets.")
            else:
                with st.spinner("Analyzing image..."):
                    result = azure_vision_analyze_image(img_bytes, vision_endpoint, vision_key)
                st.json(result)
                tags = []
                if "tags" in result:
                    tags += [t['name'] for t in result['tags']]
                if "description" in result and result['description'].get('tags'):
                    tags += result['description']['tags']
                tags = list(set(tags))
                st.write("Detected tags:", tags)
                matches = find_closest_foods_from_tags(tags, df_food)
                st.write("Closest matches from dataset:", matches)
                if matches:
                    chosen = st.selectbox("Choose dish to log", [m[0] for m in matches])
                    sers = st.number_input("Servings (1 serve=100g)", value=1.0, min_value=0.25, max_value=10.0, step=0.25)
                    if st.button("Add detected dish to today's entry"):
                        nut = calculate_nutrition_for_servings(df_food, chosen, sers)
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

# -------------------------
# Helper functions (daily)
# -------------------------
def generate_daily_stats(username: str, log_csv_path: str):
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

def daily_suggestions(profile: dict, stats: dict):
    suggestions = []
    if not stats:
        return ["No data logged today. Add your meals and steps first."]
    goal_steps = profile.get("steps_goal", 8000)
    if stats['steps'] < goal_steps:
        suggestions.append(f"Your steps today ({stats['steps']}) are below your goal ({goal_steps}). Try a short walk or a quick home workout.")
    else:
        suggestions.append("Great job! You reached your step goal today.")
    goal_cal = profile.get("calorie_goal", 2000)
    if stats['calories'] > goal_cal + 200:
        suggestions.append("Calorie intake today is higher than your goal. Consider lighter meals for dinner and reduce snacks.")
    elif stats['calories'] < goal_cal - 200:
        suggestions.append("Calorie intake is lower than your goal. Add a nutritious snack like nuts, milk, or fruit.")
    else:
        suggestions.append("Your calorie intake today is within a healthy range.")
    if stats['protein'] < 40:
        suggestions.append("Protein intake is low today. Add paneer, dal, sprouts, yogurt, or nuts.")
    else:
        suggestions.append("Protein intake looks good today.")
    if stats['carbs'] > 300:
        suggestions.append("High-carb day — balance with veggies and fiber.")
    if stats['fats'] > 80:
        suggestions.append("High fat intake — avoid fried foods for the next meal.")
    return suggestions
