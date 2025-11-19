# app.py
import streamlit as st
from datetime import datetime
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from ai_utils import safe_load_json, safe_save_json, azure_keyphrases, azure_vision_analyze_image, find_closest_foods_from_tags, calculate_nutrition_for_servings

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
    df = pd.read_csv(path)
    return df

df_food = load_food_dataset(DATASET_PATH)

# safe JSON load/save wrappers
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
        df = pd.DataFrame(columns=cols)
        df.to_csv(LOG_CSV, index=False)
    df = pd.read_csv(LOG_CSV)
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(LOG_CSV, index=False)

# auth
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

st.sidebar.title("Account")
menu = st.sidebar.selectbox("Menu", ["Login","Signup","Dashboard","AI Insights","Profile"])

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
                # calculate nutrition
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
                # plot steps and calories
                fig, ax = plt.subplots(1,2, figsize=(10,3))
                last.plot(x='date', y='steps', ax=ax[0], marker='o', title='Steps (last entries)')
                last.plot(x='date', y='calories', ax=ax[1], marker='o', title='Calories (last entries)')
                st.pyplot(fig)
            else:
                st.info("No logs yet. Add today's entry!")

elif menu == "AI Insights":
    if not st.session_state.logged_in:
        st.info("Please login first.")
    else:
        u = st.session_state.user
        st.title("AI Insights")
        # Weekly report
        st.subheader("Weekly AI-generated Report")
        if st.button("Generate Weekly Report"):
            from ai_utils import generate_weekly_stats  # we'll patch this quickly below if not exists
            try:
                stats, text = generate_weekly_stats(u, LOG_CSV)
            except Exception:
                # fallback simple summarizer
                import pandas as pd
                if not os.path.exists(LOG_CSV):
                    st.info("No logs yet.")
                else:
                    df = pd.read_csv(LOG_CSV, parse_dates=['date'])
                    df_user = df[df['username']==u]
                    if df_user.empty:
                        st.info("No logs yet.")
                    else:
                        today = pd.to_datetime(datetime.now().date())
                        week_start = today - pd.Timedelta(days=6)
                        df_week = df_user[(df_user['date'] >= week_start.strftime("%Y-%m-%d"))]
                        total_steps = int(df_week['steps'].sum())
                        avg_steps = int(df_week['steps'].mean())
                        total_cal = float(df_week['calories'].sum())
                        avg_cal = float(df_week['calories'].mean())
                        text = f"Weekly summary: total steps {total_steps}, avg steps {avg_steps}, total calories {round(total_cal,1)}, avg cal {round(avg_cal,1)}"
                        st.write(text)
            else:
                st.write(text)

        # Diet suggestions
        st.subheader("Diet Suggestions")
        if st.button("Get Diet Suggestions"):
            # load profile
            prof = profiles.get(u, {})
            try:
                from ai_utils import diet_suggestions_from_profile
                stats, _ = generate_weekly_stats(u, LOG_CSV)
                sugg = diet_suggestions_from_profile(prof, stats)
            except Exception:
                sugg = ["No suggestion available (no logs/profile)."]
            for s in sugg:
                st.write("• " + s)

        # Photo analyzer
        st.subheader("Food Photo Analyzer")
        uploaded = st.file_uploader("Upload a food photo", type=["jpg","jpeg","png"])
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

# small helper functions integrated from ai_utils (if not present)
# add generate_weekly_stats and diet_suggestions_from_profile here for completeness
def generate_weekly_stats(username: str, log_csv_path: str):
    if not os.path.exists(log_csv_path):
        return None, "No logs found."
    df = pd.read_csv(log_csv_path, parse_dates=['date'])
    if 'username' not in df.columns:
        return None, "Log file structure invalid."
    df_user = df[df['username']==username].copy()
    if df_user.empty:
        return None, "No log data available yet."
    today = pd.to_datetime(datetime.now().date())
    week_start = today - pd.Timedelta(days=6)
    df_user['date'] = pd.to_datetime(df_user['date'])
    df_week = df_user[(df_user['date'] >= week_start) & (df_user['date'] <= today)]
    if df_week.empty:
        return None, "No entries in the last 7 days."
    total_steps = int(df_week['steps'].sum())
    avg_steps = int(df_week['steps'].mean())
    total_cal = float(df_week['calories'].sum())
    avg_cal = float(df_week['calories'].mean())
    protein = df_week['protein_g'].sum() if 'protein_g' in df_week.columns else None
    fats = df_week['fat_g'].sum() if 'fat_g' in df_week.columns else None
    carbs = df_week['carbs_g'].sum() if 'carbs_g' in df_week.columns else None
    stats = {
        "total_steps": total_steps,
        "avg_daily_steps": avg_steps,
        "total_calories": round(total_cal,1),
        "avg_daily_calories": round(avg_cal,1),
        "protein_g": round(protein,1) if protein is not None else None,
        "fat_g": round(fats,1) if fats is not None else None,
        "carbs_g": round(carbs,1) if carbs is not None else None,
        "days_with_entries": int(df_week.shape[0])
    }
    summary = (
        f"Weekly Report ({week_start.date()} — {today.date()}):\n"
        f"- Logged on {stats['days_with_entries']} out of 7 days.\n"
        f"- Total steps: {stats['total_steps']} (avg {stats['avg_daily_steps']} /day).\n"
        f"- Total calories: {stats['total_calories']} kcal (avg {stats['avg_daily_calories']} kcal/day).\n"
    )
    if stats['protein_g'] is not None:
        summary += f"- Macronutrients (week total): Protein {stats['protein_g']} g, Fat {stats['fat_g']} g, Carbs {stats['carbs_g']} g.\n"
    suggestions = []
    if stats['avg_daily_steps'] < 7000:
        suggestions.append("Average steps are low — add short walks.")
    else:
        suggestions.append("Great job on steps this week!")
    if stats['avg_daily_calories'] > 2200:
        suggestions.append("Calories high — consider portion control and more veggies.")
    elif stats['avg_daily_calories'] < 1400:
        suggestions.append("Calories low — ensure adequate nutrition.")
    else:
        suggestions.append("Calorie intake looks reasonable.")
    return stats, summary + "\nRecommendations:\n- " + "\n- ".join(suggestions)

def diet_suggestions_from_profile(profile: dict, recent_stats: dict):
    suggestions = []
    if not profile:
        return ["No profile found to generate suggestions."]
    # protein guidance
    protein_weekly = recent_stats.get('protein_g') if recent_stats else None
    if protein_weekly:
        avg_protein = protein_weekly / 7
        if avg_protein < 50:
            suggestions.append("Increase high-protein vegetarian foods: paneer, lentils, soya, yogurt, nuts.")
        else:
            suggestions.append("Protein intake is adequate; maintain variety.")
    # calorie guidance
    avg_cal = recent_stats.get('avg_daily_calories') if recent_stats else None
    cal_goal = profile.get('calorie_goal')
    if avg_cal and cal_goal:
        if avg_cal > cal_goal:
            suggestions.append(f"Weekly average calories {avg_cal} exceed your goal {cal_goal} — reduce portions or avoid fried foods.")
        elif avg_cal < cal_goal - 300:
            suggestions.append("Calorie intake below goal — add nutritious snacks (nuts, milk, paneer).")
        else:
            suggestions.append("Calories aligned with your goal — good consistency!")
    # generic
    if profile.get('diet') == 'vegetarian':
        suggestions.append("Prefer dals, paneer, legumes, and whole grains for balanced nutrition.")
    return suggestions