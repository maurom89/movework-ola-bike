# =========================================================
# 🚲 MOVEBIKE — STREAMLIT APP (OLA BIKE DEMAND FORECAST)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")

# ---------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="🚲 MoveBike — Smart Mobility",
    page_icon="🚲",
    layout="wide"
)

# ---------------------------------------------------------
# STYLE + BACKGROUND IMAGE
# ---------------------------------------------------------
st.markdown("""
<style>
/* Image de fond */
.stApp {
    background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                url("images/bike.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Titres */
.main-title {
    font-size: 64px;
    font-weight: 900;
    text-align: center;
    color: #00E5FF;
    text-shadow: 3px 3px 10px rgba(0,0,0,0.85);
    margin-top: 10px;
}

.subtitle {
    font-size: 26px;
    text-align: center;
    color: #FFFFFF;
    margin-top: -10px;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.75);
}

/* Glass box */
.glass-box {
    background: rgba(255,255,255,0.14);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 22px;
    margin-top: 18px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.18);
}

/* Boutons */
.stButton>button {
    background: linear-gradient(135deg, #00E5FF, #00B0FF);
    color: #001018;
    font-size: 18px;
    border-radius: 30px;
    height: 55px;
    width: 100%;
    font-weight: 800;
    box-shadow: 0 4px 15px rgba(0,229,255,0.55);
    border: 0;
}
.stButton>button:hover {
    filter: brightness(1.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.78);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Inputs */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] textarea {
    border-radius: 12px !important;
}

/* Charts container spacing */
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# UTILITAIRES
# ---------------------------------------------------------
def get_season_from_month(m: int) -> int:
    # 1=printemps, 2=été, 3=automne, 4=hiver
    if m in [3, 4, 5]:
        return 1
    if m in [6, 7, 8]:
        return 2
    if m in [9, 10, 11]:
        return 3
    return 4

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ---------------------------------------------------------
# CHARGEMENT DATA
# ---------------------------------------------------------
@st.cache_data
def load_data(path="ola.csv"):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # time features
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["season"] = df["month"].apply(get_season_from_month)

    return df

# ---------------------------------------------------------
# PRÉPARATION FEATURES + TRAIN MODELS
# ---------------------------------------------------------
@st.cache_resource
def prepare_and_train(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # --- KMeans cluster ---
    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(
        df[["hour", "weekday", "month", "temp", "humidity", "windspeed"]]
    )
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_cluster)

    # --- lags / rolling ---
    df["lag_1"] = df["count"].shift(1)
    df["lag_2"] = df["count"].shift(2)
    df["lag_24"] = df["count"].shift(24)
    df["rolling_3"] = df["count"].rolling(3).mean()
    df["rolling_24"] = df["count"].rolling(24).mean()
    df = df.dropna().reset_index(drop=True)

    features = [
        "season", "weather", "temp", "humidity", "windspeed",
        "hour", "weekday", "month", "cluster",
        "lag_1", "lag_2", "lag_24", "rolling_3", "rolling_24"
    ]

    # time split (no shuffle)
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train = train[features]
    y_train = train["count"]
    X_test = test[features]
    y_test = test["count"]

    # --- 4 models ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    lasso = Lasso(alpha=0.001)
    lasso.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gbr.fit(X_train, y_train)

    models = {
        "Régression Linéaire": lr,
        "Lasso": lasso,
        "Random Forest": rf,
        "Gradient Boosting": gbr
    }

    # evaluate
    rows = []
    preds = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        mae, rmse, r2 = evaluate_regression(y_test, y_pred)
        rows.append({"Modèle": name, "MAE": mae, "RMSE": rmse, "R²": r2})

    results_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = results_df.iloc[0]["Modèle"]
    best_model = models[best_name]
    best_pred = preds[best_name]

    return df, features, scaler_cluster, kmeans, models, results_df, best_name, best_model, X_test, y_test, best_pred

# ---------------------------------------------------------
# PREDICTION "MODE UBER" (saisie minimale)
# ---------------------------------------------------------
def predict_uber_like(df_feat, features, scaler_cluster, kmeans, best_model,
                      date_input, hour, weather, temp, humidity, windspeed):
    target_dt = pd.Timestamp(
        year=date_input.year, month=date_input.month, day=date_input.day, hour=hour
    )

    month = target_dt.month
    weekday = target_dt.weekday()
    season = get_season_from_month(month)

    # history for lags
    history = df_feat[df_feat["datetime"] < target_dt].sort_values("datetime")
    if len(history) < 24:
        raise ValueError("Historique insuffisant (min 24 heures) pour calculer lag/rolling.")

    lag_1 = history.iloc[-1]["count"]
    lag_2 = history.iloc[-2]["count"]

    lag_24_row = history[history["datetime"] == target_dt - pd.Timedelta(hours=24)]
    lag_24 = lag_24_row.iloc[0]["count"] if not lag_24_row.empty else history.iloc[-24]["count"]

    rolling_3 = history.tail(3)["count"].mean()
    rolling_24 = history.tail(24)["count"].mean()

    # cluster auto
    cluster_features = pd.DataFrame([{
        "hour": hour,
        "weekday": weekday,
        "month": month,
        "temp": temp,
        "humidity": humidity,
        "windspeed": windspeed
    }])
    cluster_scaled = scaler_cluster.transform(cluster_features)
    cluster = int(kmeans.predict(cluster_scaled)[0])

    X_new = pd.DataFrame([{
        "season": season,
        "weather": weather,
        "temp": temp,
        "humidity": humidity,
        "windspeed": windspeed,
        "hour": hour,
        "weekday": weekday,
        "month": month,
        "cluster": cluster,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_24": lag_24,
        "rolling_3": rolling_3,
        "rolling_24": rolling_24
    }])[features]

    pred = float(best_model.predict(X_new)[0])

    # infos utiles (optionnel)
    debug = {
        "datetime": target_dt,
        "season": season,
        "cluster": cluster,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_24": lag_24,
        "rolling_3": rolling_3,
        "rolling_24": rolling_24
    }
    return pred, debug

# ---------------------------------------------------------
# LOAD + TRAIN
# ---------------------------------------------------------
df_raw = load_data("ola.csv")
df_feat, FEATURES, scaler_cluster, kmeans, MODELS, results_df, best_name, best_model, X_test, y_test, best_pred = prepare_and_train(df_raw)

# ---------------------------------------------------------
# HEADER (BIG TITLE)
# ---------------------------------------------------------
st.markdown("""
<div style="margin-top: 18px;">
    <div class="main-title">🚲 MoveBike 🌍</div>
    <div class="subtitle">
        Prévision intelligente de la demande de trajets<br>
        🌆 Ville • 🏔️ Montagne • ♻️ Mobilité durable • 🤖 IA
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR NAV
# ---------------------------------------------------------
st.sidebar.markdown("## 🚲 MoveBike")
st.sidebar.markdown("### Navigation")
menu = st.sidebar.radio("", ["🏠 Accueil", "📊 Analyse", "🎯 Prédiction"])

# ---------------------------------------------------------
# PAGE: ACCUEIL
# ---------------------------------------------------------
if menu == "🏠 Accueil":
    st.markdown("""
    <div class="glass-box">
        <h2>✨ Application de référence — Mobilité & Data</h2>
        <p>
        <b>MoveBike</b> prédit la demande de trajets à vélo en s’appuyant sur :
        les facteurs temporels, la météo, et l’historique récent (calculé automatiquement).
        </p>
        <ul>
            <li>📈 Anticiper la demande et réduire l’attente</li>
            <li>🚲 Optimiser la flotte (ville / montagne)</li>
            <li>🌦️ Mesurer l’impact météo sur l’utilisation</li>
            <li>🤖 Utiliser un modèle performant (Gradient Boosting)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='glass-box'><h3>📊 Data</h3><p>Analyse & tendances</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass-box'><h3>🧠 Modèle</h3><p>ML + features temporelles</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='glass-box'><h3>🚀 Prédiction</h3><p>Usage simple (mode Uber)</p></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE: ANALYSE
# ---------------------------------------------------------
elif menu == "📊 Analyse":
    st.markdown(f"""
    <div class="glass-box">
        <h2>📊 Analyse & Performance</h2>
        <p>🏆 Modèle retenu automatiquement : <b>{best_name}</b></p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='glass-box'><h3>📈 Distribution de la demande</h3></div>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots()
        sns.histplot(df_raw["count"], bins=40, kde=True, ax=ax1)
        ax1.set_title("Distribution de count")
        st.pyplot(fig1)

    with right:
        st.markdown("<div class='glass-box'><h3>🧾 Résultats des modèles</h3></div>", unsafe_allow_html=True)
        show_df = results_df.copy()
        show_df["MAE"] = show_df["MAE"].map(lambda x: f"{x:.2f}")
        show_df["RMSE"] = show_df["RMSE"].map(lambda x: f"{x:.2f}")
        show_df["R²"] = show_df["R²"].map(lambda x: f"{x:.4f}")
        st.dataframe(show_df, use_container_width=True)

        # Bar chart RMSE
        figb, axb = plt.subplots()
        axb.bar(results_df["Modèle"], results_df["RMSE"])
        axb.set_title("Comparaison des modèles (RMSE)")
        axb.set_ylabel("RMSE")
        axb.tick_params(axis="x", rotation=20)
        st.pyplot(figb)

    st.markdown("<div class='glass-box'><h3>🔎 Réel vs Prédit (sur le test)</h3></div>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, best_pred, alpha=0.35)
    minv, maxv = float(y_test.min()), float(y_test.max())
    ax2.plot([minv, maxv], [minv, maxv], "r--")
    ax2.set_xlabel("Réel")
    ax2.set_ylabel("Prédit")
    ax2.set_title("Réel vs Prédit")
    st.pyplot(fig2)

# ---------------------------------------------------------
# PAGE: PRÉDICTION
# ---------------------------------------------------------
else:
    st.markdown("""
    <div class="glass-box">
        <h2>🎯 Prédire la demande (mode Uber)</h2>
        <p>
        Saisissez uniquement la <b>date</b>, l’<b>heure</b> et la <b>météo</b>.
        Les indicateurs historiques (lags, tendances) sont calculés automatiquement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        date_input = st.date_input("📅 Date")
        hour = st.slider("🕘 Heure", 0, 23, 9)
        weather = st.selectbox(
            "🌦️ Météo",
            [1, 2, 3, 4],
            format_func=lambda x: {1: "Clair ☀️", 2: "Nuageux ⛅", 3: "Pluie 🌧️", 4: "Orage ⛈️"}[x]
        )

    with col2:
        temp = st.number_input("🌡️ Température (°C)", value=25.0)
        humidity = st.number_input("💧 Humidité (%)", value=60.0)
        windspeed = st.number_input("🌬️ Vent", value=12.0)

    st.markdown("<div class='glass-box'><h3>🚲 Lance la prédiction</h3></div>", unsafe_allow_html=True)

    if st.button("🔮 Prédire maintenant"):
        try:
            pred, debug = predict_uber_like(
                df_feat=df_feat,
                features=FEATURES,
                scaler_cluster=scaler_cluster,
                kmeans=kmeans,
                best_model=best_model,
                date_input=date_input,
                hour=hour,
                weather=weather,
                temp=temp,
                humidity=humidity,
                windspeed=windspeed
            )

            st.markdown(f"""
            <div class="glass-box" style="text-align:center;">
                <h2>🌟 Résultat</h2>
                <div style="font-size:56px; font-weight:900; color:#00E5FF;">
                    {pred:.0f} trajets
                </div>
                <p>📅 {debug["datetime"]} • 🧭 Cluster {debug["cluster"]} • 🌡️ {temp}°C</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Détails techniques (optionnel)"):
                st.write(debug)

        except Exception as e:
            st.error(str(e))