import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="Bank Customer Retention Analytics", page_icon="📊")

# ── Custom CSS: Ultra Glassmorphism Theme ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── Animated mesh background ── */
.stApp {
    background: #050510;
    background-attachment: fixed;
    min-height: 100vh;
}
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(123,47,247,0.25) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(0,210,255,0.2) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 60% 30%, rgba(255,0,128,0.1) 0%, transparent 55%);
    z-index: 0;
    pointer-events: none;
    animation: bgPulse 10s ease-in-out infinite alternate;
}
@keyframes bgPulse {
    0%   { opacity: 0.8; }
    100% { opacity: 1; }
}

/* Floating orbs */
.stApp::after {
    content: '';
    position: fixed;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(123,47,247,0.12), transparent 70%);
    top: -100px; right: -100px;
    animation: float 8s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes float {
    0%, 100% { transform: translateY(0px) scale(1); }
    50%       { transform: translateY(30px) scale(1.05); }
}

/* ── Hero header banner ── */
.hero-banner {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 32px 40px;
    margin-bottom: 28px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow:
        0 0 0 1px rgba(123,47,247,0.15),
        0 20px 60px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(123,47,247,0.8), rgba(0,210,255,0.8), transparent);
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -60%; right: -10%;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(123,47,247,0.15), transparent 70%);
    pointer-events: none;
}

.glass-title {
    background: linear-gradient(90deg, #ffffff, #a78bfa, #00d2ff, #a78bfa, #ffffff);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shine 6s linear infinite;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}
@keyframes shine {
    to { background-position: 300% center; }
}
.glass-subtitle {
    color: rgba(160,174,192,0.9);
    font-size: 0.95rem;
    margin-top: 8px;
    font-weight: 400;
    letter-spacing: 0.3px;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(123,47,247,0.3), rgba(0,210,255,0.3));
    border: 1px solid rgba(0,210,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #00d2ff;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* ── Ultra glass metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 20px !important;
    padding: 24px 28px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    box-shadow:
        0 0 0 1px rgba(123,47,247,0.1),
        0 20px 40px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.08) !important;
    transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,210,255,0.6), transparent);
}
[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 60%;
    background: linear-gradient(to top, rgba(123,47,247,0.05), transparent);
    pointer-events: none;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-6px) scale(1.02) !important;
    border-color: rgba(0,210,255,0.3) !important;
    box-shadow:
        0 0 0 1px rgba(0,210,255,0.2),
        0 30px 60px rgba(0,0,0,0.5),
        0 0 40px rgba(123,47,247,0.2),
        inset 0 1px 0 rgba(255,255,255,0.12) !important;
}
[data-testid="metric-container"] label {
    color: rgba(160,174,192,0.8) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff, #00d2ff) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    filter: drop-shadow(0 0 12px rgba(0,210,255,0.4)) !important;
}

/* ── Section glass panels ── */
.glass-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.glass-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}

/* ── Neon glow divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(123,47,247,0.6), rgba(0,210,255,0.6), transparent) !important;
    margin: 24px 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    backdrop-filter: blur(10px) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: rgba(160,174,192,0.8) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 10px 20px !important;
    transition: all 0.25s ease !important;
    border: 1px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.05) !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.1) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7b2ff7, #00d2ff) !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(123,47,247,0.5), 0 0 0 1px rgba(0,210,255,0.3) !important;
    font-weight: 600 !important;
}

/* ── Sidebar ultra glass ── */
[data-testid="stSidebar"] {
    background: rgba(5,5,16,0.75) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 1px;
    background: linear-gradient(180deg, transparent, rgba(123,47,247,0.5), rgba(0,210,255,0.5), transparent);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #00d2ff !important;
    font-weight: 700 !important;
}

/* ── Headings ── */
h1, h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}
h2::after {
    content: '';
    display: block;
    width: 40px;
    height: 3px;
    background: linear-gradient(90deg, #7b2ff7, #00d2ff);
    border-radius: 2px;
    margin-top: 8px;
}

/* ── Dataframe ultra glass ── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    overflow: hidden !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

/* ── Info boxes ── */
[data-testid="stNotification"] {
    background: rgba(0,210,255,0.06) !important;
    border: 1px solid rgba(0,210,255,0.2) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 0 20px rgba(0,210,255,0.08), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #7b2ff7, #00d2ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 25px rgba(123,47,247,0.5), 0 0 0 1px rgba(0,210,255,0.2) !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    position: relative !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-3px) scale(1.03) !important;
    box-shadow: 0 8px 40px rgba(123,47,247,0.7), 0 0 0 1px rgba(0,210,255,0.4) !important;
}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
    background: linear-gradient(135deg, rgba(123,47,247,0.6), rgba(0,210,255,0.6)) !important;
    border: 1px solid rgba(0,210,255,0.3) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.01); border-radius: 3px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #7b2ff7, #00d2ff);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #9b4fff, #33dbff); }

/* ── Write / paragraph text ── */
p, .stMarkdown p { color: rgba(226,232,240,0.85) !important; }
</style>
""", unsafe_allow_html=True)

# Hero header
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🏦 AI-Powered Analytics</div>
    <p class="glass-title">📊 Customer Engagement & Retention Intelligence</p>
    <p class="glass-subtitle">Strategic Retention Dashboard &nbsp;·&nbsp; Risk Detector &nbsp;·&nbsp; ML Churn Prediction</p>
</div>
""", unsafe_allow_html=True)

# 2. Data Loading Functions
@st.cache_data
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)

    df['IsActiveMember'] = df['IsActiveMember'].astype(int)
    df['HasCrCard'] = df['HasCrCard'].astype(int)
    df['Exited'] = df['Exited'].astype(int)

    df['RSI'] = (np.minimum(df['NumOfProducts'] / 3, 1) * 40) + (df['IsActiveMember'] * 40) + (df['HasCrCard'] * 20)

    median_balance = df['Balance'].median()
    def segment_profile(row):
        if row['IsActiveMember'] == 1 and row['NumOfProducts'] >= 2:
            return 'Active Engaged'
        elif row['IsActiveMember'] == 0 and row['NumOfProducts'] == 1:
            return 'Inactive Disengaged'
        elif row['IsActiveMember'] == 1 and row['NumOfProducts'] == 1:
            return 'Active Low-Product'
        elif row['IsActiveMember'] == 0 and row['Balance'] > median_balance:
            return 'Inactive High-Balance'
        else:
            return 'Standard/Other'

    df['Engagement_Profile'] = df.apply(segment_profile, axis=1)
    return df

@st.cache_resource
def train_retention_model(df):
    features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    X = df[features].copy()
    y = df['Exited']

    le_geo = LabelEncoder()
    le_gen = LabelEncoder()
    X['Geography'] = le_geo.fit_transform(X['Geography'])
    X['Gender'] = le_gen.fit_transform(X['Gender'])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_geo, le_gen

DATA_PATH = "European_Bank.csv"

try:
    raw_df = load_and_process_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ Could not find your dataset at `{DATA_PATH}`. Please ensure your file is in the correct directory.")
    st.stop()

# Train model and generate churn probabilities
model, le_geo, le_gen = train_retention_model(raw_df)
X_pred = raw_df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].copy()
X_pred['Geography'] = le_geo.transform(X_pred['Geography'])
X_pred['Gender'] = le_gen.transform(X_pred['Gender'])
raw_df['Churn_Probability'] = (model.predict_proba(X_pred)[:, 1] * 100).round(2)

# 3. Sidebar Filtering Panel
st.sidebar.header("🎯 Filter Analytics Space")
geography = st.sidebar.multiselect("Select Region/Geography", options=list(raw_df['Geography'].unique()), default=list(raw_df['Geography'].unique()))
gender = st.sidebar.multiselect("Select Gender", options=list(raw_df['Gender'].unique()), default=list(raw_df['Gender'].unique()))
age_range = st.sidebar.slider("Age Dynamic Filter", int(raw_df['Age'].min()), int(raw_df['Age'].max()), (25, 65))

# Apply Dynamic Filters
df = raw_df[
    (raw_df['Geography'].isin(geography)) &
    (raw_df['Gender'].isin(gender)) &
    (raw_df['Age'].between(age_range[0], age_range[1]))
]

# 4. Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Executive KPIs & Overview", "🎯 Product Depth Matrix", "⚠️ At-Risk Premium Detector", "🤖 ML Churn Insights"])

# --- TAB 1: EXECUTIVE OVERVIEW ---
with tab1:
    st.subheader("Key Performance Indicators (KPIs)")

    churn_active = df[df['IsActiveMember'] == 1]['Exited'].mean()
    churn_inactive = df[df['IsActiveMember'] == 0]['Exited'].mean()
    err = churn_active / churn_inactive if churn_inactive > 0 else 0

    median_bal = df['Balance'].median()
    hbdr_denom = len(df[(df['Balance'] > median_bal) & (df['IsActiveMember'] == 0)])
    hbdr_num = len(df[(df['Balance'] > median_bal) & (df['IsActiveMember'] == 0) & (df['Exited'] == 1)])
    hbdr = (hbdr_num / hbdr_denom) * 100 if hbdr_denom > 0 else 0

    avg_rsi = df['RSI'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Engagement Retention Ratio (ERR)", value=f"{err:.2f}", help="Ratio of Active Churn vs Inactive Churn. Lower means active profiles stay longer.")
    col2.metric(label="High-Balance Disengagement Rate", value=f"{hbdr:.1f}%", help="Percentage of inactive premium customers who churned.")
    col3.metric(label="Avg Relationship Strength Index (RSI)", value=f"{avg_rsi:.1f} / 100", help="Aggregated health metric across the selected pool.")

    st.markdown("---")
    st.subheader("Retention Performance by Engineered Engagement Profile")
    profile_summary = df.groupby('Engagement_Profile').agg(
        Total_Customers=('CustomerId', 'count'),
        Churn_Rate=('Exited', 'mean')
    ).reset_index()
    profile_summary['Churn_Rate'] = (profile_summary['Churn_Rate'] * 100).round(2)

    fig_profile = px.bar(profile_summary, x='Engagement_Profile', y='Churn_Rate',
                         text=profile_summary['Churn_Rate'].map('{:,.1f}%'.format),
                         title="Churn Rate Across Behavioral Archetypes",
                         labels={'Churn_Rate': 'Churn Rate (%)'}, color='Engagement_Profile',
                         color_discrete_sequence=['#7b2ff7','#00d2ff','#ff0080','#00ff88','#ffaa00'],
                         template='plotly_dark')
    fig_profile.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0', family='Inter'),
        title=dict(font=dict(color='#ffffff', size=16, family='Inter'), x=0.01),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    fig_profile.update_traces(marker_line_width=0, textfont=dict(color='white'))
    st.plotly_chart(fig_profile, width='stretch')

# --- TAB 2: PRODUCT UTILIZATION IMPACT ---
with tab2:
    st.subheader("The Impact of Product Depth and Mix on Customer Loyalty")

    prod_summary = df.groupby('NumOfProducts').agg(
        Volume=('CustomerId', 'count'),
        Churn_Rate=('Exited', 'mean')
    ).reset_index()
    prod_summary['Churn_Rate'] = (prod_summary['Churn_Rate'] * 100).round(2)

    col_graph, col_insights = st.columns([2, 1])

    with col_graph:
        fig_prod = px.line(prod_summary, x='NumOfProducts', y='Churn_Rate', markers=True,
                           title="The Multi-Product Paradox Matrix",
                           labels={'Churn_Rate': 'Observed Churn Rate (%)', 'NumOfProducts': 'Number of Active Bank Products'},
                           template='plotly_dark', color_discrete_sequence=['#00d2ff'])
        fig_prod.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
            font=dict(color='#a0aec0', family='Inter'),
            title=dict(font=dict(color='#ffffff', size=16), x=0.01),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        fig_prod.update_traces(
            line=dict(width=3, color='#00d2ff'),
            marker=dict(size=12, color='#7b2ff7', line=dict(color='#00d2ff', width=2.5)),
            fill='tozeroy', fillcolor='rgba(0,210,255,0.05)'
        )
        st.plotly_chart(fig_prod, width='stretch')

    with col_insights:
        st.markdown("#### 💡 Strategic Product Insights")
        st.info("""
        * **Single-Product Vulnerability:** Customers with only 1 product typically experience the highest baseline silent churn.
        * **The Sweet Spot:** Moving a customer from 1 product to 2 products usually triggers a severe drop in churn risk.
        * **Cross-Selling Pitfall:** Be alert if churn spikes at 3 or 4 products. This often points to forced cross-selling or mismatched financial packages.
        """)

# --- TAB 3: HIGH-VALUE DISENGAGED DETECTOR ---
with tab3:
    st.subheader("🚨 Actionable Risk Mitigation: Premium Inactive Accounts")
    st.markdown("These high-net-worth customers are currently marked as **Inactive** but have **not yet churned (Exited = 0)**. They represent immediate retention targets for marketing cross-sell pipelines.")

    median_global_balance = df['Balance'].median()

    at_risk_premium = df[
        (df['IsActiveMember'] == 0) &
        (df['Balance'] > median_global_balance) &
        (df['Exited'] == 0)
    ].sort_values(by='Balance', ascending=False)

    st.write(f"**Identified At-Risk Premium Accounts in Current View:** {len(at_risk_premium)} customers")

    st.dataframe(at_risk_premium[[
        'CustomerId', 'Surname', 'CreditScore', 'Geography',
        'Age', 'Balance', 'NumOfProducts', 'RSI', 'Churn_Probability'
    ]], width='stretch')

    csv_data = at_risk_premium.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export At-Risk Target List for Retention Campaigns",
        data=csv_data,
        file_name="at_risk_premium_retention_targets.csv",
        mime="text/csv"
    )
with tab4:
    st.subheader("🤖 Machine Learning: Churn Drivers")
    
    # Feature Importance
    importances = model.feature_importances_
    feat_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                     title="What Drives Churn? (Model Feature Importance)",
                     color='Importance', color_continuous_scale='Plasma',
                     template='plotly_dark')
    fig_imp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0', family='Inter'),
        title=dict(font=dict(color='#ffffff', size=16), x=0.01),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
        coloraxis_colorbar=dict(tickcolor='#a0aec0', tickfont=dict(color='#a0aec0')),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    fig_imp.update_traces(marker_line_width=0)
    st.plotly_chart(fig_imp, width='stretch')
    
    st.info("""
    **Interpreting the Model:**
    The chart above shows which variables the AI uses most to predict churn.
    If **Age** or **NumOfProducts** is at the top, it confirms that behavioral patterns
    are more predictive than simple demographics.
    """)

    st.markdown("### 🔮 AI-Predicted Flight Risk")
    ml_at_risk = df[df['Exited'] == 0].sort_values(by='Churn_Probability', ascending=False).head(50)
    st.write("Top 50 customers the AI predicts will leave next:")
    st.dataframe(ml_at_risk[['CustomerId', 'Surname', 'Churn_Probability', 'RSI', 'Balance', 'NumOfProducts']])