import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    font-size: 4.3rem;
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
    <div class="hero-badge"></div>
    <p class="glass-title">📊 Customer Engagement & Retention Intelligence</p>
    <p class="glass-subtitle">Strategic Retention Dashboard &nbsp;·&nbsp; Risk Detector &nbsp;·&nbsp; ML Churn Prediction</p>
</div>
""", unsafe_allow_html=True)

# 2. Data Loading Functions
@st.cache_data
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)

    # Binary variable consistency enforcement
    df['IsActiveMember'] = df['IsActiveMember'].astype(int)
    df['HasCrCard'] = df['HasCrCard'].astype(int)
    df['Exited'] = df['Exited'].astype(int)

    # RSI computation
    df['RSI'] = (np.minimum(df['NumOfProducts'] / 3, 1) * 40) + (df['IsActiveMember'] * 40) + (df['HasCrCard'] * 20)

    # Engagement profile classification
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

    # Salary-Balance mismatch flag (high earner, low balance)
    salary_q75 = df['EstimatedSalary'].quantile(0.75)
    balance_q25 = df['Balance'].quantile(0.25)
    df['Salary_Balance_Mismatch'] = (
        (df['EstimatedSalary'] >= salary_q75) & (df['Balance'] <= balance_q25)
    ).astype(int)

    return df

@st.cache_resource
def train_retention_model(df):
    features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
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
X_pred = raw_df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                  'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].copy()
X_pred['Geography'] = le_geo.transform(X_pred['Geography'])
X_pred['Gender'] = le_gen.transform(X_pred['Gender'])
raw_df['Churn_Probability'] = (model.predict_proba(X_pred)[:, 1] * 100).round(2)

# 3. Sidebar Filtering Panel
st.sidebar.header("🎯 Filter Analytics Space")

st.sidebar.markdown("**Engagement Filters**")
geography = st.sidebar.multiselect("Region / Geography", options=list(raw_df['Geography'].unique()), default=list(raw_df['Geography'].unique()))
gender = st.sidebar.multiselect("Gender", options=list(raw_df['Gender'].unique()), default=list(raw_df['Gender'].unique()))
age_range = st.sidebar.slider("Age Range", int(raw_df['Age'].min()), int(raw_df['Age'].max()), (25, 65))

st.sidebar.markdown("---")
st.sidebar.markdown("**Product Count Filter**")
prod_range = st.sidebar.slider("Number of Products", int(raw_df['NumOfProducts'].min()), int(raw_df['NumOfProducts'].max()),
                                (int(raw_df['NumOfProducts'].min()), int(raw_df['NumOfProducts'].max())))

st.sidebar.markdown("---")
st.sidebar.markdown("**Balance Threshold**")
bal_range = st.sidebar.slider("Balance (€)", int(raw_df['Balance'].min()), int(raw_df['Balance'].max()),
                               (int(raw_df['Balance'].min()), int(raw_df['Balance'].max())), step=1000)

st.sidebar.markdown("---")
st.sidebar.markdown("**Salary Threshold**")
sal_range = st.sidebar.slider("Estimated Salary (€)", int(raw_df['EstimatedSalary'].min()), int(raw_df['EstimatedSalary'].max()),
                               (int(raw_df['EstimatedSalary'].min()), int(raw_df['EstimatedSalary'].max())), step=1000)

# Apply Dynamic Filters
df = raw_df[
    (raw_df['Geography'].isin(geography)) &
    (raw_df['Gender'].isin(gender)) &
    (raw_df['Age'].between(age_range[0], age_range[1])) &
    (raw_df['NumOfProducts'].between(prod_range[0], prod_range[1])) &
    (raw_df['Balance'].between(bal_range[0], bal_range[1])) &
    (raw_df['EstimatedSalary'].between(sal_range[0], sal_range[1]))
]

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Customers", f"{len(df):,}", delta=f"{len(df)-len(raw_df):,} from total")

# 4. Tabs Layout
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Data Validation",
    "🚀 Executive KPIs & Engagement",
    "🎯 Product Utilization",
    "💰 Financial vs Engagement",
    "🤖 ML Churn Insights",
    "💎 Retention Strength"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 0: DATA INGESTION & VALIDATION
# ═══════════════════════════════════════════════════════════════════
with tab0:
    st.subheader("Dataset Load Confirmation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(raw_df):,}")
    c2.metric("Total Features", f"{raw_df.shape[1]}")
    c3.metric("Missing Values", f"{raw_df.isnull().sum().sum()}")
    c4.metric("Duplicate Rows", f"{raw_df.duplicated().sum()}")

    st.markdown("---")

    # ── Binary Variable Consistency ──────────────────────────────
    st.subheader("Binary Variable Consistency Check")
    binary_cols = ['IsActiveMember', 'HasCrCard', 'Exited']
    consistency_rows = []
    for col in binary_cols:
        unique_vals = sorted(raw_df[col].unique().tolist())
        is_valid = set(unique_vals).issubset({0, 1})
        consistency_rows.append({
            'Field': col,
            'Unique Values': str(unique_vals),
            'Expected': '[0, 1]',
            'Status': '✅ Valid' if is_valid else '❌ Invalid'
        })
    st.dataframe(pd.DataFrame(consistency_rows), width='stretch')

    st.markdown("---")

    # ── Engagement & Product Field Validation ────────────────────
    st.subheader("Engagement & Product Field Validation")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**IsActiveMember Distribution**")
        act_counts = raw_df['IsActiveMember'].value_counts().reset_index()
        act_counts.columns = ['Status', 'Count']
        act_counts['Status'] = act_counts['Status'].map({1: 'Active (1)', 0: 'Inactive (0)'})
        fig_act = px.pie(act_counts, names='Status', values='Count',
                         color_discrete_sequence=['#00d2ff', '#7b2ff7'],
                         template='plotly_dark', hole=0.5)
        fig_act.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='#a0aec0'),
                               legend=dict(bgcolor='rgba(0,0,0,0)'),
                               margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_act, width='stretch')

    with col_r:
        st.markdown("**NumOfProducts Distribution**")
        prod_counts = raw_df['NumOfProducts'].value_counts().sort_index().reset_index()
        prod_counts.columns = ['Products', 'Count']
        fig_prod_val = px.bar(prod_counts, x='Products', y='Count',
                              color='Count', color_continuous_scale='Plasma',
                              template='plotly_dark')
        fig_prod_val.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(255,255,255,0.02)',
                                    font=dict(color='#a0aec0'),
                                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                                    margin=dict(l=10, r=10, t=20, b=10))
        fig_prod_val.update_traces(marker_line_width=0)
        st.plotly_chart(fig_prod_val, width='stretch')

    st.markdown("---")

    # ── Churn Label Accuracy ─────────────────────────────────────
    st.subheader("Churn Labeling Accuracy")
    churn_dist = raw_df['Exited'].value_counts().reset_index()
    churn_dist.columns = ['Exited', 'Count']
    churn_dist['Label'] = churn_dist['Exited'].map({0: 'Retained (0)', 1: 'Churned (1)'})
    churn_dist['Pct'] = (churn_dist['Count'] / len(raw_df) * 100).round(2)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.dataframe(churn_dist[['Label', 'Count', 'Pct']].rename(columns={'Pct': '% of Total'}), width='stretch')
        imbalance_ratio = churn_dist['Count'].max() / churn_dist['Count'].min()
        if imbalance_ratio > 3:
            st.warning(f"⚠️ Class imbalance detected (ratio {imbalance_ratio:.1f}:1). Consider oversampling before ML training.")
        else:
            st.success(f"✅ Class balance acceptable (ratio {imbalance_ratio:.1f}:1).")
    with col_b:
        fig_churn_val = px.bar(churn_dist, x='Label', y='Count',
                                text=churn_dist['Pct'].map('{:.1f}%'.format),
                                color='Label',
                                color_discrete_sequence=['#00ff88', '#ff4466'],
                                template='plotly_dark',
                                title="Churn Label Distribution")
        fig_churn_val.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(255,255,255,0.02)',
                                     font=dict(color='#a0aec0'),
                                     title=dict(font=dict(color='#ffffff', size=15), x=0.01),
                                     xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                                     yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                                     showlegend=False,
                                     margin=dict(l=10, r=10, t=50, b=10))
        fig_churn_val.update_traces(marker_line_width=0, textfont=dict(color='white'))
        st.plotly_chart(fig_churn_val, width='stretch')

    st.markdown("---")
    st.subheader("Field Summary Statistics")
    st.dataframe(raw_df.describe().T.round(2), width='stretch')


# ═══════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE KPIs & ENGAGEMENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Key Performance Indicators (KPIs)")

    # ── KPI Calculations ─────────────────────────────────────────
    # 1. Engagement Retention Ratio (ERR)
    churn_active   = df[df['IsActiveMember'] == 1]['Exited'].mean()
    churn_inactive = df[df['IsActiveMember'] == 0]['Exited'].mean()
    err = churn_active / churn_inactive if churn_inactive > 0 else 0

    # 2. Product Depth Index (PDI) — churn rate of 1-product vs 2+-product customers
    churn_single = df[df['NumOfProducts'] == 1]['Exited'].mean()
    churn_multi  = df[df['NumOfProducts'] >= 2]['Exited'].mean()
    pdi = churn_single / churn_multi if churn_multi > 0 else 0

    # 3. High-Balance Disengagement Rate (HBDR)
    median_bal = df['Balance'].median()
    hbdr_denom = len(df[(df['Balance'] > median_bal) & (df['IsActiveMember'] == 0)])
    hbdr_num   = len(df[(df['Balance'] > median_bal) & (df['IsActiveMember'] == 0) & (df['Exited'] == 1)])
    hbdr = (hbdr_num / hbdr_denom) * 100 if hbdr_denom > 0 else 0

    # 4. Credit Card Stickiness Score (CCSS) — churn rate spread: no card vs card holder
    churn_no_card   = df[df['HasCrCard'] == 0]['Exited'].mean()
    churn_with_card = df[df['HasCrCard'] == 1]['Exited'].mean()
    ccss = (churn_no_card - churn_with_card) * 100  # positive = card reduces churn

    # 5. Relationship Strength Index (RSI)
    avg_rsi = df['RSI'].mean()

    # ── KPI Display — Row 1 ──────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Engagement Retention Ratio",
        f"{err:.2f}x",
        help="Active churn ÷ Inactive churn. Lower ratio = active members stay significantly longer."
    )
    col2.metric(
        "Product Depth Index",
        f"{pdi:.2f}x",
        help="Single-product churn ÷ Multi-product churn. Higher = stronger loyalty lift from cross-selling."
    )
    col3.metric(
        "High-Balance Disengagement Rate",
        f"{hbdr:.1f}%",
        help="% of inactive premium-balance customers who churned — direct premium revenue risk."
    )

    # ── KPI Display — Row 2 ──────────────────────────────────────
    col4, col5, _ = st.columns(3)
    col4.metric(
        "Credit Card Stickiness Score",
        f"{ccss:+.1f} pp",
        help="Churn rate of non-card holders minus card holders (percentage points). Positive = card ownership reduces churn."
    )
    col5.metric(
        "Relationship Strength Index",
        f"{avg_rsi:.1f} / 100",
        help="Composite score (activity × product count × card ownership). Higher = stronger retention bond."
    )

    st.markdown("---")

    # ── Engagement Classification ────────────────────────────────
    st.subheader("Engagement Classification — Behavioral Archetypes")

    profile_summary = df.groupby('Engagement_Profile').agg(
        Total_Customers=('CustomerId', 'count'),
        Churn_Rate=('Exited', 'mean'),
        Avg_RSI=('RSI', 'mean'),
        Avg_Balance=('Balance', 'mean')
    ).reset_index()
    profile_summary['Churn_Rate'] = (profile_summary['Churn_Rate'] * 100).round(2)
    profile_summary['Avg_RSI']    = profile_summary['Avg_RSI'].round(1)
    profile_summary['Avg_Balance'] = profile_summary['Avg_Balance'].round(0)

    col_bar, col_sunburst = st.columns(2)

    with col_bar:
        fig_profile = px.bar(profile_summary, x='Engagement_Profile', y='Churn_Rate',
                             text=profile_summary['Churn_Rate'].map('{:,.1f}%'.format),
                             title="Churn Rate Across Behavioral Archetypes",
                             labels={'Churn_Rate': 'Churn Rate (%)'},
                             color='Engagement_Profile',
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

    with col_sunburst:
        fig_sun = px.sunburst(
            profile_summary, path=['Engagement_Profile'], values='Total_Customers',
            color='Churn_Rate', color_continuous_scale='RdYlGn_r',
            title="Profile Size & Churn Intensity",
            template='plotly_dark'
        )
        fig_sun.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0aec0'),
            title=dict(font=dict(color='#ffffff', size=16), x=0.01),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_sun, width='stretch')

    st.markdown("#### Profile Reference Guide")
    st.info("""
    | Profile | Definition | Risk Level |
    |---|---|---|
    | **Active Engaged** | Active member with ≥2 products | 🟢 Low |
    | **Active Low-Product** | Active member with only 1 product | 🟡 Medium |
    | **Inactive Disengaged** | Inactive member with only 1 product | 🔴 High |
    | **Inactive High-Balance** | Inactive member holding above-median balance | 🟠 Premium Risk |
    | **Standard/Other** | All remaining customers | ⚪ Baseline |
    """)

    st.markdown("---")
    st.subheader("Profile Summary Table")
    st.dataframe(profile_summary.rename(columns={
        'Total_Customers': 'Customers',
        'Churn_Rate': 'Churn Rate (%)',
        'Avg_RSI': 'Avg RSI',
        'Avg_Balance': 'Avg Balance (€)'
    }), width='stretch')


# ═══════════════════════════════════════════════════════════════════
# TAB 2: PRODUCT UTILIZATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Product Utilization Analysis — Depth vs Churn Relationship")

    prod_summary = df.groupby('NumOfProducts').agg(
        Volume=('CustomerId', 'count'),
        Churn_Rate=('Exited', 'mean')
    ).reset_index()
    prod_summary['Churn_Rate'] = (prod_summary['Churn_Rate'] * 100).round(2)

    col_graph, col_insights = st.columns([2, 1])

    with col_graph:
        fig_prod = px.line(prod_summary, x='NumOfProducts', y='Churn_Rate', markers=True,
                           title="Product Depth vs Churn Rate (The Multi-Product Paradox)",
                           labels={'Churn_Rate': 'Observed Churn Rate (%)', 'NumOfProducts': 'Number of Active Products'},
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
        * **Single-Product Vulnerability:** Customers with only 1 product experience the highest baseline churn.
        * **The Sweet Spot:** Moving from 1 → 2 products typically triggers a sharp drop in churn risk.
        * **Cross-Selling Pitfall:** A churn spike at 3–4 products may signal forced or mismatched packages.
        """)

    st.markdown("---")

    # ── Single vs Multi-Product Retention ───────────────────────
    st.subheader("Single-Product vs Multi-Product Retention Comparison")

    df_prod_class = df.copy()
    df_prod_class['Product_Tier'] = df_prod_class['NumOfProducts'].apply(
        lambda x: 'Single Product' if x == 1 else 'Multi-Product'
    )

    retention_compare = df_prod_class.groupby('Product_Tier').agg(
        Customers=('CustomerId', 'count'),
        Churn_Rate=('Exited', 'mean'),
        Avg_RSI=('RSI', 'mean'),
        Avg_Balance=('Balance', 'mean')
    ).reset_index()
    retention_compare['Churn_Rate'] = (retention_compare['Churn_Rate'] * 100).round(2)
    retention_compare['Avg_RSI']    = retention_compare['Avg_RSI'].round(1)

    col_sp, col_mp = st.columns(2)

    with col_sp:
        fig_tier = px.bar(retention_compare, x='Product_Tier', y='Churn_Rate',
                          text=retention_compare['Churn_Rate'].map('{:.1f}%'.format),
                          color='Product_Tier',
                          color_discrete_sequence=['#ff0080', '#00ff88'],
                          title="Churn Rate: Single vs Multi-Product",
                          template='plotly_dark')
        fig_tier.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
            font=dict(color='#a0aec0'),
            title=dict(font=dict(color='#ffffff', size=15), x=0.01),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            showlegend=False, margin=dict(l=10, r=10, t=50, b=10)
        )
        fig_tier.update_traces(marker_line_width=0, textfont=dict(color='white'))
        st.plotly_chart(fig_tier, width='stretch')

    with col_mp:
        st.dataframe(retention_compare.rename(columns={
            'Churn_Rate': 'Churn Rate (%)',
            'Avg_RSI': 'Avg RSI',
            'Avg_Balance': 'Avg Balance (€)'
        }), width='stretch')

    st.markdown("---")

    # ── Product Depth × Activity × Churn Heatmap ─────────────────
    st.subheader("Product Depth × Activity Status — Churn Heatmap")

    heat_data = df.groupby(['NumOfProducts', 'IsActiveMember'])['Exited'].mean().reset_index()
    heat_data['IsActiveMember'] = heat_data['IsActiveMember'].map({1: 'Active', 0: 'Inactive'})
    heat_data['Churn_Rate'] = (heat_data['Exited'] * 100).round(2)
    heat_pivot = heat_data.pivot(index='IsActiveMember', columns='NumOfProducts', values='Churn_Rate')

    fig_heat = px.imshow(
        heat_pivot, text_auto='.1f',
        color_continuous_scale='RdYlGn_r',
        title="Churn Rate (%) by Activity Status and Product Count",
        template='plotly_dark',
        aspect='auto'
    )
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0'),
        title=dict(font=dict(color='#ffffff', size=16), x=0.01),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_heat, width='stretch')


# ═══════════════════════════════════════════════════════════════════
# TAB 3: FINANCIAL COMMITMENT vs ENGAGEMENT
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Financial Commitment vs Engagement Analysis")

    # ── Balance × Activity Cross-Analysis ───────────────────────
    st.markdown("### Balance vs Activity Status — Cross Analysis")

    bal_act = df.groupby('IsActiveMember').agg(
        Customers=('CustomerId', 'count'),
        Avg_Balance=('Balance', 'mean'),
        Median_Balance=('Balance', 'median'),
        Churn_Rate=('Exited', 'mean')
    ).reset_index()
    bal_act['IsActiveMember'] = bal_act['IsActiveMember'].map({1: 'Active', 0: 'Inactive'})
    bal_act['Churn_Rate'] = (bal_act['Churn_Rate'] * 100).round(2)

    col_bal1, col_bal2 = st.columns(2)
    with col_bal1:
        fig_bal_box = px.box(df, x=df['IsActiveMember'].map({1: 'Active', 0: 'Inactive'}),
                             y='Balance', color=df['IsActiveMember'].map({1: 'Active', 0: 'Inactive'}),
                             color_discrete_sequence=['#00d2ff', '#7b2ff7'],
                             title="Balance Distribution by Activity Status",
                             template='plotly_dark')
        fig_bal_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
            font=dict(color='#a0aec0'),
            title=dict(font=dict(color='#ffffff', size=15), x=0.01),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Activity Status'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            showlegend=False, margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_bal_box, width='stretch')

    with col_bal2:
        fig_bal_churn = px.bar(bal_act, x='IsActiveMember', y='Avg_Balance',
                               color='Churn_Rate', color_continuous_scale='RdYlGn_r',
                               text=bal_act['Avg_Balance'].map('€{:,.0f}'.format),
                               title="Avg Balance & Churn Rate by Activity",
                               template='plotly_dark')
        fig_bal_churn.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
            font=dict(color='#a0aec0'),
            title=dict(font=dict(color='#ffffff', size=15), x=0.01),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        fig_bal_churn.update_traces(marker_line_width=0, textfont=dict(color='white'))
        st.plotly_chart(fig_bal_churn, width='stretch')

    st.markdown("---")

    # ── Salary–Balance Mismatch Detection ───────────────────────
    st.markdown("### Salary–Balance Mismatch Detection")
    st.markdown("Customers with **top-quartile salary** but **bottom-quartile balance** — high earners who aren't depositing.")

    mismatch_summary = df.groupby('Salary_Balance_Mismatch').agg(
        Customers=('CustomerId', 'count'),
        Churn_Rate=('Exited', 'mean'),
        Avg_Salary=('EstimatedSalary', 'mean'),
        Avg_Balance=('Balance', 'mean')
    ).reset_index()
    mismatch_summary['Label'] = mismatch_summary['Salary_Balance_Mismatch'].map({0: 'No Mismatch', 1: 'Mismatch Detected'})
    mismatch_summary['Churn_Rate'] = (mismatch_summary['Churn_Rate'] * 100).round(2)

    col_mis1, col_mis2 = st.columns([1, 2])
    with col_mis1:
        mismatch_count = int(mismatch_summary[mismatch_summary['Salary_Balance_Mismatch'] == 1]['Customers'].sum())
        mismatch_churn = mismatch_summary[mismatch_summary['Salary_Balance_Mismatch'] == 1]['Churn_Rate'].values
        st.metric("Mismatch Customers", f"{mismatch_count:,}")
        if len(mismatch_churn) > 0:
            st.metric("Their Churn Rate", f"{mismatch_churn[0]:.1f}%")
        st.dataframe(mismatch_summary[['Label', 'Customers', 'Churn_Rate', 'Avg_Salary', 'Avg_Balance']].rename(
            columns={'Churn_Rate': 'Churn %', 'Avg_Salary': 'Avg Salary (€)', 'Avg_Balance': 'Avg Balance (€)'}
        ), width='stretch')

    with col_mis2:
        fig_scatter = px.scatter(
            df.sample(min(2000, len(df)), random_state=42),
            x='EstimatedSalary', y='Balance',
            color=df.sample(min(2000, len(df)), random_state=42)['Exited'].map({0: 'Retained', 1: 'Churned'}),
            symbol=df.sample(min(2000, len(df)), random_state=42)['Salary_Balance_Mismatch'].map({0: 'Normal', 1: 'Mismatch'}),
            color_discrete_map={'Retained': '#00d2ff', 'Churned': '#ff4466'},
            title="Salary vs Balance — Churn & Mismatch Overlay",
            template='plotly_dark',
            opacity=0.6
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
            font=dict(color='#a0aec0'),
            title=dict(font=dict(color='#ffffff', size=15), x=0.01),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Estimated Salary (€)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Balance (€)'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_scatter, width='stretch')

    st.markdown("---")

    # ── At-Risk Premium Customers ────────────────────────────────
    st.markdown("### 🚨 At-Risk Premium Customer Identification")
    st.markdown("Inactive customers with above-median balance who have **not yet churned** — immediate retention targets.")

    median_global_balance = df['Balance'].median()
    at_risk_premium = df[
        (df['IsActiveMember'] == 0) &
        (df['Balance'] > median_global_balance) &
        (df['Exited'] == 0)
    ].sort_values(by='Balance', ascending=False)

    st.write(f"**Identified At-Risk Premium Accounts:** {len(at_risk_premium)} customers")
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


# ═══════════════════════════════════════════════════════════════════
# TAB 4: ML CHURN INSIGHTS
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🤖 Machine Learning: Churn Drivers")

    importances = model.feature_importances_
    feat_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                  'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
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


# ═══════════════════════════════════════════════════════════════════
# TAB 5: RETENTION STRENGTH ASSESSMENT
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("💎 Retention Strength Index (RSI) Scoring Panel")
    st.markdown("RSI scores each customer 0–100 based on activity, product count, and card ownership. Higher = stronger retention bond.")

    def rsi_tier(score):
        if score >= 80:   return '🟢 Champion'
        elif score >= 60: return '🔵 Loyal'
        elif score >= 40: return '🟡 At Risk'
        else:             return '🔴 Critical'

    df = df.copy()
    df['RSI_Tier'] = df['RSI'].apply(rsi_tier)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Champions",  len(df[df['RSI_Tier'] == '🟢 Champion']),  help="RSI ≥ 80")
    c2.metric("🔵 Loyal",      len(df[df['RSI_Tier'] == '🔵 Loyal']),      help="RSI 60–79")
    c3.metric("🟡 At Risk",    len(df[df['RSI_Tier'] == '🟡 At Risk']),    help="RSI 40–59")
    c4.metric("🔴 Critical",   len(df[df['RSI_Tier'] == '🔴 Critical']),   help="RSI < 40")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        fig_rsi_dist = px.histogram(df, x='RSI', nbins=30, color='RSI_Tier',
                                    title="RSI Score Distribution by Tier",
                                    color_discrete_map={
                                        '🟢 Champion': '#00ff88',
                                        '🔵 Loyal':    '#00d2ff',
                                        '🟡 At Risk':  '#ffaa00',
                                        '🔴 Critical': '#ff4466'
                                    },
                                    template='plotly_dark')
        fig_rsi_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
            font=dict(color='#a0aec0', family='Inter'),
            title=dict(font=dict(color='#ffffff', size=16), x=0.01),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=10),
            bargap=0.05
        )
        st.plotly_chart(fig_rsi_dist, width='stretch')

    with col_r:
        tier_churn = df.groupby('RSI_Tier').agg(
            Churn_Rate=('Exited', 'mean'),
            Count=('CustomerId', 'count')
        ).reset_index()
        tier_churn['Churn_Rate'] = (tier_churn['Churn_Rate'] * 100).round(2)

        fig_tier = px.bar(tier_churn, x='RSI_Tier', y='Churn_Rate',
                          text=tier_churn['Churn_Rate'].map('{:.1f}%'.format),
                          title="Churn Rate by RSI Tier",
                          color='RSI_Tier',
                          color_discrete_map={
                              '🟢 Champion': '#00ff88',
                              '🔵 Loyal':    '#00d2ff',
                              '🟡 At Risk':  '#ffaa00',
                              '🔴 Critical': '#ff4466'
                          },
                          template='plotly_dark')
        fig_tier.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
            font=dict(color='#a0aec0', family='Inter'),
            title=dict(font=dict(color='#ffffff', size=16), x=0.01),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=False),
            showlegend=False, margin=dict(l=10, r=10, t=50, b=10)
        )
        fig_tier.update_traces(marker_line_width=0, textfont=dict(color='white'))
        st.plotly_chart(fig_tier, width='stretch')

    st.markdown("---")

    # ── Sticky Customer Profiles ─────────────────────────────────
    st.subheader("Sticky Customer Profiles — Zero-Churn Champions")
    st.markdown("Customers who are **Champions (RSI ≥ 80)**, **retained**, and **multi-product** — the gold standard for loyalty.")

    sticky = df[
        (df['RSI_Tier'] == '🟢 Champion') &
        (df['Exited'] == 0) &
        (df['NumOfProducts'] >= 2)
    ]

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Sticky Customers", f"{len(sticky):,}")
    sc2.metric("Avg Balance (€)", f"€{sticky['Balance'].mean():,.0f}" if len(sticky) > 0 else "N/A")
    sc3.metric("Avg Tenure (yrs)", f"{sticky['Tenure'].mean():.1f}" if len(sticky) > 0 else "N/A")

    if len(sticky) > 0:
        sticky_geo = sticky.groupby('Geography').size().reset_index(name='Count')
        fig_sticky = px.pie(sticky_geo, names='Geography', values='Count',
                            title="Sticky Customer Geography Mix",
                            color_discrete_sequence=['#00ff88', '#00d2ff', '#7b2ff7'],
                            template='plotly_dark', hole=0.4)
        fig_sticky.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0aec0'),
            title=dict(font=dict(color='#ffffff', size=15), x=0.01),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_sticky, width='stretch')

    st.markdown("---")

    # ── Churn Stability Across Engagement Tiers ──────────────────
    st.subheader("Churn Stability Across Engagement Tiers")

    tier_stability = df.groupby(['RSI_Tier', 'Engagement_Profile']).agg(
        Customers=('CustomerId', 'count'),
        Churn_Rate=('Exited', 'mean')
    ).reset_index()
    tier_stability['Churn_Rate'] = (tier_stability['Churn_Rate'] * 100).round(2)

    fig_stability = px.bar(
        tier_stability, x='RSI_Tier', y='Churn_Rate',
        color='Engagement_Profile', barmode='group',
        title="Churn Rate by RSI Tier & Engagement Profile",
        color_discrete_sequence=['#7b2ff7','#00d2ff','#ff0080','#00ff88','#ffaa00'],
        template='plotly_dark',
        text='Churn_Rate'
    )
    fig_stability.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0'),
        title=dict(font=dict(color='#ffffff', size=16), x=0.01),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Churn Rate (%)'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    fig_stability.update_traces(marker_line_width=0, texttemplate='%{text:.1f}%', textposition='outside',
                                 textfont=dict(color='white', size=9))
    st.plotly_chart(fig_stability, width='stretch')

    st.markdown("---")

    # ── Engagement Thresholds Linked to Retention ─────────────────
    st.subheader("Engagement Thresholds Linked to Retention")
    st.markdown("RSI threshold analysis — identifying the score band where churn risk sharply changes.")

    rsi_bins = pd.cut(df['RSI'], bins=range(0, 105, 10), right=False)
    threshold_df = df.groupby(rsi_bins, observed=False).agg(
        Customers=('CustomerId', 'count'),
        Churn_Rate=('Exited', 'mean')
    ).reset_index()
    threshold_df['RSI_Band'] = threshold_df['RSI'].astype(str)
    threshold_df['Churn_Rate'] = (threshold_df['Churn_Rate'] * 100).round(2)

    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Bar(
        x=threshold_df['RSI_Band'], y=threshold_df['Customers'],
        name='Customer Volume', marker_color='rgba(123,47,247,0.5)',
        yaxis='y2'
    ))
    fig_thresh.add_trace(go.Scatter(
        x=threshold_df['RSI_Band'], y=threshold_df['Churn_Rate'],
        name='Churn Rate (%)', mode='lines+markers',
        line=dict(color='#00d2ff', width=3),
        marker=dict(size=9, color='#ff0080'),
        yaxis='y'
    ))
    fig_thresh.update_layout(
        title=dict(text='RSI Band — Churn Rate & Volume Overlay', font=dict(color='#ffffff', size=16), x=0.01),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0'),
        xaxis=dict(title='RSI Score Band', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Churn Rate (%)', gridcolor='rgba(255,255,255,0.05)', side='left'),
        yaxis2=dict(title='Customer Volume', overlaying='y', side='right', showgrid=False),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=50, b=10),
        template='plotly_dark'
    )
    st.plotly_chart(fig_thresh, width='stretch')

    st.markdown("---")
    st.subheader("📋 RSI Scoring Breakdown — Full Customer List")

    display_cols = ['CustomerId', 'Surname', 'Geography', 'Age', 'IsActiveMember',
                    'NumOfProducts', 'HasCrCard', 'Balance', 'RSI', 'RSI_Tier', 'Churn_Probability']
    tier_order = {'🔴 Critical': 0, '🟡 At Risk': 1, '🔵 Loyal': 2, '🟢 Champion': 3}
    sorted_df = df[display_cols].copy()
    sorted_df['_sort'] = sorted_df['RSI_Tier'].map(tier_order)
    sorted_df = sorted_df.sort_values('_sort').drop(columns='_sort')
    st.dataframe(sorted_df, width='stretch')

    csv_rsi = sorted_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export RSI Scoring Report",
        data=csv_rsi,
        file_name="rsi_retention_scoring.csv",
        mime="text/csv"
    )
