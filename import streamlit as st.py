import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="Bank Customer Retention Analytics")
st.title("📊 Customer Engagement & Product Utilization Analytics")
st.markdown("### Strategic Retention Dashboard & Risk Detector")

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
                         labels={'Churn_Rate': 'Churn Rate (%)'}, color='Engagement_Profile')
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
                           labels={'Churn_Rate': 'Observed Churn Rate (%)', 'NumOfProducts': 'Number of Active Bank Products'})
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
                     color='Importance', color_continuous_scale='Reds')
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