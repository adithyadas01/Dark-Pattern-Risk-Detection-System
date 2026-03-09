import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.inspection import permutation_importance

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Dark Pattern Risk Detection",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    h1 { color: #ff4444; font-size: 2.5em; }
    h2 { color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

st.title("⚠️ Dark Pattern Risk Detection System")
st.write("AI-powered detection of manipulative UX behavior patterns")

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def create_risk_gauge(probability):
    """Create an interactive risk gauge"""
    risk_percentage = probability * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Dark Pattern Risk Score"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "#90EE90"},
                {'range': [25, 50], 'color': "#FFD700"},
                {'range': [50, 75], 'color': "#FF8C00"},
                {'range': [75, 100], 'color': "#FF4444"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=400)
    return fig

def get_risk_label(probability):
    """Get risk category label"""
    if probability < 0.25:
        return "🟢 Very Low Risk", "Low"
    elif probability < 0.5:
        return "🟡 Low Risk", "Low"
    elif probability < 0.75:
        return "🟠 Medium Risk", "Medium"
    else:
        return "🔴 High Risk", "High"

def create_feature_importance_chart(model, X_sample):
    """Create feature importance visualization"""
    try:
        importances = model.feature_importances_
        feature_names = FEATURE_COLUMNS
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(10)
        
        fig = px.barh(importance_df, x='Importance', y='Feature',
                      title='Top 10 Most Important Features',
                      color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(height=400, showlegend=False)
        return fig
    except:
        return None

# --------------------------------------------------
# LOAD SAVED ARTIFACTS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("dark_pattern_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_month = joblib.load("month_encoder.pkl")
    le_visitor = joblib.load("visitor_encoder.pkl")
    return model, scaler, le_month, le_visitor

model, scaler, le_month, le_visitor = load_artifacts()

# --------------------------------------------------
# FEATURE SCHEMA (MUST MATCH TRAINING)
# --------------------------------------------------
FEATURE_COLUMNS = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues',
    'SpecialDay',
    'Month',
    'VisitorType',
    'Weekend',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType'
]

# --------------------------------------------------
# SIDEBAR MODE SELECTION
# --------------------------------------------------
st.sidebar.title("Input Mode")
mode = st.sidebar.radio(
    "Choose Prediction Mode",
    ["Single User Prediction", "CSV Upload Prediction"]
)

# ==================================================
# SINGLE USER PREDICTION
# ==================================================
if mode == "Single User Prediction":

    st.subheader("🧑 Single User Behavior Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        Administrative = st.number_input("Administrative Pages", 0, 50, 1)
        Informational = st.number_input("Informational Pages", 0, 50, 1)
        ProductRelated = st.number_input("Product Related Pages", 0, 300, 10)
        BounceRates = st.slider("Bounce Rate", 0.0, 1.0, 0.2)
        ExitRates = st.slider("Exit Rate", 0.0, 1.0, 0.3)
        PageValues = st.number_input("Page Values", 0.0, 500.0, 0.0)

    with col2:
        Administrative_Duration = st.number_input("Admin Duration", 0.0, 10000.0, 100.0)
        Informational_Duration = st.number_input("Info Duration", 0.0, 10000.0, 50.0)
        ProductRelated_Duration = st.number_input("Product Duration", 0.0, 20000.0, 500.0)
        SpecialDay = st.slider("Special Day", 0.0, 1.0, 0.0)
        Weekend = st.selectbox("Weekend", [0, 1])

    with col3:
        Month = st.selectbox("Month", le_month.classes_)
        VisitorType = st.selectbox("Visitor Type", le_visitor.classes_)
        OperatingSystems = st.number_input("Operating Systems", 1, 8, 2)
        Browser = st.number_input("Browser", 1, 13, 2)
        Region = st.number_input("Region", 1, 9, 1)
        TrafficType = st.number_input("Traffic Type", 1, 20, 2)

    if st.button("🔮 Predict Dark Pattern Risk"):

        # ENCODE CATEGORICALS
        month_enc = le_month.transform([Month])[0]
        visitor_enc = le_visitor.transform([VisitorType])[0]

        # CREATE DATAFRAME (ALL FEATURES)
        input_df = pd.DataFrame([{
            'Administrative': Administrative,
            'Administrative_Duration': Administrative_Duration,
            'Informational': Informational,
            'Informational_Duration': Informational_Duration,
            'ProductRelated': ProductRelated,
            'ProductRelated_Duration': ProductRelated_Duration,
            'BounceRates': BounceRates,
            'ExitRates': ExitRates,
            'PageValues': PageValues,
            'SpecialDay': SpecialDay,
            'Month': month_enc,
            'VisitorType': visitor_enc,
            'Weekend': Weekend,
            'OperatingSystems': OperatingSystems,
            'Browser': Browser,
            'Region': Region,
            'TrafficType': TrafficType
        }])

        # FORCE COLUMN ORDER
        input_df = input_df[FEATURE_COLUMNS]

        # SCALE (convert to numpy to avoid feature name validation)
        X_scaled = scaler.transform(input_df.values)

        # PREDICT
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        # DISPLAY RESULTS
        risk_label, risk_cat = get_risk_label(probability)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(create_risk_gauge(probability), use_container_width=True)
        
        with col2:
            st.metric("Risk Category", risk_label, f"{probability:.1%}")
            st.metric("Confidence Score", f"{max(probability, 1-probability):.1%}")
        
        # INSIGHTS
        st.divider()
        st.subheader("📊 Prediction Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        with insight_col1:
            st.metric("Bounce Rate", f"{BounceRates:.1%}")
        with insight_col2:
            st.metric("Exit Rate", f"{ExitRates:.1%}")
        with insight_col3:
            st.metric("Page Values", f"${PageValues:.2f}")
        
        # RECOMMENDATIONS
        st.divider()
        st.subheader("💡 Risk Mitigation Recommendations")
        recommendations = []
        
        if BounceRates > 0.5:
            recommendations.append("🔴 **High Bounce Rate** - Users leaving quickly. Improve landing page clarity and UX.")
        if ExitRates > 0.5:
            recommendations.append("🔴 **High Exit Rate** - Many users abandoning pages. Review exit points and friction.")
        if PageValues < 1:
            recommendations.append("🟡 **Low Page Values** - Limited engagement value. Add more relevant content/features.")
        if ProductRelated_Duration < 100:
            recommendations.append("🟡 **Low Product Engagement** - Users spend little time on products. Improve product presentation.")
        
        if not recommendations:
            st.success("✅ No major risk factors detected. User behavior appears legitimate.")
        else:
            for rec in recommendations:
                st.warning(rec)

# ==================================================
# CSV UPLOAD PREDICTION
# ==================================================
else:
    st.subheader("📂 Upload Dataset for Bulk Prediction")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df = pd.read_csv(file)

        # ENCODE CATEGORICALS
        df['Month'] = le_month.transform(df['Month'])
        df['VisitorType'] = le_visitor.transform(df['VisitorType'])

        # SELECT AND REINDEX FEATURES (ENSURE ORDER)
        X = df.loc[:, FEATURE_COLUMNS]
        X = X.reindex(columns=FEATURE_COLUMNS)

        # SCALE (convert to numpy to avoid feature name validation)
        X_scaled = scaler.transform(X.values)

        # PREDICT
        df['Predicted_Risk'] = model.predict(X_scaled)
        df['Risk_Probability'] = model.predict_proba(X_scaled)[:, 1]
        df['Risk_Category'] = df['Risk_Probability'].apply(lambda x: get_risk_label(x)[1])

        st.success("✅ Predictions Completed")
        
        # ANALYTICS DASHBOARD
        st.divider()
        st.subheader("📊 Analytics Dashboard")
        
        # Summary metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            high_risk_count = (df['Predicted_Risk'] == 1).sum()
            st.metric("High Risk Users", high_risk_count)
        with metric_col2:
            avg_risk = df['Risk_Probability'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.1%}")
        with metric_col3:
            st.metric("Total Users", len(df))
        with metric_col4:
            safe_pct = (df['Predicted_Risk'] == 0).sum() / len(df) * 100
            st.metric("Safe Users", f"{safe_pct:.1f}%")
        
        st.divider()
        
        # Visualizations
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Risk distribution
            risk_dist = df['Risk_Category'].value_counts()
            colors_map = {'Very Low': '#90EE90', 'Low': '#FFD700', 'Medium': '#FF8C00', 'High': '#FF4444'}
            colors = [colors_map.get(cat, '#999') for cat in risk_dist.index]
            
            fig_pie = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title='Risk Distribution',
                color_discrete_map={cat: colors_map.get(cat, '#999') for cat in risk_dist.index}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_col2:
            # Risk probability histogram
            fig_hist = px.histogram(
                df,
                x='Risk_Probability',
                nbins=30,
                title='Risk Score Distribution',
                labels={'Risk_Probability': 'Risk Probability'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Feature analysis
        st.divider()
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            # High risk vs Low risk comparison
            high_risk_users = df[df['Predicted_Risk'] == 1]
            low_risk_users = df[df['Predicted_Risk'] == 0]
            
            comparison_data = pd.DataFrame({
                'Metric': ['Avg Bounce Rate', 'Avg Exit Rate', 'Avg Page Values'],
                'High Risk': [
                    high_risk_users['BounceRates'].mean() if len(high_risk_users) > 0 else 0,
                    high_risk_users['ExitRates'].mean() if len(high_risk_users) > 0 else 0,
                    high_risk_users['PageValues'].mean() if len(high_risk_users) > 0 else 0,
                ],
                'Low Risk': [
                    low_risk_users['BounceRates'].mean(),
                    low_risk_users['ExitRates'].mean(),
                    low_risk_users['PageValues'].mean(),
                ]
            })
            
            fig_comp = px.bar(
                comparison_data,
                x='Metric',
                y=['High Risk', 'Low Risk'],
                title='High Risk vs Low Risk Behavior',
                barmode='group',
                color_discrete_map={'High Risk': '#FF4444', 'Low Risk': '#90EE90'}
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with chart_col4:
            # Risk by visitor type
            visitor_risk = df.groupby('VisitorType')['Risk_Probability'].mean().sort_values(ascending=False)
            visitor_type_names = le_visitor.inverse_transform(visitor_risk.index)
            
            fig_visitor = px.bar(
                x=visitor_type_names,
                y=visitor_risk.values,
                title='Average Risk by Visitor Type',
                labels={'x': 'Visitor Type', 'y': 'Avg Risk Score'},
                color=visitor_risk.values,
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_visitor, use_container_width=True)
        
        st.divider()
        st.subheader("👥 User Details")
        
        # Data table with filtering
        col_display = st.multiselect(
            "Select columns to display",
            df.columns.tolist(),
            default=['BounceRates', 'ExitRates', 'PageValues', 'Risk_Probability', 'Risk_Category']
        )
        
        if col_display:
            st.dataframe(
                df[col_display].sort_values('Risk_Probability', ascending=False),
                use_container_width=True,
                height=400
            )
        
        st.divider()
        st.subheader("⬇️ Export Results")
        
        # Download options
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV Predictions",
            csv,
            "dark_pattern_predictions.csv",
            "text/csv"
        )
        
        # Export high-risk users only
        high_risk_df = df[df['Predicted_Risk'] == 1]
        if len(high_risk_df) > 0:
            csv_high_risk = high_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "🚨 Download High-Risk Users Only",
                csv_high_risk,
                "high_risk_users.csv",
                "text/csv"
            )
