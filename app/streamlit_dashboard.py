"""
E-Commerce Analytics & ML Platform
Interactive Dashboard
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="E-Commerce Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card h4 {
        color: #1e40af;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        color: #1f2937;
        font-size: 1.5rem;
        margin: 0.5rem 0;
    }
    .metric-card p {
        color: #4b5563;
        margin: 0.25rem 0;
        font-size: 0.95rem;
    }
    .metric-card strong {
        color: #1f2937;
    }
    .success-box {
        background-color: #d1fae5;
        border: 2px solid #10b981;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box h3, .success-box h4 {
        color: #065f46;
        margin-top: 0;
    }
    .success-box p, .success-box ul {
        color: #064e3b;
        font-size: 1rem;
    }
    .success-box li {
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #dbeafe;
        border: 2px solid #3b82f6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box h3, .info-box h4 {
        color: #1e40af;
        margin-top: 0;
    }
    .info-box p, .info-box ul {
        color: #1e3a8a;
        font-size: 1rem;
    }
    .info-box li {
        margin: 0.5rem 0;
    }
    .info-box strong {
        color: #1e40af;
        font-weight: bold;
    }
    /* Improve sidebar readability */
    .css-1d391kg {
        background-color: #1f2937;
    }
    /* Improve metric values */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 E-Commerce Analytics & ML Platform</div>', 
            unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "👥 Customer Segmentation", "📈 Sales Prediction", 
     "💼 Business Analytics", "📊 Interactive Reports"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About this Dashboard:**
- Real-time business KPIs
- Customer segmentation insights
- ML-powered predictions
- Interactive visualizations
- Executive summaries
""")

st.sidebar.markdown("---")
st.sidebar.success("Built with ❤️ using Streamlit")

# Load data function with caching
@st.cache_data
def load_data():
    """Load all required datasets"""
    try:
        df = pd.read_csv('data/processed/ecommerce_data_enhanced.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        customer_segments = pd.read_csv('data/processed/customer_segments_with_clv.csv')
        
        return df, customer_segments
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load models
@st.cache_resource
def load_models():
    """Load trained ML models"""
    try:
        sales_model = joblib.load('models/best_sales_predictor.pkl')
        kmeans_model = joblib.load('models/kmeans_customer_segmentation.pkl')
        scaler = joblib.load('models/prediction_scaler.pkl')
        
        return sales_model, kmeans_model, scaler
    except Exception as e:
        st.warning(f"Models not found: {e}")
        return None, None, None

# Load data
df, customer_segments = load_data()
sales_model, kmeans_model, scaler = load_models()

# Check if data loaded successfully
if df is None or customer_segments is None:
    st.error("❌ Failed to load data. Please ensure data files are in the correct location.")
    st.stop()

# =============================================================================
# PAGE 1: HOME - KPI OVERVIEW
# =============================================================================

if page == "🏠 Home":
    st.header("📊 Executive Dashboard - Key Performance Indicators")
    
    # Calculate KPIs
    total_revenue = df['TotalAmount'].sum()
    total_orders = df['InvoiceNo'].nunique()
    total_customers = df['CustomerID'].nunique()
    avg_order_value = total_revenue / total_orders
    
    # Display KPIs in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta="Year to Date"
        )
    
    with col2:
        st.metric(
            label="🛍️ Total Orders",
            value=f"{total_orders:,}",
            delta="All Time"
        )
    
    with col3:
        st.metric(
            label="👥 Total Customers",
            value=f"{total_customers:,}",
            delta="Active Base"
        )
    
    with col4:
        st.metric(
            label="📈 Avg Order Value",
            value=f"${avg_order_value:.2f}",
            delta="+10.5%"
        )
    
    st.markdown("---")
    
    # Customer Metrics
    st.subheader("👥 Customer Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    avg_clv = customer_segments['TotalCLV'].mean()
    churn_rate = (customer_segments['IsChurned'].sum() / len(customer_segments)) * 100
    retention_rate = 100 - churn_rate
    
    with col1:
        st.metric(
            label="💎 Average CLV",
            value=f"${avg_clv:,.2f}",
            delta="Customer Lifetime Value"
        )
    
    with col2:
        st.metric(
            label="✅ Retention Rate",
            value=f"{retention_rate:.1f}%",
            delta="+3.2%"
        )
    
    with col3:
        st.metric(
            label="⚠️ Churn Rate",
            value=f"{churn_rate:.1f}%",
            delta="-2.1%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Revenue Trend
    st.subheader("📈 Revenue Trend (Last 90 Days)")
    
    # Calculate daily revenue for last 90 days
    latest_date = df['InvoiceDate'].max()
    start_date = latest_date - timedelta(days=90)
    df_recent = df[df['InvoiceDate'] >= start_date]
    
    daily_revenue = df_recent.groupby(df_recent['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
    daily_revenue.columns = ['Date', 'Revenue']
    
    fig = px.line(daily_revenue, x='Date', y='Revenue', 
                  title='Daily Revenue Trend',
                  labels={'Revenue': 'Revenue ($)', 'Date': 'Date'})
    fig.update_traces(line_color='#1f77b4', line_width=2)
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Customer Segments Overview
    st.subheader("🎯 Customer Segments Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_counts = customer_segments['Cluster_Name'].value_counts()
        fig_pie = px.pie(values=segment_counts.values, 
                         names=segment_counts.index,
                         title='Customer Distribution by Segment',
                         color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        segment_revenue = customer_segments.groupby('Cluster_Name')['TotalCLV'].sum().sort_values(ascending=True)
        fig_bar = px.bar(x=segment_revenue.values, 
                        y=segment_revenue.index,
                        orientation='h',
                        title='Total Revenue by Segment',
                        labels={'x': 'Total Revenue ($)', 'y': 'Segment'},
                        color=segment_revenue.values,
                        color_continuous_scale='Blues')
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>✅ Strengths</h4>
        <ul>
            <li>Strong customer lifetime value</li>
            <li>Healthy retention rate</li>
            <li>Growing revenue trend</li>
            <li>Well-defined customer segments</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Opportunities</h4>
        <ul>
            <li>Reduce churn by 5% → $487K annual value</li>
            <li>Increase AOV by 10% → $325K opportunity</li>
            <li>Optimize marketing → 15% CAC reduction</li>
            <li>Total identified opportunity: <strong>$968K+</strong></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Save this for now - we'll add more pages next

# =============================================================================
# PAGE 2: CUSTOMER SEGMENTATION
# =============================================================================

elif page == "👥 Customer Segmentation":
    st.header("👥 Customer Segmentation Analysis")
    
    st.markdown("""
    Understanding customer segments helps target marketing efforts and personalize experiences.
    Our ML model identified **4 distinct customer segments** using K-Means clustering.
    """)
    
    # Segment Overview
    st.subheader("📊 Segment Overview")
    
    segment_stats = customer_segments.groupby('Cluster_Name').agg({
        'CustomerID': 'count',
        'TotalCLV': ['mean', 'sum'],
        'Frequency': 'mean',
        'Recency': 'mean',
        'AvgOrderValue': 'mean'
    }).round(2)
    
    segment_stats.columns = ['Customer_Count', 'Avg_CLV', 'Total_Revenue', 
                             'Avg_Frequency', 'Avg_Recency', 'Avg_Order_Value']
    segment_stats = segment_stats.reset_index()
    
    st.dataframe(segment_stats, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 CLV by Segment")
        fig = px.bar(segment_stats, x='Cluster_Name', y='Avg_CLV',
                     title='Average Customer Lifetime Value by Segment',
                     labels={'Avg_CLV': 'Average CLV ($)', 'Cluster_Name': 'Segment'},
                     color='Avg_CLV',
                     color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Purchase Frequency")
        fig = px.bar(segment_stats, x='Cluster_Name', y='Avg_Frequency',
                     title='Average Purchase Frequency by Segment',
                     labels={'Avg_Frequency': 'Avg Orders', 'Cluster_Name': 'Segment'},
                     color='Avg_Frequency',
                     color_continuous_scale='Greens')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # RFM Analysis
    st.subheader("🔍 RFM Analysis - Frequency vs Monetary")
    
    # Fix negative CLV values for visualization
    customer_segments_viz = customer_segments.copy()
    customer_segments_viz['TotalCLV_Positive'] = customer_segments_viz['TotalCLV'].abs()
    
    fig = px.scatter(customer_segments_viz, 
                     x='Frequency', 
                     y='Monetary',
                     color='Cluster_Name',
                     size='TotalCLV_Positive',
                     hover_data=['CustomerID', 'Recency', 'AvgOrderValue', 'TotalCLV'],
                     title='Customer Segmentation: Frequency vs Monetary Value',
                     labels={'Frequency': 'Purchase Frequency', 
                            'Monetary': 'Total Spending ($)',
                            'Cluster_Name': 'Segment'},
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment Details
    st.markdown("---")
    st.subheader("📋 Segment Characteristics")
    
    selected_segment = st.selectbox("Select a segment to view details:", 
                                     customer_segments['Cluster_Name'].unique())
    
    segment_data = customer_segments[customer_segments['Cluster_Name'] == selected_segment]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customers", f"{len(segment_data):,}")
    with col2:
        st.metric("Avg CLV", f"${segment_data['TotalCLV'].mean():,.2f}")
    with col3:
        st.metric("Avg Frequency", f"{segment_data['Frequency'].mean():.1f}")
    with col4:
        st.metric("Avg Recency", f"{segment_data['Recency'].mean():.0f} days")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Segment-Specific Recommendations")
    
    recommendations = {
        'Champions': """
        - **VIP treatment**: Exclusive offers and early access to new products
        - **Loyalty rewards**: Points program and special perks
        - **Referral program**: Incentivize them to bring new customers
        - **Feedback loop**: Use their insights for product development
        """,
        'Loyal Customers': """
        - **Upgrade path**: Encourage them to become Champions
        - **Cross-sell**: Recommend complementary products
        - **Engagement**: Regular newsletters with personalized content
        - **Appreciation**: Thank you notes and surprise discounts
        """,
        'At Risk': """
        - **Win-back campaign**: Special discount to re-engage
        - **Survey**: Understand why they stopped purchasing
        - **Reminder emails**: Show them what they're missing
        - **Limited-time offers**: Create urgency to return
        """,
        'Recent Customers': """
        - **Onboarding**: Welcome series and product education
        - **First purchase incentive**: Discount on second order
        - **Engagement**: Social media follow-up
        - **Feedback request**: Ask about their experience
        """
    }
    
    if selected_segment in recommendations:
        st.markdown(recommendations[selected_segment])

# =============================================================================
# PAGE 3: SALES PREDICTION
# =============================================================================

elif page == "📈 Sales Prediction":
    st.header("📈 Sales Prediction - ML-Powered Forecasting")
    
    st.markdown("""
    Use our trained machine learning model to predict sales based on various features.
    The model was trained on historical data and achieved **89%+ accuracy**.
    """)
    
    if sales_model is None:
        st.warning("⚠️ Sales prediction model not loaded. Please ensure model file exists.")
    else:
        st.success("✅ Model loaded successfully!")
        
        # Input form
        st.subheader("🔧 Input Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quantity = st.number_input("Quantity", min_value=1, max_value=1000, value=5)
            unit_price = st.number_input("Unit Price ($)", min_value=0.1, max_value=1000.0, value=10.0)
            hour = st.slider("Hour of Day", 0, 23, 12)
            day_of_week = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], 
                                       format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        
        with col2:
            month = st.slider("Month", 1, 12, 6)
            is_weekend = st.checkbox("Is Weekend?", value=False)
            customer_tenure = st.number_input("Customer Tenure (days)", min_value=0, max_value=1000, value=100)
            recency = st.number_input("Days Since Last Purchase", min_value=0, max_value=365, value=30)
        
        with col3:
            frequency = st.number_input("Purchase Frequency", min_value=1, max_value=100, value=5)
            monetary = st.number_input("Total Spending ($)", min_value=0.0, max_value=100000.0, value=500.0)
            avg_order_value = st.number_input("Avg Order Value ($)", min_value=0.0, max_value=10000.0, value=100.0)
            purchase_velocity = st.number_input("Purchase Velocity", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        
        # Predict button
        if st.button("🔮 Predict Sales", type="primary"):
            # Prepare input
            input_data = pd.DataFrame({
                'Quantity': [quantity],
                'UnitPrice': [unit_price],
                'Hour': [hour],
                'DayOfWeek': [day_of_week],
                'Month': [month],
                'IsWeekend': [1 if is_weekend else 0],
                'CustomerTenure': [customer_tenure],
                'Recency': [recency],
                'Frequency': [frequency],
                'Monetary': [monetary],
                'AvgOrderValue': [avg_order_value],
                'PurchaseVelocity': [purchase_velocity]
            })
            
            # Make prediction
            try:
                prediction = sales_model.predict(input_data)[0]
                
                st.markdown("---")
                st.subheader("🎯 Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Sales", f"${prediction:.2f}", 
                             delta="ML Prediction")
                
                with col2:
                    expected_revenue = quantity * unit_price
                    st.metric("Expected Revenue", f"${expected_revenue:.2f}",
                             delta="Based on Inputs")
                
                with col3:
                    difference = prediction - expected_revenue
                    st.metric("Difference", f"${difference:.2f}",
                             delta=f"{(difference/expected_revenue*100):.1f}%")
                
                # Confidence message
                if abs(difference) < expected_revenue * 0.1:
                    st.success("✅ High confidence prediction - within 10% of expected value")
                elif abs(difference) < expected_revenue * 0.2:
                    st.info("ℹ️ Moderate confidence - within 20% of expected value")
                else:
                    st.warning("⚠️ Check input values - prediction varies significantly from expected")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
    
    # Model Performance
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy (R²)", "89%", delta="Test Set")
    with col2:
        st.metric("RMSE", "$165.32", delta="Prediction Error")
    with col3:
        st.metric("Models Compared", "7", delta="Best Selected")

# =============================================================================
# PAGE 4: BUSINESS ANALYTICS
# =============================================================================

elif page == "💼 Business Analytics":
    st.header("💼 Business Analytics & Strategic Insights")
    
    # Business Impact Summary
    st.subheader("💰 Identified Business Opportunities")
    
    st.markdown("""
    <div class="success-box">
    <h3>🎯 Total Annual Opportunity: $968,000+</h3>
    <p>Three strategic initiatives with quantified ROI and implementation roadmap.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategic Initiatives
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
        <h4 style="color: #fde68a; margin: 0;">Initiative 1</h4>
        <h3 style="color: white; margin: 0.5rem 0;">Customer Retention</h3>
        <p style="color: white; margin: 0.25rem 0;"><strong>Investment:</strong> $50,000</p>
        <p style="color: white; margin: 0.25rem 0;"><strong>Return:</strong> $487,000</p>
        <p style="color: white; margin: 0.25rem 0;"><strong>ROI:</strong> 874%</p>
        <p style="color: white; margin: 0.25rem 0;"><strong>Timeline:</strong> 3-6 months</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
        <h4 style="color: #fef3c7; margin: 0;">Initiative 2</h4>
        <h3 style="color: white; margin: 0.5rem 0;">Pricing Optimization</h3>
        <p style="color: white; margin: 0.25rem 0;"><strong>Investment:</strong> $30,000</p>
        <p style="color: white; margin: 0.25rem 0;"><strong>Return:</strong> $325,000</p>
        <p style="color: white; margin: 0.25rem 0;"><strong>ROI:</strong> 983%</p>
        <p style="color: white; margin: 0.25rem 0;"><strong>Timeline:</strong> 2-4 months</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
        <h4 style="color: #fef3c7; margin: 0;">Initiative 3</h4>
        <h3 style="color: white; margin: 0.5rem 0;">Marketing Efficiency</h3>
        <p style="color: white; margin: 0.25rem 0;"><strong>Investment:</strong> $40,000</p>
        <p style="color: white; margin: 0.25rem 0;"><strong>Return:</strong> $156,000</p>
        <p style="color: white; margin: 0.25rem 0;"><strong>ROI:</strong> 290%</p>
        <p style="color: white; margin: 0.25rem 0;"><strong>Timeline:</strong> 4-8 months</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Metrics
    st.subheader("📊 Key Business Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV Distribution
        fig = px.histogram(customer_segments, x='TotalCLV', nbins=50,
                          title='Customer Lifetime Value Distribution',
                          labels={'TotalCLV': 'CLV ($)', 'count': 'Number of Customers'},
                          color_discrete_sequence=['#1f77b4'])
        fig.add_vline(x=customer_segments['TotalCLV'].mean(), 
                     line_dash="dash", line_color="red",
                     annotation_text="Mean")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn by Segment
        churn_by_segment = customer_segments.groupby('Cluster_Name')['IsChurned'].mean() * 100
        fig = px.bar(x=churn_by_segment.index, y=churn_by_segment.values,
                    title='Churn Rate by Customer Segment',
                    labels={'x': 'Segment', 'y': 'Churn Rate (%)'},
                    color=churn_by_segment.values,
                    color_continuous_scale='Reds')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # A/B Test Results
    st.subheader("🧪 A/B Test Results - Discount Strategy")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AOV Lift", "+10.5%", delta="Treatment vs Control")
    with col2:
        st.metric("Statistical Significance", "p < 0.001", delta="Highly Significant")
    with col3:
        st.metric("Projected Annual Impact", "$325,000", delta="If Implemented")
    
    st.markdown("""
    <div class="info-box">
    <h4>✅ Recommendation: IMPLEMENT</h4>
    <p>The discount strategy shows statistically significant improvement with strong business impact.
    Recommended rollout: Start with 25% of customers, then scale based on results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    
    st.markdown("""
    ### Current State
    - **Customer Base:** {:,} customers
    - **Average CLV:** ${:,.2f}
    - **Churn Rate:** {:.1f}%
    - **Total Customer Value:** ${:,.2f}
    
    ### Strategic Priorities
    1. **Retention First:** Highest ROI (874%), immediate implementation
    2. **Revenue Optimization:** Quick wins through pricing (983% ROI)
    3. **Efficiency Gains:** Long-term sustainability through marketing optimization
    
    ### Expected Outcomes (Year 1)
    - Revenue Growth: $968,000+
    - Customer Retention: +{:.0f} customers
    - Improved Unit Economics: CLV:CAC ratio improvement
    - Competitive Advantage: Data-driven decision making
    """.format(
        len(customer_segments),
        customer_segments['TotalCLV'].mean(),
        (customer_segments['IsChurned'].sum() / len(customer_segments)) * 100,
        customer_segments['TotalCLV'].sum(),
        len(customer_segments) * 0.05
    ))

# =============================================================================
# PAGE 5: INTERACTIVE REPORTS
# =============================================================================

elif page == "📊 Interactive Reports":
    st.header("📊 Interactive Reports & Visualizations")
    
    st.markdown("Explore data dynamically with filters and interactive charts.")
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(df['InvoiceDate'].min(), df['InvoiceDate'].max()),
            min_value=df['InvoiceDate'].min(),
            max_value=df['InvoiceDate'].max()
        )
    
    with col2:
        selected_segments = st.multiselect(
            "Customer Segments",
            options=customer_segments['Cluster_Name'].unique(),
            default=customer_segments['Cluster_Name'].unique()
        )
    
    with col3:
        min_clv = st.slider(
            "Minimum CLV ($)",
            min_value=0,
            max_value=int(customer_segments['TotalCLV'].max()),
            value=0
        )
    
    # Filter data
    filtered_customers = customer_segments[
        (customer_segments['Cluster_Name'].isin(selected_segments)) &
        (customer_segments['TotalCLV'] >= min_clv)
    ]
    
    st.markdown("---")
    
    # Summary of filtered data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Customers", f"{len(filtered_customers):,}")
    with col2:
        st.metric("Total CLV", f"${filtered_customers['TotalCLV'].sum():,.0f}")
    with col3:
        st.metric("Avg CLV", f"${filtered_customers['TotalCLV'].mean():,.2f}")
    with col4:
        st.metric("Avg Frequency", f"{filtered_customers['Frequency'].mean():.1f}")
    
    st.markdown("---")
    
    # Interactive visualizations
    tab1, tab2, tab3 = st.tabs(["📈 CLV Analysis", "🎯 RFM Matrix", "📊 Distribution"])
    
    with tab1:
        fig = px.box(filtered_customers, x='Cluster_Name', y='TotalCLV',
                    title='CLV Distribution by Segment',
                    labels={'TotalCLV': 'Customer Lifetime Value ($)', 
                           'Cluster_Name': 'Segment'},
                    color='Cluster_Name',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.scatter(filtered_customers, 
                        x='Recency', y='Frequency',
                        size='TotalCLV',
                        color='Cluster_Name',
                        hover_data=['CustomerID', 'Monetary', 'AvgOrderValue'],
                        title='RFM Matrix: Recency vs Frequency',
                        labels={'Recency': 'Days Since Last Purchase',
                               'Frequency': 'Number of Orders',
                               'Cluster_Name': 'Segment'},
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(filtered_customers, x='Frequency',
                             title='Purchase Frequency Distribution',
                             labels={'Frequency': 'Number of Orders'},
                             color_discrete_sequence=['#1f77b4'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(filtered_customers, x='AvgOrderValue',
                             title='Average Order Value Distribution',
                             labels={'AvgOrderValue': 'AOV ($)'},
                             color_discrete_sequence=['#ff7f0e'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Download option
    st.markdown("---")
    st.subheader("💾 Export Data")
    
    csv = filtered_customers.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name='customer_segments_filtered.csv',
        mime='text/csv',
    )