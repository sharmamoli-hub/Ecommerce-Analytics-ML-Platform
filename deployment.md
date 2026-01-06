# 🚀 Deployment Guide - E-Commerce Analytics Platform

## Dashboard Overview

Interactive Streamlit dashboard featuring:
- 📊 Real-time business KPIs
- 👥 Customer segmentation insights
- 📈 ML-powered sales predictions
- 💼 Business analytics with $968K+ opportunity identified
- 🎯 Interactive reports with dynamic filters

---

## 🖥️ Local Deployment

### Prerequisites
- Python 3.13+
- All dependencies installed (see requirements.txt)

### Steps

1. **Activate Virtual Environment**
```bash
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
```

2. **Navigate to Project Directory**
```bash
   cd D:\Ecommerce-Analytics-ML-Platform
```

3. **Run the Dashboard**
```bash
   streamlit run app/streamlit_dashboard.py
```

4. **Access Dashboard**
   - Browser will open automatically at `http://localhost:8501`
   - If not, manually navigate to that URL

5. **Stop the Dashboard**
   - Press `Ctrl + C` in the terminal

---

## 📁 Required Files

Ensure these files exist before running:
```
Ecommerce-Analytics-ML-Platform/
├── app/
│   └── streamlit_dashboard.py
├── data/
│   └── processed/
│       ├── ecommerce_data_enhanced.csv
│       └── customer_segments_with_clv.csv
├── models/
│   ├── best_sales_predictor.pkl
│   ├── kmeans_customer_segmentation.pkl
│   └── prediction_scaler.pkl
└── images/
    └── results/
        └── (all visualization images)
```

---

## 🌐 Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

1. **Push code to GitHub** (already done ✅)

2. **Go to:** https://streamlit.io/cloud

3. **Sign in** with GitHub

4. **Deploy:**
   - Click "New app"
   - Select your repository: `Ecommerce-Analytics-ML-Platform`
   - Main file: `app/streamlit_dashboard.py`
   - Click "Deploy"

5. **Live in 2-3 minutes!**

**Note:** May need to adjust file paths for cloud deployment:
- Change `data/processed/...` to relative paths
- Ensure all data files are in GitHub (not .gitignored)

---

### Option 2: Heroku

1. **Create `Procfile`:**
```
   web: streamlit run app/streamlit_dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Deploy:**
```bash
   heroku create your-app-name
   git push heroku main
```

---

### Option 3: AWS EC2 / Azure / GCP

1. **Launch instance**
2. **Install Python & dependencies**
3. **Clone repository**
4. **Run with nohup:**
```bash
   nohup streamlit run app/streamlit_dashboard.py --server.port=8501 &
```

---

## ⚙️ Configuration

### Change Port
```bash
streamlit run app/streamlit_dashboard.py --server.port 8502
```

### Change Theme
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#1f2937"
font = "sans serif"
```

---

## 🔧 Troubleshooting

### Issue: Data files not found
**Solution:** Ensure CSV files are in correct location:
```
data/processed/ecommerce_data_enhanced.csv
data/processed/customer_segments_with_clv.csv
```

### Issue: Models not loading
**Solution:** Verify model files exist:
```
models/best_sales_predictor.pkl
models/kmeans_customer_segmentation.pkl
models/prediction_scaler.pkl
```

### Issue: Port already in use
**Solution:** Use different port:
```bash
streamlit run app/streamlit_dashboard.py --server.port 8502
```

### Issue: Memory error
**Solution:** Reduce data size or upgrade hosting

---

## 📊 Dashboard Features

### Page 1: Home 🏠
- Executive KPI overview
- Revenue trends
- Customer segment distribution
- Key insights and opportunities

### Page 2: Customer Segmentation 👥
- Segment statistics and profiles
- CLV analysis by segment
- RFM scatter plots
- Segment-specific recommendations

### Page 3: Sales Prediction 📈
- Interactive prediction interface
- ML model with 89% accuracy
- Real-time predictions
- Model performance metrics

### Page 4: Business Analytics 💼
- $968K+ opportunity breakdown
- Strategic initiatives with ROI
- A/B test results
- Executive summary

### Page 5: Interactive Reports 📊
- Dynamic filters
- Multiple visualization tabs
- Export functionality
- Custom analysis

---

## 🎯 Best Practices

1. **Regular Updates:** Keep data fresh for accurate insights
2. **Model Retraining:** Retrain ML models quarterly
3. **Performance Monitoring:** Track dashboard load times
4. **User Feedback:** Gather stakeholder input for improvements
5. **Security:** Don't expose sensitive customer data publicly

---

## 📞 Support

For issues or questions:
- Check documentation in `/docs`
- Review notebooks in `/notebooks`
- Contact: [Your Email]

---

Built with ❤️ using Streamlit, Python, and Machine Learning