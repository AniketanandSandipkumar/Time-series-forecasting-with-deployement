📈Time Series Forecasting with Deployment

An **end-to-end Time Series Forecasting application** that integrates **statistical models, deep learning, interactive data visualization, and business intelligence dashboards** into a single deployable solution.

This project demonstrates the **complete data science lifecycle** — from exploratory data analysis and model training to forecasting, visualization, and executive-level insights using **Power BI**.

---

🚀 Key Features

- 📊 Interactive **Streamlit dashboard**
- 📈 Multiple **forecasting models**
- 🔍 Comprehensive **Exploratory Data Analysis (EDA)**
- 🧠 Deep learning–based forecasting with **LSTM**
- 📉 Time-series diagnostics (ACF, PACF, decomposition)
- 📥 Downloadable predictions & future forecasts
- 📊 Embedded **Power BI executive dashboard**
- 🌐 Deployment-ready project structure

---

🧠 Forecasting Models Used

| Model   | Description |
|--------|-------------|
| ARIMA  | Classical time series forecasting |
| SARIMA | Seasonal ARIMA for periodic patterns |
| Prophet | Robust trend & seasonality forecasting |
| LSTM | Deep learning model for sequence prediction |

Each model supports:
- Training on uploaded data  
- Evaluation on test split  
- Visualization of actual vs predicted values  
- Future forecasting with configurable horizons  

---

📊 Exploratory Data Analysis (EDA)

The application provides rich EDA capabilities, including:

- Monthly resampling (mean)
- Rolling mean analysis (30 days)
- Seasonal decomposition (trend, seasonality, residuals)
- Candlestick (OHLC) charts
- Daily returns distribution
- Volume vs price analysis
- Correlation heatmap
- Autocorrelation (ACF)
- Partial autocorrelation (PACF)

All EDA features are **interactive and controlled from the sidebar**.

---

📊 Business Intelligence (Power BI)

This project uniquely integrates **Power BI** within a machine learning application.

### Power BI Dashboard Highlights
- Executive-level market insights
- High-level performance metrics
- Business-friendly visualizations
- Complements ML-based forecasting results

The Power BI report is embedded directly inside the Streamlit app using an iframe for a seamless analytics experience.

---

🖥️ Application Workflow

1. Upload a stock price CSV file
2. View and filter the dataset
3. Perform exploratory data analysis
4. Select forecasting features and models
5. Train models and evaluate results
6. Generate future forecasts
7. Download predictions
8. View executive insights via Power BI dashboard

---

📁 Project Structure  
Time-series-forecasting-with-deployment/  
│  
├── app.py # Main Streamlit application<br>
├── train_models.py # Model training & evaluation logic<br>
├── eda_utils.py # EDA & visualization utilities<br>
├── requirements.txt # Python dependencies<br>
├── README.md # Project documentation<br>
└── .devcontainer/ # Development container setup <br>

---

🛠️ Tech Stack

### Programming & Libraries
- Python
- Pandas, NumPy
- Matplotlib, Plotly
- Statsmodels
- Scikit-learn
- TensorFlow / Keras
- Prophet

### Visualization & BI
- Streamlit
- Power BI (Embedded)

### Concepts & Skills
- Time Series Analysis
- Forecasting & Model Evaluation
- Deep Learning (LSTM)
- Data Visualization
- Business Intelligence
- ML Deployment

---

⚙️ Installation & Usage  

1️⃣ Clone the Repository  
git clone https://github.com/your-username/Time-series-forecasting-with-deployment.git  
cd Time-series-forecasting-with-deployment

2️⃣ Install Dependencies  
pip install -r requirements.txt  

3️⃣ Run the Application  
streamlit run app.py

📥 Input Data Format-  
The uploaded CSV file should contain:  
A datetime index<br><br>
Stock-related columns such as:<br>
Open<br>
High<br>
Low<br>
Close<br>
Volume<br>

Example:<br>
Date,Open,High,Low,Close,Volume<br>
2023-01-01,120,125,118,123,1000000<br>

🌟 Why This Project Matters:<br>
-Demonstrates real-world data science workflow<br>
-Combines ML, DL, visualization, and BI<br>
-Production-ready and deployable<br>
-Suitable for internships, final-year projects, and portfolios<br>
-Strong focus on both technical depth and business insights<br>

👨‍💻 Author<br>
Aniketanand Sandipkumar<br>
Final-year B.Tech (Computer Science)<br>
🔗App Link:https://time-series-forecasting-with-deployement-fpra5tvjd4ekczbmmtvr3.streamlit.app/



