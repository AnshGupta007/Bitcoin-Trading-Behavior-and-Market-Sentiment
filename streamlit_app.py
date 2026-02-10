"""
Bitcoin Trading Behavior & Market Sentiment - Interactive Dashboard
====================================================================
Professional Streamlit dashboard for ML model deployment and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ===========================
# PAGE CONFIG (must be first Streamlit command)
# ===========================
st.set_page_config(
    page_title="Bitcoin Trading Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h2 {
        color: white !important;
        margin: 0;
    }
    .metric-card p {
        color: #e0e0e0;
        margin: 0;
        font-size: 0.9rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ===========================
# HELPER FUNCTIONS
# ===========================

@st.cache_resource
def load_model():
    """Load the trained ML model, scaler, and feature columns"""
    model, scaler, feature_columns, label_encoder = None, None, None, None

    model_paths = {
        'model': 'models/best_model.pkl',
        'scaler': 'models/scaler.pkl',
        'features': 'models/feature_columns.pkl',
        'encoder': 'models/label_encoder.pkl'
    }

    # Check which files exist
    missing = [k for k, v in model_paths.items() if not os.path.exists(v)]
    if 'model' in missing:
        return None, None, None, None, missing

    try:
        model = joblib.load(model_paths['model'])
    except Exception as e:
        return None, None, None, None, [f"model load error: {e}"]

    try:
        scaler = joblib.load(model_paths['scaler'])
    except FileNotFoundError:
        scaler = None  # Not all models need a scaler

    try:
        feature_columns = joblib.load(model_paths['features'])
    except FileNotFoundError:
        feature_columns = None

    try:
        label_encoder = joblib.load(model_paths['encoder'])
    except FileNotFoundError:
        label_encoder = None

    return model, scaler, feature_columns, label_encoder, missing


@st.cache_data
def load_data():
    """Load trading and sentiment datasets with robust parsing"""
    trader_df, sentiment_df = None, None
    errors = []

    # --- Load Trading Data ---
    trade_path = 'csv_files/historical_data.csv'
    if os.path.exists(trade_path):
        try:
            trader_df = pd.read_csv(trade_path)

            # Try multiple datetime columns / formats
            dt_col = None
            for candidate in ['Timestamp IST', 'Time', 'Timestamp', 'Date', 'datetime']:
                if candidate in trader_df.columns:
                    dt_col = candidate
                    break

            if dt_col:
                for fmt in ['%d-%m-%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', None]:
                    try:
                        trader_df[dt_col] = pd.to_datetime(trader_df[dt_col], format=fmt, errors='coerce')
                        if trader_df[dt_col].notna().sum() > 0:
                            break
                    except Exception:
                        continue

            # Ensure required columns exist
            if 'Closed PnL' not in trader_df.columns:
                numeric_cols = trader_df.select_dtypes(include=[np.number]).columns
                pnl_candidates = [c for c in numeric_cols if 'pnl' in c.lower() or 'profit' in c.lower()]
                if pnl_candidates:
                    trader_df['Closed PnL'] = trader_df[pnl_candidates[0]]
                else:
                    trader_df['Closed PnL'] = 0
                    errors.append("'Closed PnL' column not found — defaulted to 0")

            if 'Side' not in trader_df.columns:
                trader_df['Side'] = 'Unknown'

            if 'Coin' not in trader_df.columns:
                coin_candidates = [c for c in trader_df.columns if 'coin' in c.lower() or 'symbol' in c.lower()]
                if coin_candidates:
                    trader_df['Coin'] = trader_df[coin_candidates[0]]
                else:
                    trader_df['Coin'] = 'BTC'

        except Exception as e:
            errors.append(f"Trading data error: {e}")
            trader_df = None
    else:
        errors.append(f"File not found: {trade_path}")

    # --- Load Sentiment Data ---
    sentiment_path = 'csv_files/fear_greed_index.csv'
    if os.path.exists(sentiment_path):
        try:
            sentiment_df = pd.read_csv(sentiment_path)

            # Find date column
            date_col = None
            for candidate in ['date', 'Date', 'timestamp', 'Timestamp']:
                if candidate in sentiment_df.columns:
                    date_col = candidate
                    break

            if date_col:
                sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col], errors='coerce')
                if date_col != 'date':
                    sentiment_df.rename(columns={date_col: 'date'}, inplace=True)

            # Ensure required columns
            if 'value' not in sentiment_df.columns:
                val_candidates = [c for c in sentiment_df.columns if 'value' in c.lower() or 'index' in c.lower()]
                if val_candidates:
                    sentiment_df['value'] = pd.to_numeric(sentiment_df[val_candidates[0]], errors='coerce')
                else:
                    sentiment_df['value'] = 50
                    errors.append("'value' column not found in sentiment data — defaulted to 50")

            if 'classification' not in sentiment_df.columns:
                class_candidates = [c for c in sentiment_df.columns
                                    if 'class' in c.lower() or 'category' in c.lower() or 'sentiment' in c.lower()]
                if class_candidates:
                    sentiment_df['classification'] = sentiment_df[class_candidates[0]]
                else:
                    sentiment_df['classification'] = 'Neutral'

        except Exception as e:
            errors.append(f"Sentiment data error: {e}")
            sentiment_df = None
    else:
        errors.append(f"File not found: {sentiment_path}")

    return trader_df, sentiment_df, errors


def build_feature_vector(inputs, feature_columns, label_encoder=None):
    """
    Build a feature vector that matches the trained model's expected features.
    Returns a DataFrame with the correct column order.
    """
    # Map user inputs to all possible feature names
    feature_map = {
        'Size USD_mean': inputs.get('size_mean', 0),
        'Size USD_sum': inputs.get('size_sum', 0),
        'Size USD_std': inputs.get('size_std', 0),
        'Size USD_count': inputs.get('trade_count', 0),
        'Closed PnL_mean': inputs.get('pnl_mean', 0),
        'Closed PnL_std': inputs.get('pnl_std', 0),
        'Execution Price_mean': inputs.get('price_mean', 0),
        'Execution Price_std': inputs.get('price_std', 0),
        'Fee_sum': inputs.get('fee_sum', 0),
        'Hour': inputs.get('hour', 12),
        'value': inputs.get('sentiment_value', 50),
        'avg_profit_per_trade': inputs.get('pnl_mean', 0),
        'total_daily_volume': inputs.get('size_sum', 0),
        'trade_frequency': inputs.get('trade_count', 0),
        'volatility': inputs.get('price_std', 0),
        'profit_volatility': inputs.get('pnl_std', 0),
        'buy_sell_ratio': inputs.get('buy_sell_ratio', 1.0),
        'sentiment_encoded': inputs.get('sentiment_encoded', 2),
    }

    if feature_columns is not None:
        # Use exact feature order from training
        feature_values = {col: feature_map.get(col, 0) for col in feature_columns}
        return pd.DataFrame([feature_values])[feature_columns]
    else:
        # Fallback: use all mapped features
        return pd.DataFrame([feature_map])


def predict_profitability(feature_df, model, scaler):
    """Make prediction with proper error handling"""
    try:
        # Determine if scaling is needed
        model_name = type(model).__name__.lower()
        needs_scaling = any(name in model_name for name in ['logistic', 'svc', 'svm', 'knn', 'mlp'])

        if needs_scaling and scaler is not None:
            feature_array = scaler.transform(feature_df)
        else:
            feature_array = feature_df.values

        prediction = model.predict(feature_array)[0]

        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(feature_array)[0]
        else:
            probability = np.array([1 - prediction, prediction], dtype=float)

        return prediction, probability, None

    except Exception as e:
        return None, None, str(e)


def render_metric_card(label, value, icon="📊"):
    """Render a styled metric card using pure HTML"""
    st.markdown(f"""
    <div class="metric-card">
        <p>{icon} {label}</p>
        <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)


# ===========================
# MAIN APP
# ===========================

def main():
    # Header
    st.markdown(
        '<p class="main-header">📈 Bitcoin Trading Sentiment Analyzer</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        "**Predict trading outcomes using market sentiment and trading patterns**"
    )

    # Load model and data
    model, scaler, feature_columns, label_encoder, model_errors = load_model()
    trader_df, sentiment_df, data_errors = load_data()

    # --- Status indicators in sidebar ---
    st.sidebar.title("🔧 System Status")

    if model is not None:
        st.sidebar.success(f"✅ Model loaded: `{type(model).__name__}`")
    else:
        st.sidebar.error("❌ Model not loaded")
        if model_errors:
            for err in model_errors:
                st.sidebar.warning(f"  → {err}")

    if trader_df is not None:
        st.sidebar.success(f"✅ Trading data: {len(trader_df):,} rows")
    else:
        st.sidebar.error("❌ Trading data not loaded")

    if sentiment_df is not None:
        st.sidebar.success(f"✅ Sentiment data: {len(sentiment_df):,} rows")
    else:
        st.sidebar.error("❌ Sentiment data not loaded")

    if data_errors:
        with st.sidebar.expander("⚠️ Data Warnings"):
            for err in data_errors:
                st.warning(err)

    st.sidebar.markdown("---")

    # Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Dashboard", "🔮 Make Predictions", "📊 Data Explorer", "📈 Model Insights"]
    )

    # ===========================
    # PAGE 1: DASHBOARD
    # ===========================
    if page == "🏠 Dashboard":
        st.markdown(
            '<p class="sub-header">Overview & Key Metrics</p>',
            unsafe_allow_html=True
        )

        if trader_df is None and sentiment_df is None:
            st.error("⚠️ No data available. Place CSV files in `csv_files/` directory.")
            st.stop()

        # Key Metrics Row
        if trader_df is not None:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                render_metric_card("Total Trades", f"{len(trader_df):,}", "📋")
            with col2:
                profitable_trades = (trader_df['Closed PnL'] > 0).sum()
                win_rate = profitable_trades / len(trader_df) * 100 if len(trader_df) > 0 else 0
                render_metric_card("Win Rate", f"{win_rate:.1f}%", "🎯")
            with col3:
                total_pnl = trader_df['Closed PnL'].sum()
                render_metric_card("Total P&L", f"${total_pnl:,.2f}", "💰")
            with col4:
                avg_pnl = trader_df['Closed PnL'].mean()
                render_metric_card("Avg P&L/Trade", f"${avg_pnl:,.2f}", "📊")

        st.markdown("---")

        # Charts Row
        col1, col2 = st.columns(2)

        with col1:
            if sentiment_df is not None and 'classification' in sentiment_df.columns:
                st.subheader("📊 Market Sentiment Distribution")
                sentiment_counts = sentiment_df['classification'].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sentiment classification data not available.")

        with col2:
            if trader_df is not None:
                st.subheader("💰 Profit Distribution")
                pnl_data = trader_df['Closed PnL'].dropna()
                # Clip outliers for better visualization
                q1, q99 = pnl_data.quantile(0.01), pnl_data.quantile(0.99)
                pnl_clipped = pnl_data[(pnl_data >= q1) & (pnl_data <= q99)]

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=pnl_clipped, nbinsx=50,
                    marker_color='lightblue', opacity=0.7,
                    name='P&L'
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="red",
                              annotation_text="Break Even")
                fig.update_layout(
                    xaxis_title="Closed P&L (USD)",
                    yaxis_title="Frequency",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        # Sentiment Time Series
        if sentiment_df is not None and 'date' in sentiment_df.columns and 'value' in sentiment_df.columns:
            st.subheader("📈 Sentiment Value Over Time")
            sentiment_sorted = sentiment_df.sort_values('date').dropna(subset=['date', 'value'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sentiment_sorted['date'],
                y=sentiment_sorted['value'],
                mode='lines',
                fill='tozeroy',
                line=dict(color='royalblue', width=2),
                name='Sentiment Value'
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="red",
                          annotation_text="Neutral (50)")
            fig.add_hline(y=25, line_dash="dot", line_color="orange",
                          annotation_text="Extreme Fear (25)")
            fig.add_hline(y=75, line_dash="dot", line_color="green",
                          annotation_text="Greed (75)")
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Sentiment Value (0-100)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # PAGE 2: MAKE PREDICTIONS
    # ===========================
    elif page == "🔮 Make Predictions":
        st.markdown(
            '<p class="sub-header">Predict Trading Day Profitability</p>',
            unsafe_allow_html=True
        )

        if model is None:
            st.markdown("""
            <div class="error-box">
            <h4>❌ Model Not Available</h4>
            <p>Please train the model first by running your ML pipeline notebook.</p>
            <p>Required files in <code>models/</code> directory:</p>
            <ul>
                <li><code>best_model.pkl</code> (required)</li>
                <li><code>scaler.pkl</code> (optional)</li>
                <li><code>feature_columns.pkl</code> (recommended)</li>
                <li><code>label_encoder.pkl</code> (optional)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        st.markdown(
            '<div class="info-box">Enter trading parameters to predict if the '
            'day will be profitable based on market sentiment and trading patterns.</div>',
            unsafe_allow_html=True
        )

        # Show expected features
        if feature_columns is not None:
            with st.expander("ℹ️ Model expects these features"):
                st.write(feature_columns)

        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**📦 Trading Volume Metrics**")
                size_mean = st.number_input(
                    "Avg Trade Size (USD)", min_value=0.0, value=1000.0, step=100.0,
                    help="Average size of individual trades in USD"
                )
                size_sum = st.number_input(
                    "Total Daily Volume (USD)", min_value=0.0, value=50000.0, step=1000.0,
                    help="Total trading volume for the day"
                )
                size_std = st.number_input(
                    "Volume Std Dev", min_value=0.0, value=500.0, step=50.0,
                    help="Standard deviation of trade sizes"
                )
                trade_count = st.number_input(
                    "Number of Trades", min_value=1, value=50, step=1,
                    help="Total number of trades executed"
                )

            with col2:
                st.markdown("**💹 Profit & Price Metrics**")
                pnl_mean = st.number_input(
                    "Avg P&L per Trade (USD)", value=10.0, step=5.0,
                    help="Average profit/loss per trade"
                )
                pnl_std = st.number_input(
                    "P&L Std Dev", min_value=0.0, value=50.0, step=10.0,
                    help="Variability in profit/loss across trades"
                )
                price_mean = st.number_input(
                    "Avg Execution Price (USD)", min_value=0.0, value=50000.0, step=1000.0,
                    help="Average price at which trades were executed"
                )
                price_std = st.number_input(
                    "Price Std Dev", min_value=0.0, value=1000.0, step=100.0,
                    help="Price volatility during the trading day"
                )

            with col3:
                st.markdown("**⚙️ Other Metrics**")
                fee_sum = st.number_input(
                    "Total Fees (USD)", min_value=0.0, value=100.0, step=10.0,
                    help="Total fees paid for all trades"
                )
                hour = st.slider(
                    "Peak Trading Hour", min_value=0, max_value=23, value=12,
                    help="Hour of peak trading activity (0-23)"
                )
                sentiment_value = st.slider(
                    "Market Sentiment (0-100)", min_value=0, max_value=100, value=50,
                    help="Fear & Greed Index value"
                )

                sentiment_options = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
                sentiment_class = st.selectbox(
                    "Sentiment Classification", sentiment_options, index=2
                )
                buy_sell_ratio = st.number_input(
                    "Buy/Sell Ratio", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                    help="Ratio of buy trades to sell trades"
                )

            submitted = st.form_submit_button(
                "🔮 Predict Profitability", use_container_width=True
            )

        if submitted:
            # Encode sentiment
            if label_encoder is not None:
                try:
                    sentiment_encoded = label_encoder.transform([sentiment_class])[0]
                except ValueError:
                    # Class not seen during training — use a safe default
                    sentiment_encoded = label_encoder.transform(
                        [label_encoder.classes_[0]]
                    )[0]
                    st.warning(
                        f"⚠️ '{sentiment_class}' not in training labels. "
                        f"Using '{label_encoder.classes_[0]}' as fallback."
                    )
            else:
                # Fallback manual encoding (alphabetical order from LabelEncoder)
                encoding_map = {
                    "Extreme Fear": 0, "Extreme Greed": 1,
                    "Fear": 2, "Greed": 3, "Neutral": 4
                }
                sentiment_encoded = encoding_map.get(sentiment_class, 4)

            # Build feature vector
            user_inputs = {
                'size_mean': size_mean,
                'size_sum': size_sum,
                'size_std': size_std,
                'trade_count': trade_count,
                'pnl_mean': pnl_mean,
                'pnl_std': pnl_std,
                'price_mean': price_mean,
                'price_std': price_std,
                'fee_sum': fee_sum,
                'hour': hour,
                'sentiment_value': sentiment_value,
                'buy_sell_ratio': buy_sell_ratio,
                'sentiment_encoded': sentiment_encoded,
            }

            feature_df = build_feature_vector(user_inputs, feature_columns, label_encoder)

            # Predict
            prediction, probability, error = predict_profitability(feature_df, model, scaler)

            if error:
                st.error(f"❌ Prediction failed: {error}")
                with st.expander("🔍 Debug Info"):
                    st.write("**Feature DataFrame:**")
                    st.dataframe(feature_df)
                    st.write(f"**Model type:** {type(model).__name__}")
                    if feature_columns is not None:
                        st.write(f"**Expected features ({len(feature_columns)}):** {feature_columns}")
                    st.write(f"**Provided features ({len(feature_df.columns)}):** {list(feature_df.columns)}")
                st.stop()

            # --- Display Results ---
            st.markdown("---")
            st.subheader("🎯 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction == 1:
                    st.markdown(
                        '<div class="success-box">'
                        '<h3>✅ PROFITABLE Day Predicted!</h3>'
                        '<p>The model predicts this trading day will be profitable.</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="warning-box">'
                        '<h3>⚠️ NOT Profitable Day Predicted</h3>'
                        '<p>The model predicts this trading day may result in a loss.</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )

            with col2:
                st.metric(
                    "Probability of Profit",
                    f"{probability[1] * 100:.1f}%",
                    delta=f"{(probability[1] - 0.5) * 100:+.1f}% vs neutral"
                )

            with col3:
                confidence = max(probability) * 100
                confidence_label = (
                    "🟢 High" if confidence > 80
                    else "🟡 Moderate" if confidence > 60
                    else "🔴 Low"
                )
                st.metric("Confidence", f"{confidence:.1f}%", delta=confidence_label)

            # Probability Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Profitability Probability", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "green"},
                       'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': '#ff6b6b'},
                        {'range': [25, 50], 'color': '#ffd93d'},
                        {'range': [50, 75], 'color': '#6bcb77'},
                        {'range': [75, 100], 'color': '#2d6a4f'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            # Show feature breakdown
            with st.expander("📋 Feature Values Used for Prediction"):
                st.dataframe(feature_df.T.rename(columns={0: 'Value'}), use_container_width=True)

    # ===========================
    # PAGE 3: DATA EXPLORER
    # ===========================
    elif page == "📊 Data Explorer":
        st.markdown(
            '<p class="sub-header">Explore Trading & Sentiment Data</p>',
            unsafe_allow_html=True
        )

        if trader_df is None and sentiment_df is None:
            st.error("⚠️ No data available. Place CSV files in `csv_files/` directory.")
            st.stop()

        tab1, tab2 = st.tabs(["📊 Trading Data", "🧠 Sentiment Data"])

        with tab1:
            if trader_df is not None:
                st.subheader("📋 Historical Trading Records")

                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    available_sides = trader_df['Side'].dropna().unique().tolist()
                    side_filter = st.multiselect(
                        "Filter by Side", options=available_sides,
                        default=available_sides
                    )
                with col2:
                    available_coins = trader_df['Coin'].dropna().unique().tolist()
                    default_coins = available_coins[:min(5, len(available_coins))]
                    coin_filter = st.multiselect(
                        "Filter by Coin", options=available_coins,
                        default=default_coins
                    )

                # Apply filters
                filtered_df = trader_df[
                    trader_df['Side'].isin(side_filter) & trader_df['Coin'].isin(coin_filter)
                ]

                st.write(f"Showing **{len(filtered_df):,}** of {len(trader_df):,} records")

                # Summary stats
                if len(filtered_df) > 0:
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.metric("Trades", f"{len(filtered_df):,}")
                    with stat_col2:
                        st.metric("Avg P&L", f"${filtered_df['Closed PnL'].mean():.2f}")
                    with stat_col3:
                        st.metric("Total P&L", f"${filtered_df['Closed PnL'].sum():,.2f}")
                    with stat_col4:
                        win_rate = (filtered_df['Closed PnL'] > 0).mean() * 100
                        st.metric("Win Rate", f"{win_rate:.1f}%")

                st.dataframe(filtered_df.head(1000), height=400, use_container_width=True)

                # Download
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv_data,
                    file_name=f"filtered_trading_data_{datetime.now():%Y%m%d}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Trading data not available.")

        with tab2:
            if sentiment_df is not None:
                st.subheader("📋 Market Sentiment History")
                st.dataframe(sentiment_df, height=400, use_container_width=True)

                st.subheader("📊 Sentiment Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average", f"{sentiment_df['value'].mean():.1f}")
                with col2:
                    st.metric("Median", f"{sentiment_df['value'].median():.1f}")
                with col3:
                    st.metric("Std Dev", f"{sentiment_df['value'].std():.1f}")
                with col4:
                    st.metric("Range",
                              f"{sentiment_df['value'].min():.0f} - {sentiment_df['value'].max():.0f}")

                # Sentiment heatmap by month
                if 'date' in sentiment_df.columns:
                    st.subheader("📅 Monthly Sentiment Heatmap")
                    sent_copy = sentiment_df.dropna(subset=['date']).copy()
                    sent_copy['month'] = sent_copy['date'].dt.month_name()
                    sent_copy['year'] = sent_copy['date'].dt.year
                    monthly = sent_copy.groupby(['year', 'month'])['value'].mean().reset_index()

                    if len(monthly) > 1:
                        pivot = monthly.pivot(index='year', columns='month', values='value')
                        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                       'July', 'August', 'September', 'October', 'November', 'December']
                        pivot = pivot[[m for m in month_order if m in pivot.columns]]

                        fig = px.imshow(
                            pivot, color_continuous_scale='RdYlGn',
                            labels=dict(x="Month", y="Year", color="Sentiment"),
                            aspect="auto"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Sentiment data not available.")

    # ===========================
    # PAGE 4: MODEL INSIGHTS
    # ===========================
    elif page == "📈 Model Insights":
        st.markdown(
            '<p class="sub-header">Machine Learning Model Performance</p>',
            unsafe_allow_html=True
        )

        if model is None:
            st.error("⚠️ Model not loaded. Train the model first.")
            st.stop()

        # Model info card
        model_name = type(model).__name__
        n_features = len(feature_columns) if feature_columns else 'N/A'

        st.markdown(f"""
        <div class="info-box">
        <h4>🤖 Model Information</h4>
        <table style="width:100%">
            <tr><td><strong>Algorithm:</strong></td><td>{model_name}</td></tr>
            <tr><td><strong>Features Used:</strong></td><td>{n_features}</td></tr>
            <tr><td><strong>Scaler Available:</strong></td><td>{'✅ Yes' if scaler else '❌ No'}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # Feature columns list
        if feature_columns is not None:
            with st.expander("📋 Feature Columns"):
                for i, col in enumerate(feature_columns, 1):
                    st.write(f"  {i}. `{col}`")

        # Saved visualizations
        st.subheader("📊 Performance Visualizations")

        if os.path.exists('outputs'):
            viz_files = [
                ("outputs/07_model_comparison.png", "Model Performance Comparison"),
                ("outputs/08_confusion_matrix.png", "Confusion Matrix"),
                ("outputs/09_roc_curves.png", "ROC Curves"),
                ("outputs/10_feature_importance.png", "Feature Importance"),
            ]

            found_any = False
            for filepath, title in viz_files:
                if os.path.exists(filepath):
                    found_any = True
                    with st.expander(f"📊 {title}", expanded=True):
                        st.image(filepath, use_container_width=True)

            if not found_any:
                st.info("No visualization files found in `outputs/` directory.")
        else:
            st.warning("📁 `outputs/` folder not found. Run the ML pipeline notebook first.")

        # Model Parameters
        st.subheader("⚙️ Model Parameters")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            params_df = pd.DataFrame({
                'Parameter': list(params.keys()),
                'Value': [str(v) for v in params.values()]
            })
            st.dataframe(params_df, height=400, use_container_width=True)

        # Feature importance (live from model)
        if hasattr(model, 'feature_importances_') and feature_columns is not None:
            st.subheader("🏆 Live Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)

            fig = px.bar(
                importance_df, x='Importance', y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='viridis',
                title="Feature Importance (from trained model)"
            )
            fig.update_layout(height=max(400, len(feature_columns) * 25))
            st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # FOOTER
    # ===========================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem 0;'>
        <p>© 2025 Bitcoin Trading Sentiment Analyzer | Built with Streamlit & Python</p>
        <p>Created by <strong>Ansh Gupta</strong></p>
        <p style='font-size: 0.8rem;'>
            📦 Model: {model_type} | 📊 Data: {n_trades} trades |
            🧠 Sentiment: {n_sent} records
        </p>
    </div>
    """.format(
        model_type=type(model).__name__ if model else 'N/A',
        n_trades=f"{len(trader_df):,}" if trader_df is not None else 'N/A',
        n_sent=f"{len(sentiment_df):,}" if sentiment_df is not None else 'N/A'
    ), unsafe_allow_html=True)


if __name__ == "__main__":
    main()