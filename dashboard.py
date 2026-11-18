import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Stock Market Analytics", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp {background:#0b1120;color:#e5e7eb}
[data-testid="stSidebar"] {background:#111827;border-right:1px solid #263247}
.block-container {padding-top:1.25rem;max-width:1500px}
h1,h2,h3 {color:#f8fafc!important}
div[data-testid="stMetric"] {background:#111827;border:1px solid #263247;border-radius:12px;padding:14px 16px;min-height:105px}
div[data-testid="stMetricLabel"] {color:#94a3b8}
div[data-testid="stMetricValue"] {color:#f8fafc}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol, start, end_exclusive):
    raw = yf.download(symbol, start=start, end=end_exclusive, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c) for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    raw.index.name = "Date"
    df = raw.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            raise ValueError(f"Missing expected Yahoo Finance column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["Date", "Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Daily_Return"] = df["Close"].pct_change()
    for window in (20, 50, 200):
        df[f"SMA{window}"] = df["Close"].rolling(window, min_periods=window).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    delta = df["Close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gain.rolling(14, min_periods=14).mean(), loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    df.loc[(avg_gain == 0) & (avg_loss > 0), "RSI"] = 0
    df.loc[(avg_gain == 0) & (avg_loss == 0), "RSI"] = 50
    ema12, ema26 = df["Close"].ewm(span=12, adjust=False).mean(), df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["Drawdown"] = df["Close"] / df["Close"].cummax() - 1
    return df

def money(v):
    return f"${v:,.2f}" if pd.notna(v) else "N/A"
def percent(v, signed=False):
    return "N/A" if pd.isna(v) else (f"{v*100:+.2f}%" if signed else f"{v*100:.2f}%")
def style_chart(fig, height=420, title=None):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="#0b1120", plot_bgcolor="#111827",
        font=dict(color="#cbd5e1", family="Arial"), margin=dict(l=15,r=20,t=55,b=30), hovermode="x unified",
        legend=dict(orientation="h",y=1.02,x=0), xaxis=dict(gridcolor="#1f2937",rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#1f2937",side="right"))
    if title: fig.update_layout(title=title)
    return fig

# Sidebar
st.sidebar.markdown("## 📊 Market Analyzer")
st.sidebar.caption("Stock market research dashboard")
st.sidebar.divider()
ticker = st.sidebar.text_input("Stock Symbol", "AAPL").strip().upper()
preset = st.sidebar.selectbox("Quick Date Range", ["1 Month","3 Months","6 Months","1 Year","2 Years","5 Years","Custom"], index=3)
today = date.today()
days = {"1 Month":30,"3 Months":90,"6 Months":180,"1 Year":365,"2 Years":730,"5 Years":1825}
if preset == "Custom":
    start_date = st.sidebar.date_input("Start Date", date(2023,1,1))
    end_date = st.sidebar.date_input("End Date", today)
else:
    start_date, end_date = today-timedelta(days=days[preset]), today
st.sidebar.divider()
st.sidebar.caption("Source: Yahoo Finance")
st.sidebar.caption("Educational use only. Not financial advice.")

st.title("📈 Stock Market Analytics")
st.caption("Analyze market trends, price action, momentum and volatility.")
if not ticker:
    st.warning("Enter a ticker symbol in the sidebar."); st.stop()
if start_date >= end_date:
    st.error("Start date must be earlier than end date."); st.stop()
try:
    with st.spinner(f"Fetching {ticker} market data..."):
        data = fetch_data(ticker, start_date, end_date + timedelta(days=1))
except Exception as exc:
    st.error(f"Could not retrieve data for {ticker}: {exc}"); st.stop()
if data.empty:
    st.error("No data returned. Verify ticker symbol and date range."); st.stop()
if len(data) < 2:
    st.warning("Only one row of data returned. Select a longer period."); st.stop()

latest, previous = data.iloc[-1], data.iloc[-2]
price, prev = float(latest.Close), float(previous.Close)
daily_change_pct = price/prev-1 if prev else np.nan
period_return = price/float(data.iloc[0].Close)-1
returns = data.Daily_Return.dropna()
daily_vol = returns.std() if len(returns)>1 else np.nan
annual_vol = daily_vol*np.sqrt(252) if pd.notna(daily_vol) else np.nan
avg_ret = returns.mean() if len(returns) else np.nan
avg_vol = data.Volume.mean()
rsi_now = latest.RSI
max_dd = data.Drawdown.min()

st.subheader("Market Snapshot")
a,b,c,d = st.columns(4)
a.metric("Latest Price", money(price), percent(daily_change_pct,True))
b.metric("Period Return", percent(period_return,True))
c.metric("Annualized Volatility", percent(annual_vol))
d.metric("Average Volume", f"{avg_vol:,.0f}" if pd.notna(avg_vol) else "N/A")
e,f,g,h = st.columns(4)
e.metric("Period High", money(data.High.max()))
f.metric("Period Low", money(data.Low.min()))
g.metric("Average Daily Return", percent(avg_ret,True))
h.metric("RSI (14)", f"{rsi_now:.2f}" if pd.notna(rsi_now) else "N/A")
st.divider()
t_overview,t_technical,t_risk,t_data = st.tabs(["Overview","Technical Analysis","Returns & Risk","Data Explorer"])

with t_overview:
    st.subheader("Price Performance")
    x1,x2,x3,x4 = st.columns([1.3,1,1,1.4])
    chart_type = x1.selectbox("Chart Type",["Candlestick","Line"],key="chart_type")
    sma20_on = x2.checkbox("SMA 20",True)
    sma50_on = x3.checkbox("SMA 50",True)
    sma200_on = x4.checkbox("SMA 200",False)
    fig = go.Figure()
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=data.Date,open=data.Open,high=data.High,low=data.Low,close=data.Close,name=ticker,
            increasing=dict(line=dict(color="#16c784"),fillcolor="#16c784"),decreasing=dict(line=dict(color="#ea3943"),fillcolor="#ea3943")))
    else:
        fig.add_trace(go.Scatter(x=data.Date,y=data.Close,name="Close",mode="lines",line=dict(color="#38bdf8",width=2.4)))
    for enabled, col, name, color in [(sma20_on,"SMA20","SMA 20","#fbbf24"),(sma50_on,"SMA50","SMA 50","#a78bfa"),(sma200_on,"SMA200","SMA 200","#f472b6")]:
        if enabled:
            fig.add_trace(go.Scatter(x=data.Date,y=data[col],name=name,line=dict(color=color,width=1.6)))
    style_chart(fig,520,f"{ticker} · Price Chart")
    fig.update_layout(yaxis_title="Price (USD)")
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=1,label="1M",step="month",stepmode="backward"),dict(count=3,label="3M",step="month",stepmode="backward"),dict(count=6,label="6M",step="month",stepmode="backward"),dict(count=1,label="1Y",step="year",stepmode="backward"),dict(step="all",label="ALL")],bgcolor="#1f2937",activecolor="#2563eb",font=dict(color="#e5e7eb")))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"scrollZoom":True,"modeBarButtonsToRemove":["lasso2d","select2d"]})
    st.subheader("Trading Volume")
    vol_colors = np.where(data.Close>=data.Open,"#16c784","#ea3943")
    vf = go.Figure(go.Bar(x=data.Date,y=data.Volume,marker_color=vol_colors,opacity=.75,name="Volume"))
    style_chart(vf,260,f"{ticker} · Volume"); vf.update_layout(yaxis_title="Shares")
    st.plotly_chart(vf,use_container_width=True,config={"displaylogo":False})

with t_technical:
    st.subheader("Momentum Indicators")
    left,right = st.columns(2)
    rf = go.Figure(go.Scatter(x=data.Date,y=data.RSI,name="RSI (14)",line=dict(color="#a78bfa",width=2)))
    rf.add_hline(y=70,line_dash="dash",line_color="#ea3943"); rf.add_hline(y=30,line_dash="dash",line_color="#16c784"); rf.add_hline(y=50,line_dash="dot",line_color="#64748b")
    style_chart(rf,360,"Relative Strength Index (RSI)"); rf.update_layout(yaxis=dict(range=[0,100],title="RSI"))
    left.plotly_chart(rf,use_container_width=True)
    mf = go.Figure(); mf.add_trace(go.Bar(x=data.Date,y=data.MACD_Hist,name="Histogram",marker_color="#64748b",opacity=.55)); mf.add_trace(go.Scatter(x=data.Date,y=data.MACD,name="MACD",line=dict(color="#38bdf8",width=2))); mf.add_trace(go.Scatter(x=data.Date,y=data.MACD_Signal,name="Signal",line=dict(color="#fbbf24",width=2)))
    style_chart(mf,360,"MACD (12, 26, 9)"); right.plotly_chart(mf,use_container_width=True)
    maf=go.Figure(); maf.add_trace(go.Scatter(x=data.Date,y=data.Close,name="Close",line=dict(color="#e2e8f0",width=2)))
    maf.add_trace(go.Scatter(x=data.Date,y=data.SMA20,name="SMA 20",line=dict(color="#fbbf24"))); maf.add_trace(go.Scatter(x=data.Date,y=data.SMA50,name="SMA 50",line=dict(color="#a78bfa"))); maf.add_trace(go.Scatter(x=data.Date,y=data.SMA200,name="SMA 200",line=dict(color="#f472b6")))
    style_chart(maf,440,"Price and Moving Averages"); st.plotly_chart(maf,use_container_width=True)
    if pd.isna(rsi_now): msg="RSI unavailable due to insufficient history."
    elif rsi_now>=70: msg="RSI is above 70, commonly viewed as overbought territory."
    elif rsi_now<=30: msg="RSI is below 30, commonly viewed as oversold territory."
    else: msg="RSI is between 30 and 70, outside traditional extreme zones."
    st.info(msg)
    st.info("Price is above SMA 50." if pd.notna(latest.SMA50) and price>latest.SMA50 else "Price is below SMA 50." if pd.notna(latest.SMA50) else "SMA 50 unavailable (insufficient history).")
    st.info("SMA 20 is above SMA 50." if pd.notna(latest.SMA20) and pd.notna(latest.SMA50) and latest.SMA20>latest.SMA50 else "SMA 20 is below SMA 50." if pd.notna(latest.SMA20) and pd.notna(latest.SMA50) else "SMA comparison unavailable.")

with t_risk:
    st.subheader("Returns & Risk")
    l,r = st.columns(2)
    hist=go.Figure(go.Histogram(x=returns,nbinsx=45,name="Daily Returns",marker_color="#38bdf8",opacity=.8)); style_chart(hist,370,"Daily Return Distribution"); hist.update_layout(xaxis_title="Daily return",yaxis_title="Frequency"); l.plotly_chart(hist,use_container_width=True)
    growth=(1+data.Daily_Return.fillna(0)).cumprod(); gf=go.Figure(go.Scatter(x=data.Date,y=growth,name="Growth of $1",fill="tozeroy",line=dict(color="#16c784",width=2))); style_chart(gf,370,"Cumulative Return · Growth of $1"); gf.update_layout(yaxis_title="Growth"); r.plotly_chart(gf,use_container_width=True)
    dd=go.Figure(go.Scatter(x=data.Date,y=data.Drawdown*100,name="Drawdown",fill="tozeroy",line=dict(color="#ea3943",width=1.5))); style_chart(dd,300,"Historical Drawdown"); dd.update_layout(yaxis_title="Drawdown (%)"); st.plotly_chart(dd,use_container_width=True)
    q1,q2,q3=st.columns(3); q1.metric("Maximum Drawdown",percent(max_dd)); q2.metric("Daily Volatility",percent(daily_vol)); q3.metric("Annualized Volatility",percent(annual_vol))
    st.caption("Volatility is based on daily close-to-close returns and annualized using √252. Drawdown uses adjusted closing prices.")

with t_data:
    st.subheader("Historical Data Explorer")
    st.caption(f"{len(data):,} rows · {data.Date.min().date()} through {data.Date.max().date()}")
    st.dataframe(data.sort_values("Date",ascending=False),use_container_width=True,height=440)
    st.download_button("⬇️ Download CSV",data=data.to_csv(index=False).encode("utf-8"),file_name=f"{ticker}_historical_data.csv",mime="text/csv")

st.divider()
st.caption("Stock Market Analytics · Yahoo Finance data · Educational use only; not financial advice.")
