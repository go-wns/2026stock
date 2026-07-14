"""
NVIDIA (NVDA) 인터랙티브 주식 분석 대시보드
Streamlit Cloud 배포용 - yfinance + Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────
# 페이지 기본 설정
# ──────────────────────────────
st.set_page_config(
    page_title="NVIDIA 주식 분석 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TICKER = "NVDA"

# ──────────────────────────────
# 데이터 로드 (캐시 적용)
# ──────────────────────────────
@st.cache_data(ttl=600, show_spinner="엔비디아 주가 데이터를 불러오는 중...")
def load_data(period: str, interval: str) -> pd.DataFrame:
    df = yf.download(TICKER, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_info() -> dict:
    try:
        return yf.Ticker(TICKER).info or {}
    except Exception:
        return {}


# ──────────────────────────────
# 기술적 지표 계산
# ──────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 이동평균선
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    # 볼린저 밴드 (20일, 2σ)
    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * std20
    df["BB_lower"] = df["MA20"] - 2 * std20

    # RSI (14일)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # 일간 수익률
    df["Return"] = df["Close"].pct_change() * 100
    return df


# ──────────────────────────────
# 사이드바 - 옵션
# ──────────────────────────────
with st.sidebar:
    st.title("⚙️ 분석 설정")

    period_label = st.selectbox(
        "조회 기간",
        ["1개월", "3개월", "6개월", "1년", "2년", "5년", "전체"],
        index=3,
    )
    period_map = {
        "1개월": "1mo", "3개월": "3mo", "6개월": "6mo",
        "1년": "1y", "2년": "2y", "5년": "5y", "전체": "max",
    }
    period = period_map[period_label]
    interval = "1d" if period != "1mo" else "1h"

    chart_type = st.radio("차트 유형", ["캔들스틱", "라인"], horizontal=True)

    st.subheader("기술적 지표")
    show_ma = st.checkbox("이동평균선 (20/60/120)", value=True)
    show_bb = st.checkbox("볼린저 밴드", value=False)
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD", value=True)

    st.caption("데이터 출처: Yahoo Finance (10분 캐시)")

# ──────────────────────────────
# 데이터 로드 및 검증
# ──────────────────────────────
df_raw = load_data(period, interval)

if df_raw.empty:
    st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
    st.stop()

df = add_indicators(df_raw)
info = load_info()

# ──────────────────────────────
# 헤더 & 핵심 지표
# ──────────────────────────────
st.title("📈 NVIDIA (NVDA) 주식 분석 대시보드")

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change = last_close - prev_close
change_pct = (change / prev_close) * 100 if prev_close else 0.0

period_high = float(df["High"].max())
period_low = float(df["Low"].min())
avg_volume = float(df["Volume"].mean())
period_return = (last_close / float(df["Close"].iloc[0]) - 1) * 100
volatility = float(df["Return"].std() * np.sqrt(252)) if df["Return"].notna().sum() > 2 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("현재가", f"${last_close:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
c2.metric(f"{period_label} 수익률", f"{period_return:+.2f}%")
c3.metric(f"{period_label} 최고 / 최저", f"${period_high:,.2f}", f"최저 ${period_low:,.2f}", delta_color="off")
c4.metric("평균 거래량", f"{avg_volume/1e6:,.1f}M주")
c5.metric("연환산 변동성", f"{volatility:.1f}%")

if info:
    with st.expander("🏢 기업 정보 보기"):
        i1, i2, i3 = st.columns(3)
        mc = info.get("marketCap")
        pe = info.get("trailingPE")
        div = info.get("dividendYield")
        i1.write(f"**시가총액:** {'$' + format(mc/1e12, ',.2f') + 'T' if mc else 'N/A'}")
        i2.write(f"**PER (TTM):** {f'{pe:,.1f}' if pe else 'N/A'}")
        i3.write(f"**배당수익률:** {f'{div*100:.2f}%' if div else 'N/A'}")

st.divider()

# ──────────────────────────────
# 메인 차트 (가격 + 거래량 + RSI + MACD)
# ──────────────────────────────
rows = 2 + int(show_rsi) + int(show_macd)
row_heights = [0.55, 0.15] + [0.15] * (rows - 2)
row_heights = [h / sum(row_heights) for h in row_heights]

subplot_titles = ["주가", "거래량"]
if show_rsi:
    subplot_titles.append("RSI (14)")
if show_macd:
    subplot_titles.append("MACD")

fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=row_heights,
    subplot_titles=subplot_titles,
)

# 1) 가격
if chart_type == "캔들스틱":
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="NVDA",
            increasing_line_color="#e74c3c",  # 상승: 빨강 (한국식)
            decreasing_line_color="#3498db",  # 하락: 파랑
        ),
        row=1, col=1,
    )
else:
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="종가",
                   line=dict(color="#76b900", width=2)),
        row=1, col=1,
    )

if show_ma:
    for col, color in [("MA20", "#f39c12"), ("MA60", "#9b59b6"), ("MA120", "#95a5a6")]:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name=col,
                       line=dict(color=color, width=1.2)),
            row=1, col=1,
        )

if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB 상단",
                             line=dict(color="rgba(118,185,0,0.4)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB 하단",
                             line=dict(color="rgba(118,185,0,0.4)", width=1),
                             fill="tonexty", fillcolor="rgba(118,185,0,0.07)"), row=1, col=1)

# 2) 거래량
vol_colors = np.where(df["Close"] >= df["Open"], "#e74c3c", "#3498db")
fig.add_trace(
    go.Bar(x=df.index, y=df["Volume"], name="거래량",
           marker_color=vol_colors, opacity=0.6),
    row=2, col=1,
)

current_row = 3

# 3) RSI
if show_rsi:
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                   line=dict(color="#8e44ad", width=1.5)),
        row=current_row, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="blue", opacity=0.5, row=current_row, col=1)
    current_row += 1

# 4) MACD
if show_macd:
    hist_colors = np.where(df["MACD_hist"] >= 0, "#e74c3c", "#3498db")
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist",
                         marker_color=hist_colors, opacity=0.6), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color="#2c3e50", width=1.3)), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
                             line=dict(color="#e67e22", width=1.3)), row=current_row, col=1)

fig.update_layout(
    height=300 + rows * 180,
    hovermode="x unified",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=60, b=10),
)
fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)

st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────
# 수익률 분석 탭
# ──────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 수익률 분포", "📉 낙폭(Drawdown)", "🗂 원본 데이터"])

with tab1:
    ret = df["Return"].dropna()
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=ret, nbinsx=60, marker_color="#76b900", opacity=0.8))
    fig_hist.add_vline(x=float(ret.mean()), line_dash="dash", line_color="red",
                       annotation_text=f"평균 {ret.mean():.2f}%")
    fig_hist.update_layout(
        title="일간 수익률 분포", xaxis_title="일간 수익률 (%)",
        yaxis_title="빈도", height=400, margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("평균 일간 수익률", f"{ret.mean():.3f}%")
    s2.metric("표준편차", f"{ret.std():.3f}%")
    s3.metric("최대 상승일", f"{ret.max():+.2f}%")
    s4.metric("최대 하락일", f"{ret.min():+.2f}%")

with tab2:
    cummax = df["Close"].cummax()
    drawdown = (df["Close"] / cummax - 1) * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df.index, y=drawdown, fill="tozeroy",
                                line=dict(color="#c0392b", width=1.5), name="낙폭"))
    fig_dd.update_layout(
        title=f"고점 대비 낙폭 (MDD: {drawdown.min():.1f}%)",
        yaxis_title="낙폭 (%)", height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

with tab3:
    show_df = df_raw.sort_index(ascending=False).copy()
    st.dataframe(show_df, use_container_width=True, height=400)
    csv = show_df.to_csv().encode("utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", csv, file_name=f"NVDA_{period}.csv", mime="text/csv")

st.caption("⚠️ 본 대시보드는 정보 제공 목적이며 투자 권유가 아닙니다. 투자 판단과 책임은 본인에게 있습니다.")
