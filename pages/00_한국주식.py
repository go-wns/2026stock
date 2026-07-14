"""
한국 대표 기술주 분석 대시보드 (반도체 · 로봇 · 바이오)
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
    page_title="한국 기술주 분석 대시보드",
    page_icon="🇰🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────
# 섹터별 대표 종목 (KRX 티커: .KS=코스피, .KQ=코스닥)
# ──────────────────────────────
SECTORS = {
    "🔷 반도체": {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "한미반도체": "042700.KS",
        "리노공업": "058470.KQ",
        "DB하이텍": "000990.KS",
        "이오테크닉스": "039030.KQ",
    },
    "🤖 로봇": {
        "두산로보틱스": "454910.KS",
        "레인보우로보틱스": "277810.KQ",
        "로보티즈": "108490.KQ",
        "유진로봇": "056080.KQ",
        "티로보틱스": "117730.KQ",
    },
    "🧬 바이오": {
        "삼성바이오로직스": "207940.KS",
        "셀트리온": "068270.KS",
        "SK바이오팜": "326030.KS",
        "알테오젠": "196170.KQ",
        "유한양행": "000100.KS",
        "HLB": "028300.KQ",
    },
}

SECTOR_COLORS = {"🔷 반도체": "#3498db", "🤖 로봇": "#e67e22", "🧬 바이오": "#27ae60"}

# ──────────────────────────────
# 데이터 로드 (캐시 적용)
# ──────────────────────────────
@st.cache_data(ttl=600, show_spinner="주가 데이터를 불러오는 중...")
def load_single(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


@st.cache_data(ttl=600, show_spinner="여러 종목 데이터를 불러오는 중...")
def load_multi(tickers: tuple, period: str) -> pd.DataFrame:
    """여러 종목의 종가를 하나의 데이터프레임으로 반환"""
    df = yf.download(list(tickers), period=period, interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df.dropna(how="all")


# ──────────────────────────────
# 기술적 지표 계산
# ──────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * std20
    df["BB_lower"] = df["MA20"] - 2 * std20

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    df["Return"] = df["Close"].pct_change() * 100
    return df


def fmt_krw(v: float) -> str:
    return f"{v:,.0f}원"


# ──────────────────────────────
# 사이드바
# ──────────────────────────────
with st.sidebar:
    st.title("⚙️ 분석 설정")

    period_label = st.selectbox(
        "조회 기간",
        ["1개월", "3개월", "6개월", "1년", "2년", "5년"],
        index=3,
    )
    period_map = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo",
                  "1년": "1y", "2년": "2y", "5년": "5y"}
    period = period_map[period_label]

    st.divider()
    st.subheader("개별 종목 분석")
    sector = st.selectbox("섹터 선택", list(SECTORS.keys()))
    stock_name = st.selectbox("종목 선택", list(SECTORS[sector].keys()))
    ticker = SECTORS[sector][stock_name]

    chart_type = st.radio("차트 유형", ["캔들스틱", "라인"], horizontal=True)
    show_ma = st.checkbox("이동평균선 (20/60/120)", value=True)
    show_bb = st.checkbox("볼린저 밴드", value=False)
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD", value=False)

    st.caption("데이터 출처: Yahoo Finance (10분 캐시)\n.KS=코스피 / .KQ=코스닥")

# ══════════════════════════════
# 섹션 1. 섹터 전체 비교
# ══════════════════════════════
st.title("🇰🇷 한국 대표 기술주 대시보드")
st.subheader(f"📊 섹터별 수익률 비교 ({period_label})")

all_map = {name: tk for stocks in SECTORS.values() for name, tk in stocks.items()}
ticker_to_name = {tk: name for name, tk in all_map.items()}
ticker_to_sector = {tk: sec for sec, stocks in SECTORS.items() for tk in stocks.values()}

multi = load_multi(tuple(all_map.values()), period)

if multi.empty:
    st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
    st.stop()

# 정규화 수익률 (시작일 = 0%)
norm = (multi / multi.iloc[0] - 1) * 100

fig_cmp = go.Figure()
for tk in norm.columns:
    name = ticker_to_name.get(tk, tk)
    sec = ticker_to_sector.get(tk, "")
    is_selected = (tk == ticker)
    fig_cmp.add_trace(go.Scatter(
        x=norm.index, y=norm[tk],
        name=f"{name}",
        legendgroup=sec, legendgrouptitle_text=sec,
        line=dict(
            color=SECTOR_COLORS.get(sec, "#7f8c8d"),
            width=3 if is_selected else 1.2,
        ),
        opacity=1.0 if is_selected else 0.55,
    ))
fig_cmp.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
fig_cmp.update_layout(
    height=480, hovermode="x unified",
    yaxis_title="누적 수익률 (%)",
    legend=dict(groupclick="toggleitem"),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig_cmp, use_container_width=True)
st.caption(f"굵은 선 = 현재 선택 종목({stock_name}) · 범례 클릭으로 종목 표시/숨김")

# 수익률 순위 테이블
last_ret = norm.iloc[-1].sort_values(ascending=False)
rank_df = pd.DataFrame({
    "종목": [ticker_to_name.get(tk, tk) for tk in last_ret.index],
    "섹터": [ticker_to_sector.get(tk, "") for tk in last_ret.index],
    f"{period_label} 수익률(%)": last_ret.round(2).values,
    "현재가(원)": [f"{multi[tk].dropna().iloc[-1]:,.0f}" for tk in last_ret.index],
})
with st.expander("🏆 수익률 순위표 보기", expanded=False):
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════
# 섹션 2. 개별 종목 상세 분석
# ══════════════════════════════
st.subheader(f"🔍 개별 분석 — {stock_name} ({ticker})")

df_raw = load_single(ticker, period)
if df_raw.empty:
    st.error(f"{stock_name} 데이터를 불러오지 못했습니다.")
    st.stop()

df = add_indicators(df_raw)

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change = last_close - prev_close
change_pct = (change / prev_close) * 100 if prev_close else 0.0
period_return = (last_close / float(df["Close"].iloc[0]) - 1) * 100
volatility = float(df["Return"].std() * np.sqrt(252)) if df["Return"].notna().sum() > 2 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("현재가", fmt_krw(last_close), f"{change:+,.0f}원 ({change_pct:+.2f}%)")
c2.metric(f"{period_label} 수익률", f"{period_return:+.2f}%")
c3.metric(f"{period_label} 최고가", fmt_krw(float(df["High"].max())))
c4.metric(f"{period_label} 최저가", fmt_krw(float(df["Low"].min())))
c5.metric("연환산 변동성", f"{volatility:.1f}%")

# 메인 차트
rows = 2 + int(show_rsi) + int(show_macd)
row_heights = [0.55, 0.15] + [0.15] * (rows - 2)
row_heights = [h / sum(row_heights) for h in row_heights]
subplot_titles = ["주가", "거래량"] + (["RSI (14)"] if show_rsi else []) + (["MACD"] if show_macd else [])

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=row_heights,
                    subplot_titles=subplot_titles)

if chart_type == "캔들스틱":
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=stock_name,
        increasing_line_color="#e74c3c",  # 상승 빨강 (한국식)
        decreasing_line_color="#3498db",  # 하락 파랑
    ), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="종가",
                             line=dict(color=SECTOR_COLORS.get(sector, "#2c3e50"), width=2)),
                  row=1, col=1)

if show_ma:
    for col, color in [("MA20", "#f39c12"), ("MA60", "#9b59b6"), ("MA120", "#95a5a6")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                 line=dict(color=color, width=1.2)), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB 상단",
                             line=dict(color="rgba(52,152,219,0.4)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB 하단",
                             line=dict(color="rgba(52,152,219,0.4)", width=1),
                             fill="tonexty", fillcolor="rgba(52,152,219,0.07)"), row=1, col=1)

vol_colors = np.where(df["Close"] >= df["Open"], "#e74c3c", "#3498db")
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량",
                     marker_color=vol_colors, opacity=0.6), row=2, col=1)

current_row = 3
if show_rsi:
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#8e44ad", width=1.5)), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="blue", opacity=0.5, row=current_row, col=1)
    current_row += 1

if show_macd:
    hist_colors = np.where(df["MACD_hist"] >= 0, "#e74c3c", "#3498db")
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist",
                         marker_color=hist_colors, opacity=0.6), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color="#2c3e50", width=1.3)), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
                             line=dict(color="#e67e22", width=1.3)), row=current_row, col=1)

fig.update_layout(
    height=300 + rows * 170,
    hovermode="x unified",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=60, b=10),
)
fig.update_yaxes(title_text="가격 (원)", tickformat=",", row=1, col=1)

st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────
# 분석 탭
# ──────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 수익률 분포", "📉 낙폭(MDD)", "🔗 섹터 내 상관관계", "🗂 원본 데이터"])

with tab1:
    ret = df["Return"].dropna()
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=ret, nbinsx=60,
                                    marker_color=SECTOR_COLORS.get(sector, "#2c3e50"), opacity=0.8))
    fig_hist.add_vline(x=float(ret.mean()), line_dash="dash", line_color="red",
                       annotation_text=f"평균 {ret.mean():.2f}%")
    fig_hist.update_layout(title=f"{stock_name} 일간 수익률 분포",
                           xaxis_title="일간 수익률 (%)", yaxis_title="빈도",
                           height=400, margin=dict(l=10, r=10, t=50, b=10))
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
    fig_dd.update_layout(title=f"{stock_name} 고점 대비 낙폭 (MDD: {drawdown.min():.1f}%)",
                         yaxis_title="낙폭 (%)", height=400,
                         margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_dd, use_container_width=True)

with tab3:
    sec_tickers = list(SECTORS[sector].values())
    sec_close = multi[[tk for tk in sec_tickers if tk in multi.columns]].dropna()
    if len(sec_close.columns) >= 2:
        corr = sec_close.pct_change().dropna().corr()
        labels = [ticker_to_name.get(tk, tk) for tk in corr.columns]
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=labels, y=labels,
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
        ))
        fig_corr.update_layout(title=f"{sector} 종목 간 일간 수익률 상관관계",
                               height=450, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("1에 가까울수록 같이 움직이고, 0에 가까울수록 독립적으로 움직입니다.")
    else:
        st.info("상관관계를 계산할 종목 데이터가 부족합니다.")

with tab4:
    show_df = df_raw.sort_index(ascending=False).copy()
    st.dataframe(show_df, use_container_width=True, height=400)
    csv = show_df.to_csv().encode("utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", csv,
                       file_name=f"{stock_name}_{period}.csv", mime="text/csv")

st.caption("⚠️ 본 대시보드는 정보 제공 목적이며 투자 권유가 아닙니다. 투자 판단과 책임은 본인에게 있습니다.")
