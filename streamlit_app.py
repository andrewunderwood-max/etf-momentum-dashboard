import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
import time
import math
from datetime import datetime

st.set_page_config(
    page_title="ETF Momentum Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60 * 60 * 1000, key="hourly_refresh")
except ImportError:
    pass

TOP_N       = 200
MIN_OVERLAP = 50
EXPAND_N    = 250
BATCH_SIZE  = 200

@st.cache_data(ttl=3600, show_spinner=False)
def run_pipeline():
    status = {}

    def fetch(url):
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        return r.text

    try:
        nasdaq_df = pd.read_csv(io.StringIO(fetch(
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        )), sep="|")
        nasdaq_etfs = nasdaq_df[
            nasdaq_df.get("ETF", pd.Series(dtype=str)).str.strip().str.upper() == "Y"
        ]["Symbol"].dropna().tolist() if "ETF" in nasdaq_df.columns else []

        other_df = pd.read_csv(io.StringIO(fetch(
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        )), sep="|")
        other_etfs = other_df[
            other_df.get("ETF", pd.Series(dtype=str)).str.strip().str.upper() == "Y"
        ]["ACT Symbol"].dropna().tolist() if "ETF" in other_df.columns else []

        raw_tickers = list(set(nasdaq_etfs + other_etfs))
        etf_tickers = sorted({
            t for t in raw_tickers
            if isinstance(t, str) and t.strip() and len(t.strip()) <= 5
            and t.strip().isalpha()
            and t.strip().upper() not in ("SYMBOL", "FILE", "CREATION")
        })
    except Exception:
        etf_tickers = []

    SEED = [
        "SPY","IVV","VOO","VTI","QQQ","IWM","EFA","EEM","AGG","BND",
        "GLD","SLV","TLT","HYG","LQD","VNQ","XLE","XLF","XLK","XLV",
        "XLY","XLI","XLB","XLC","XLU","XLP","ARKK","VCIT","VCSH","VXUS",
        "VWO","VEA","IEMG","IJH","IJR","IWF","IWD","USMV","QUAL","MTUM",
        "VIG","VYM","DVY","SDY","SCHD","NOBL","DGRO","PFF","JNK","EMB",
        "IAU","SGOL","USO","UNG","DBA","XBI","IBB","XHB","XRT","KRE",
        "KBE","ITA","CIBR","CLOU","HACK","ROBO","BOTZ","MCHI","KWEB",
        "FXI","ASHR","EWJ","EWZ","EWC","EWA","EWG","EWU","EWY","EWH",
    ]
    if len(etf_tickers) < 100:
        etf_tickers = sorted(set(etf_tickers + SEED))

    status["universe"] = len(etf_tickers)

    all_batches = []
    for i in range(0, len(etf_tickers), BATCH_SIZE):
        batch = etf_tickers[i:i+BATCH_SIZE]
        try:
            chunk = yf.download(
                " ".join(batch), period="1mo", interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=False,
            )
            if not chunk.empty:
                all_batches.append(chunk)
        except Exception:
            pass
        time.sleep(2)

    raw = pd.concat(all_batches, axis=1) if all_batches else pd.DataFrame()

    live_prices = {}
    for i in range(0, len(etf_tickers), BATCH_SIZE):
        batch = etf_tickers[i:i+BATCH_SIZE]
        try:
            intraday = yf.download(
                " ".join(batch), period="1d", interval="1m",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=False,
            )
            if not intraday.empty:
                for t in batch:
                    try:
                        if len(batch) == 1:
                            s_live = intraday["Close"].dropna()
                        elif t in intraday.columns.get_level_values(0):
                            s_live = intraday[t]["Close"].dropna()
                        else:
                            s_live = intraday["Close"][t].dropna() if "Close" in intraday else pd.Series(dtype=float)
                        if len(s_live) > 0:
                            live_prices[t] = float(s_live.iloc[-1])
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(1)

    closes = {}
    for t in etf_tickers:
        try:
            if t in raw.columns.get_level_values(0):
                s = raw[t]["Close"].dropna()
            else:
                s = raw["Close"][t].dropna() if "Close" in raw else pd.Series(dtype=float)
            if len(s) >= 2:
                closes[t] = s
        except Exception:
            pass

    records = []
    for t, s in closes.items():
        s = s.sort_index()
        n = len(s)
        try:
            c0 = live_prices.get(t)
            if c0 is not None:
                ret_1d = (c0 / float(s.iloc[-1]) - 1) * 100         if n >= 1  else None
                ret_5d = (c0 / float(s.iloc[max(0, n-5)]) - 1) * 100  if n >= 5  else None
                ret_1m = (c0 / float(s.iloc[max(0, n-21)]) - 1) * 100 if n >= 10 else None
            else:
                ret_1d = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100         if n >= 2  else None
                ret_5d = (float(s.iloc[-1]) / float(s.iloc[max(0, n-6)]) - 1) * 100  if n >= 6  else None
                ret_1m = (float(s.iloc[-1]) / float(s.iloc[max(0, n-22)]) - 1) * 100 if n >= 10 else None
        except Exception:
            continue
        records.append({"Ticker": t, "ret_1d": ret_1d, "ret_5d": ret_5d, "ret_1m": ret_1m})

    df = pd.DataFrame(records).dropna(subset=["ret_1d", "ret_5d", "ret_1m"])
    status["returned"] = len(df)

    df["rank_1d"] = df["ret_1d"].rank(ascending=False, method="min")
    df["rank_5d"] = df["ret_5d"].rank(ascending=False, method="min")
    df["rank_1m"] = df["ret_1m"].rank(ascending=False, method="min")

    n = len(df)
    if n >= TOP_N:
        cutoff = TOP_N
    else:
        cutoff = max(3, math.ceil(n * 0.70))

    def get_overlap(c):
        t1 = set(df[df["rank_1d"] <= c]["Ticker"])
        t5 = set(df[df["rank_5d"] <= c]["Ticker"])
        tm = set(df[df["rank_1m"] <= c]["Ticker"])
        return t1 & t5 & tm, t1, t5, tm

    overlap, top_1d, top_5d, top_1m = get_overlap(cutoff)
    if n >= TOP_N and len(overlap) < MIN_OVERLAP:
        cutoff = EXPAND_N
        overlap, top_1d, top_5d, top_1m = get_overlap(cutoff)

    df_ov = df[df["Ticker"].isin(overlap)].copy()
    df_ov["composite"] = (df_ov["rank_1d"] + df_ov["rank_5d"] + df_ov["rank_1m"]) / 3
    df_ov = df_ov.sort_values("composite").reset_index(drop=True)

    names, categories, expense_ratios = {}, {}, {}
    for t in df_ov["Ticker"].tolist():
        try:
            info = yf.Ticker(t).info
            names[t]          = info.get("longName") or info.get("shortName", "")
            categories[t]     = info.get("category") or info.get("quoteType", "")
            expense_ratios[t] = info.get("annualReportExpenseRatio") or info.get("totalExpenseRatio")
        except Exception:
            pass

    df_ov["Name"]          = df_ov["Ticker"].map(names).fillna("")
    df_ov["Category"]      = df_ov["Ticker"].map(categories).fillna("")
    df_ov["Expense Ratio"] = df_ov["Ticker"].map(expense_ratios)

    status.update({
        "cutoff": cutoff,
        "overlap_count": len(overlap),
        "top_1d": len(top_1d),
        "top_5d": len(top_5d),
        "top_1m": len(top_1m),
        "run_time": datetime.now().strftime("%b %d, %Y  %H:%M"),
        "live_prices": len(live_prices),
    })

    return df_ov, df, status

st.markdown("## 📈 ETF Momentum Dashboard")
col_hdr, col_refresh = st.columns([5, 1])
with col_refresh:
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

st.divider()

with st.spinner("Loading ETF data — this takes a few minutes on first load each hour..."):
    df_overlap, df_full, status = run_pipeline()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Universe",     f"{status['universe']:,} ETFs")
m2.metric("Overlap ETFs", status["overlap_count"])
m3.metric("Live prices",  f"{status['live_prices']:,}")
m4.metric("Last updated", status["run_time"])

st.subheader(f"Top Momentum ETFs - Top-{status['cutoff']} overlap")
st.caption("Ranked in the top tier across 1-day, 5-day, and 1-month returns simultaneously. Sorted by composite score -- lower = stronger momentum across all windows.")

def fmt_pct(v):
    if pd.isna(v): return "--"
    return f"+{v:.2f}%" if v > 0 else f"{v:.2f}%"

display = df_overlap.copy()
display["1D Return"] = display["ret_1d"].apply(fmt_pct)
display["5D Return"] = display["ret_5d"].apply(fmt_pct)
display["1M Return"] = display["ret_1m"].apply(fmt_pct)
display["1D Rank"]   = display["rank_1d"].apply(lambda x: int(x) if pd.notna(x) else "--")
display["5D Rank"]   = display["rank_5d"].apply(lambda x: int(x) if pd.notna(x) else "--")
display["1M Rank"]   = display["rank_1m"].apply(lambda x: int(x) if pd.notna(x) else "--")
display["Score"]     = display["composite"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "--")
display["Exp Ratio"] = display["Expense Ratio"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "--")

show_cols = ["Ticker", "Name", "Category", "1D Return", "5D Return", "1M Return", "1D Rank", "5D Rank", "1M Rank", "Score", "Exp Ratio"]
display = display[[c for c in show_cols if c in display.columns]]

st.dataframe(display, use_container_width=True, hide_index=True,
    column_config={
        "Ticker":    st.column_config.TextColumn("Ticker",    width=80),
        "Name":      st.column_config.TextColumn("Name",      width=250),
        "Category":  st.column_config.TextColumn("Category",  width=180),
        "1D Return": st.column_config.TextColumn("1D Return", width=100),
        "5D Return": st.column_config.TextColumn("5D Return", width=100),
        "1M Return": st.column_config.TextColumn("1M Return", width=100),
        "1D Rank":   st.column_config.TextColumn("1D Rank",   width=80),
        "5D Rank":   st.column_config.TextColumn("5D Rank",   width=80),
        "1M Rank":   st.column_config.TextColumn("1M Rank",   width=80),
        "Score":     st.column_config.TextColumn("Score",     width=70),
        "Exp Ratio": st.column_config.TextColumn("Exp Ratio", width=90),
    })

with st.expander("Run details"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**ETF universe:** {status['universe']:,}")
        st.markdown(f"**ETFs with all 3 returns:** {status['returned']:,}")
        st.markdown(f"**Cutoff used:** Top-{status['cutoff']}")
    with c2:
        st.markdown(f"**In top-{status['cutoff']} by 1D:** {status['top_1d']}")
        st.markdown(f"**In top-{status['cutoff']} by 5D:** {status['top_5d']}")
        st.markdown(f"**In top-{status['cutoff']} by 1M:** {status['top_1m']}")

st.divider()
st.caption("Data via yfinance - Refreshes automatically every hour - First load each hour takes a few minutes while data is fetched")
 
