import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
import time
import math
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF Momentum Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh every hour
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60 * 60 * 1000, key="hourly_refresh")
except ImportError:
    pass

TOP_N      = 200   # cutoff per window for "strict overlap" metric
BATCH_SIZE = 200
DISPLAY_N  = 50    # rows to show

# ── Data pipeline (cached 1 hour) ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_pipeline():
    status = {}

    # ── Step 1: ETF Universe ────────────────────────────────────────────
    def fetch(url):
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        return r.text

    name_map = {}
    try:
        nasdaq_df = pd.read_csv(io.StringIO(fetch(
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        )), sep="|")
        nasdaq_etfs = nasdaq_df[
            nasdaq_df.get("ETF", pd.Series(dtype=str)).str.strip().str.upper() == "Y"
        ]["Symbol"].dropna().tolist() if "ETF" in nasdaq_df.columns else []

        if "Symbol" in nasdaq_df.columns and "Security Name" in nasdaq_df.columns:
            for sym, name in zip(nasdaq_df["Symbol"], nasdaq_df["Security Name"]):
                sym = str(sym).strip()
                if sym:
                    name_map[sym] = str(name).strip()

        other_df = pd.read_csv(io.StringIO(fetch(
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        )), sep="|")
        other_etfs = other_df[
            other_df.get("ETF", pd.Series(dtype=str)).str.strip().str.upper() == "Y"
        ]["ACT Symbol"].dropna().tolist() if "ETF" in other_df.columns else []

        if "ACT Symbol" in other_df.columns and "Security Name" in other_df.columns:
            for sym, name in zip(other_df["ACT Symbol"], other_df["Security Name"]):
                sym = str(sym).strip()
                if sym and sym not in name_map:
                    name_map[sym] = str(name).strip()

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

    # ── Step 2: Daily closes (batched) ──────────────────────────────────────────
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

    # ── Step 2b: Live intraday prices ─────────────────────────────────────────
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

    # ── Step 3: Returns ──────────────────────────────────────────────────────
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
                ret_1d = (c0 / float(s.iloc[-1]) - 1) * 100          if n >= 1  else None
                ret_5d = (c0 / float(s.iloc[max(0, n-5)]) - 1) * 100  if n >= 5  else None
                ret_1m = (c0 / float(s.iloc[max(0, n-21)]) - 1) * 100 if n >= 10 else None
            else:
                ret_1d = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100          if n >= 2  else None
                ret_5d = (float(s.iloc[-1]) / float(s.iloc[max(0, n-6)]) - 1) * 100  if n >= 6  else None
                ret_1m = (float(s.iloc[-1]) / float(s.iloc[max(0, n-22)]) - 1) * 100 if n >= 10 else None
        except Exception:
            continue
        records.append({"Ticker": t, "ret_1d": ret_1d, "ret_5d": ret_5d, "ret_1m": ret_1m})

    df = pd.DataFrame(records).dropna(subset=["ret_1d", "ret_5d", "ret_1m"])
    status["returned"] = len(df)

    # ── Step 4: Rank & composite score ──────────────────────────────────────────
    df["rank_1d"] = df["ret_1d"].rank(ascending=False, method="min")
    df["rank_5d"] = df["ret_5d"].rank(ascending=False, method="min")
    df["rank_1m"] = df["ret_1m"].rank(ascending=False, method="min")
    df["composite"] = (df["rank_1d"] + df["rank_5d"] + df["rank_1m"]) / 3

    # Top 50 by composite momentum score
    df_top = df.sort_values("composite").head(DISPLAY_N).reset_index(drop=True)

    # Strict overlap metric (top-200 in ALL three windows simultaneously)
    cutoff = min(TOP_N, len(df))
    t1 = set(df[df["rank_1d"] <= cutoff]["Ticker"])
    t5 = set(df[df["rank_5d"] <= cutoff]["Ticker"])
    tm = set(df[df["rank_1m"] <= cutoff]["Ticker"])
    overlap = t1 & t5 & tm

    # Flag which top-50 ETFs are also in the strict overlap
    df_top["In Overlap"] = df_top["Ticker"].isin(overlap)

    # ── Step 5: Metadata ─────────────────────────────────────────────────────
    # Names: Nasdaq CSV first, then Yahoo Finance chart API as fallback
    names          = {t: name_map.get(t, "") for t in df_top["Ticker"]}
    categories     = {}
    expense_ratios = {}

    for t in df_top["Ticker"].tolist():
        # Yahoo Finance chart API — reliable for name lookup
        if not names.get(t):
            try:
                url = f"https://query2.finance.yahoo.com/v8/finance/chart/{t}?interval=1d&range=1d"
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                names[t] = meta.get("longName") or meta.get("shortName") or ""
            except Exception:
                pass
        # Category & expense ratio from yfinance .info (best-effort)
        try:
            info = yf.Ticker(t).info
            categories[t]     = info.get("category") or info.get("sector") or ""
            expense_ratios[t] = (info.get("annualReportExpenseRatio")
                                 or info.get("totalExpenseRatio")
                                 or info.get("expenseRatio"))
        except Exception:
            pass

    df_top["Name"]          = df_top["Ticker"].map(names).fillna("")
    df_top["Category"]      = df_top["Ticker"].map(categories).fillna("")
    df_top["Expense Ratio"] = df_top["Ticker"].map(expense_ratios)

    status.update({
        "cutoff":        cutoff,
        "overlap_count": len(overlap),
        "top_1d":        len(t1),
        "top_5d":        len(t5),
        "top_1m":        len(tm),
        "run_time":      datetime.now().strftime("%b %d, %Y  %H:%M"),
        "live_prices":   len(live_prices),
    })

    return df_top, df, status

# ── UI ───────────────────────────────────────────────────────────────────────────
st.markdown("## 📈 ETF Momentum Dashboard")

col_hdr, col_refresh = st.columns([5, 1])
with col_refresh:
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

st.divider()

with st.spinner("Loading ETF data — this takes a few minutes on first load each hour…"):
    df_overlap, df_full, status = run_pipeline()

# ── Header metrics ─────────────────────────────────────────────────────────────
ml, m2, m3, m4 = st.columns(4)
ml.metric("Universe",        f"{status['universe']:,} ETFs")
m2.metric("Strong overlap",  status["overlap_count"],
          help=f"ETFs in top-{status['cutoff']} across all 3 windows simultaneously")
m3.metric("Live prices",     f"{status['live_prices']:,}")
m4.metric("Last updated",    status["run_time"])

st.subheader(f"Top {DISPLAY_N} Momentum ETFs")
st.caption(
    "Ranked by composite score — average rank across 1-day, 5-day, and 1-month returns. "
    f"Lower score = stronger momentum across all windows. "
    f"✅ marks the {status['overlap_count']} ETFs that also appear in the top-{status['cutoff']} of every window simultaneously."
)

# ── Format table ───────────────────────────────────────────────────────────────────
def fmt_pct(v):
    if pd.isna(v): return "—"
    return f"+{v:.2f}%" if v > 0 else f"{v:.2f}%"

display = df_overlap.copy()
display["1D Return"]   = display["ret_1d"].apply(fmt_pct)
display["5D Return"]   = display["ret_5d"].apply(fmt_pct)
display["1M Return"]   = display["ret_1m"].apply(fmt_pct)
display["1D Rank"]     = display["rank_1d"].apply(lambda x: int(x) if pd.notna(x) else "—")
display["5D Rank"]     = display["rank_5d"].apply(lambda x: int(x) if pd.notna(x) else "—")
display["1M Rank"]     = display["rank_1m"].apply(lambda x: int(x) if pd.notna(x) else "—")
display["Score"]       = display["composite"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
display["Exp Ratio"]   = display["Expense Ratio"].apply(
    lambda x: f"{x:.2%}" if pd.notna(x) else "—"
)
display["✅"]          = display["In Overlap"].apply(lambda x: "✅" if x else "")

show_cols = ["Ticker", "Name", "Category", "1D Return", "5D Return", "1M Return",
             "1D Rank", "5D Rank", "1M Rank", "Score", "Exp Ratio", "✅"]
display = display[[c for c in show_cols if c in display.columns]]

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker":    st.column_config.TextColumn("Ticker",    width=80),
        "Name":      st.column_config.TextColumn("Name",      width=230),
        "Category":  st.column_config.TextColumn("Category",  width=160),
        "1D Return": st.column_config.TextColumn("1D Return", width=95),
        "5D Return": st.column_config.TextColumn("5D Return", width=95),
        "1M Return": st.column_config.TextColumn("1M Return", width=95),
        "1D Rank":   st.column_config.TextColumn("1D Rank",   width=75),
        "5D Rank":   st.column_config.TextColumn("5D Rank",   width=75),
        "1M Rank":   st.column_config.TextColumn("1M Rank",   width=75),
        "Score":     st.column_config.TextColumn("Score",     width=65),
        "Exp Ratio": st.column_config.TextColumn("Exp Ratio", width=85),
        "✅":        st.column_config.TextColumn("Overlap",   width=65),
    }
)

# ── Run details ────────────────────────────────────────────────────────────────────
with st.expander("Run details"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**ETF universe:** {status['universe']:,}")
        st.markdown(f"**ETFs with all 3 returns:** {status['returned']:,}")
        st.markdown(f"**Overlap cutoff:** Top-{status['cutoff']} per window")
    with c2:
        st.markdown(f"**In top-{status['cutoff']} by 1D:** {status['top_1d']}")
        st.markdown(f"**In top-{status['cutoff']} by 5D:** {status['top_5d']}")
        st.markdown(f"**In top-{status['cutoff']} by 1M:** {status['top_1m']}")

st.divider()
st.caption("Data via yfinance · Refreshes automatically every hour · "
           "First load each hour takes a few minutes while data is fetched")
 
