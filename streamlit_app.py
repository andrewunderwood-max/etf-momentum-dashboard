import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
import time
import math
from datetime import datetime
from zoneinfo import ZoneInfo

# ── Page config ────────────────────────────────────────────────────────
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

TOP_N   = 200   # cutoff per window for "strict overlap" metric
BATCH_SIZE = 200
DISPLAY_N  = 50  # rows to show

# Keywords that unambiguously identify leveraged/inverse/structured ETFs.
# Intentionally NOT included: "bear", "bull", "short" (too many false positives
# with bond ETFs like "Short-Term" or "BulletShares").
EXCLUDE_KEYWORDS = {
    "1x", "1.5x", "2x", "3x", "4x",           # leverage multiples
    "ultra",                                      # ProShares Ultra, UltraPro, UltraShort
    "inverse",                                    # explicit inverse products
    "leveraged",                                  # explicit leveraged products
    "levered",                                    # alternative spelling
    "direxion",                                   # entire issuer is leveraged/inverse
    "microsectors",                               # entire issuer is leveraged
    "defined volatility",                         # structured outcome ETFs
}

# Minimum data-points required for each return window
MIN_N_1D = 2    # need at least 2 closes for 1-day return
MIN_N_5D_LIVE  = 5   # with live price: live / close[-5]
MIN_N_5D_HIST  = 6   # without live:    close[-1] / close[-6]
MIN_N_1M_LIVE  = 21  # FIX (was 10): with live price: live / close[-21]
MIN_N_1M_HIST  = 22  # FIX (was 10): without live:    close[-1] / close[-22]

# ── Data pipeline (cached 1 hour) ──────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_pipeline():
    status = {}
    errors = []   # FIX: track errors instead of silently swallowing them

    # ── Market hours check (US Eastern) ──────────────────────────────
    et_now = datetime.now(ZoneInfo("America/New_York"))
    _h, _m = et_now.hour, et_now.minute
    market_open = (
        et_now.weekday() < 5                       # Monday–Friday
        and (_h > 9 or (_h == 9 and _m >= 30))     # at or after 09:30
        and _h < 16                                 # before 16:00
    )

    # ── Step 1: ETF Universe ──────────────────────────────────────────
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
    except Exception as e:
        errors.append(f"Nasdaq CSV fetch failed: {e}")
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

    # ── Filter out leveraged/inverse ETFs ────────────────────────────
    def is_leveraged(ticker):
        name_lower = name_map.get(ticker, "").lower()
        return any(kw in name_lower for kw in EXCLUDE_KEYWORDS)

    n_before = len(etf_tickers)
    etf_tickers = [t for t in etf_tickers if not is_leveraged(t)]
    n_excluded = n_before - len(etf_tickers)

    status["universe"]     = len(etf_tickers)
    status["excluded_lev"] = n_excluded

    # ── Step 2: Daily closes (batched) ────────────────────────────────
    # FIX: Use period="2mo" to guarantee enough data for 1-month returns
    all_batches = []
    for i in range(0, len(etf_tickers), BATCH_SIZE):
        batch = etf_tickers[i:i+BATCH_SIZE]
        try:
            chunk = yf.download(
                " ".join(batch),
                period="2mo",           # FIX (was "1mo"): ensures ≥22 trading days
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if not chunk.empty:
                all_batches.append(chunk)
        except Exception as e:
            errors.append(f"Daily download batch {i//BATCH_SIZE} failed: {e}")
        time.sleep(2)

    raw = pd.concat(all_batches, axis=1) if all_batches else pd.DataFrame()

    # ── Step 2b: Live intraday prices (market hours only) ─────────────
    live_prices = {}
    if market_open:
        for i in range(0, len(etf_tickers), BATCH_SIZE):
            batch = etf_tickers[i:i+BATCH_SIZE]
            try:
                intraday = yf.download(
                    " ".join(batch),
                    period="1d",
                    interval="1m",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if not intraday.empty:
                    for t in batch:
                        try:
                            if len(batch) == 1:
                                s_live = intraday["Close"].dropna()
                            elif t in intraday.columns.get_level_values(0):
                                s_live = intraday[t]["Close"].dropna()
                            else:
                                s_live = (intraday["Close"][t].dropna()
                                          if "Close" in intraday else pd.Series(dtype=float))
                            if len(s_live) > 0:
                                live_prices[t] = float(s_live.iloc[-1])
                        except Exception:
                            pass
            except Exception as e:
                errors.append(f"Intraday download batch {i//BATCH_SIZE} failed: {e}")
            time.sleep(1)

    # ── Step 3: Returns ───────────────────────────────────────────────
    closes = {}
    for t in etf_tickers:
        try:
            if t in raw.columns.get_level_values(0):
                s = raw[t]["Close"].dropna()
            else:
                s = (raw["Close"][t].dropna()
                     if "Close" in raw else pd.Series(dtype=float))
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
                # Live price path — c0 is today's latest tick
                ret_1d = (c0 / float(s.iloc[-1]) - 1) * 100  if n >= MIN_N_1D      else None
                ret_5d = (c0 / float(s.iloc[-5]) - 1) * 100  if n >= MIN_N_5D_LIVE else None
                ret_1m = (c0 / float(s.iloc[-21]) - 1) * 100 if n >= MIN_N_1M_LIVE else None
                # FIX: use negative indexing directly (clearer) and proper guards
            else:
                # Historical-only path
                ret_1d = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1) * 100   if n >= MIN_N_1D      else None
                ret_5d = (float(s.iloc[-1]) / float(s.iloc[-6]) - 1) * 100   if n >= MIN_N_5D_HIST else None
                ret_1m = (float(s.iloc[-1]) / float(s.iloc[-22]) - 1) * 100  if n >= MIN_N_1M_HIST else None
                # FIX: 1M guard changed from n>=10 to n>=22
        except Exception:
            continue

        # FIX: skip append if any return is None (avoids wasteful dropna later)
        if ret_1d is None or ret_5d is None or ret_1m is None:
            continue

        records.append({"Ticker": t, "ret_1d": ret_1d, "ret_5d": ret_5d, "ret_1m": ret_1m})

    df = pd.DataFrame(records)
    status["returned"] = len(df)

    if df.empty:
        status.update({
            "cutoff": 0, "overlap_count": 0,
            "top_1d": 0, "top_5d": 0, "top_1m": 0,
            "run_time": et_now.strftime("%b %d, %Y %H:%M ET"),
            "market_open": market_open, "live_prices": len(live_prices),
            "errors": errors,
        })
        return pd.DataFrame(), df, status

    # ── Step 4: Rank & composite score ────────────────────────────────
    df["rank_1d"] = df["ret_1d"].rank(ascending=False, method="min")
    df["rank_5d"] = df["ret_5d"].rank(ascending=False, method="min")
    df["rank_1m"] = df["ret_1m"].rank(ascending=False, method="min")
    df["composite"] = (df["rank_1d"] + df["rank_5d"] + df["rank_1m"]) / 3

    # Strict overlap metric (top-N in ALL three windows simultaneously)
    cutoff = min(TOP_N, len(df))
    t1 = set(df[df["rank_1d"] <= cutoff]["Ticker"])
    t5 = set(df[df["rank_5d"] <= cutoff]["Ticker"])
    tm = set(df[df["rank_1m"] <= cutoff]["Ticker"])
    overlap = t1 & t5 & tm

    # ── Step 5: Names (Nasdaq CSV → Yahoo Finance chart API fallback) ─
    # FIX: Resolve names for a wider set BEFORE trimming to top 50,
    #       then apply the second-pass leveraged filter, THEN take top 50.
    df_ranked = df.sort_values("composite").reset_index(drop=True)

    # We only need names for the top candidates (take extra to account for filtering)
    CANDIDATE_N = DISPLAY_N + 30   # FIX: take extra rows to survive second-pass filter
    df_candidates = df_ranked.head(CANDIDATE_N).copy()

    names = {t: name_map.get(t, "") for t in df_candidates["Ticker"]}
    for t in df_candidates["Ticker"].tolist():
        if not names.get(t):
            try:
                url = f"https://query2.finance.yahoo.com/v8/finance/chart/{t}?interval=1d&range=1d"
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                names[t] = meta.get("longName") or meta.get("shortName") or ""
            except Exception:
                pass

    df_candidates["Name"] = df_candidates["Ticker"].map(names).fillna("")

    # ── Second-pass leveraged filter ──────────────────────────────────
    # FIX: Apply BEFORE .head(DISPLAY_N) so the final table always has
    #       up to DISPLAY_N rows and doesn't miss legitimate ETFs.
    def name_is_excluded(name):
        n = name.lower()
        return any(kw in n for kw in EXCLUDE_KEYWORDS)

    before2 = len(df_candidates)
    df_candidates = df_candidates[~df_candidates["Name"].apply(name_is_excluded)].reset_index(drop=True)
    status["excluded_lev"] += before2 - len(df_candidates)

    # Now take the final top DISPLAY_N
    df_top = df_candidates.head(DISPLAY_N).copy()

    # Flag which top ETFs are also in the strict overlap
    df_top["In Overlap"] = df_top["Ticker"].isin(overlap)

    status.update({
        "cutoff":        cutoff,
        "overlap_count": len(overlap),
        "top_1d":        len(t1),
        "top_5d":        len(t5),
        "top_1m":        len(tm),
        "run_time":      et_now.strftime("%b %d, %Y %H:%M ET"),
        "market_open":   market_open,
        "live_prices":   len(live_prices),
        "errors":        errors,
    })

    return df_top, df, status


# ── UI ─────────────────────────────────────────────────────────────────
st.markdown("## 📈 ETF Momentum Dashboard")

col_hdr, col_refresh = st.columns([5, 1])
with col_refresh:
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

st.divider()

with st.spinner("Loading ETF data — this takes a few minutes on first load each hour…"):
    df_overlap, df_full, status = run_pipeline()

# ── Surface errors if any ─────────────────────────────────────────────
# FIX: Show pipeline errors so users know if data is incomplete
if status.get("errors"):
    with st.expander(f"⚠️ {len(status['errors'])} warning(s) during data load", expanded=False):
        for err in status["errors"]:
            st.warning(err)

# ── Header metrics ────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Universe", f"{status['universe']:,} ETFs")
m2.metric("Strong overlap", status["overlap_count"],
          help=f"ETFs in top‑{status['cutoff']} across all 3 windows simultaneously")
m3.metric("Prices", "Live ✓" if status["market_open"] else "Prior close")
m4.metric("Last updated", status["run_time"])

if df_overlap.empty:
    st.error("No ETF data was returned. Check the warnings above or try refreshing.")
    st.stop()

st.subheader(f"Top {DISPLAY_N} Momentum ETFs")
st.caption(
    "Ranked by composite score — average rank across 1‑day, 5‑day, and 1‑month returns. "
    f"Lower score = stronger momentum across all windows. "
    f"✅ marks the {status['overlap_count']} ETFs that also appear in the "
    f"top‑{status['cutoff']} of every window simultaneously."
)

# ── Format table ──────────────────────────────────────────────────────
display = df_overlap[["Ticker", "Name", "ret_1d", "ret_5d", "ret_1m",
                       "rank_1d", "rank_5d", "rank_1m", "composite",
                       "In Overlap"]].copy()

display["rank_1d"] = display["rank_1d"].apply(lambda x: int(x) if pd.notna(x) else None)
display["rank_5d"] = display["rank_5d"].apply(lambda x: int(x) if pd.notna(x) else None)
display["rank_1m"] = display["rank_1m"].apply(lambda x: int(x) if pd.notna(x) else None)
display["✅"] = display["In Overlap"].apply(lambda x: "✅" if x else "")
display = display[["Ticker", "Name", "ret_1d", "ret_5d", "ret_1m",
                    "rank_1d", "rank_5d", "rank_1m", "composite", "✅"]]

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker":    st.column_config.TextColumn("Ticker",     width=80),
        "Name":      st.column_config.TextColumn("Name",       width=260),
        "ret_1d":    st.column_config.NumberColumn("1D Ret %",  format="%.2f%%", width=95),
        "ret_5d":    st.column_config.NumberColumn("5D Ret %",  format="%.2f%%", width=95),
        "ret_1m":    st.column_config.NumberColumn("1M Ret %",  format="%.2f%%", width=95),
        "rank_1d":   st.column_config.NumberColumn("1D Rank",   format="%d",     width=80),
        "rank_5d":   st.column_config.NumberColumn("5D Rank",   format="%d",     width=80),
        "rank_1m":   st.column_config.NumberColumn("1M Rank",   format="%d",     width=80),
        "composite": st.column_config.NumberColumn("Score",     format="%.1f",   width=70),
        "✅":        st.column_config.TextColumn("Overlap",     width=65),
    }
)

# ── Run details ───────────────────────────────────────────────────────
with st.expander("Run details"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**ETF universe:** {status['universe']:,}")
        st.markdown(f"**Leveraged/inverse excluded:** {status['excluded_lev']:,}")
        st.markdown(f"**ETFs with all 3 returns:** {status['returned']:,}")
        st.markdown(f"**Overlap cutoff:** Top‑{status['cutoff']} per window")
    with c2:
        st.markdown(f"**In top‑{status['cutoff']} by 1D:** {status['top_1d']}")
        st.markdown(f"**In top‑{status['cutoff']} by 5D:** {status['top_5d']}")
        st.markdown(f"**In top‑{status['cutoff']} by 1M:** {status['top_1m']}")

st.divider()
st.caption("Data via yfinance · Returns based on prior close outside market hours (09:30–16:00 ET) · "
           "Refreshes automatically every hour · First load each hour takes a few minutes")

