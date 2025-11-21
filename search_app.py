# search_app.py
"""
Streamlit app for GDELT sentiment share + volume with optional source-country filtering
(using gdeltdoc when user selects source country(s); otherwise falls back to raw GDELT requests).

Requirements:
    pip install streamlit pandas requests matplotlib
Optional for source-country filtering:
    pip install gdeltdoc
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from io import BytesIO
import io

# Try to import gdeltdoc (optional). If unavailable, we will fall back.
try:
    from gdeltdoc import GdeltDoc, Filters
    _GDELTDOC_AVAILABLE = True
except Exception:
    _GDELTDOC_AVAILABLE = False

# -------------------------
# Config / Defaults
# -------------------------
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# compact sidebar CSS (optional, reduces vertical spacing)
st.set_page_config(page_title="GDELT Sentiment Share Explorer", layout="wide")
st.markdown(
    """
    <style>
    /* Reduce space between widgets inside sidebar */
    section[data-testid="stSidebar"] .block-container { padding: 0.6rem 1rem; }
    div[data-testid="stVerticalBlock"] > div { padding-top: 0.12rem; padding-bottom: 0.12rem; }
    h2, h3 { margin-bottom: 0.18rem; margin-top: 0.18rem; }
    div.stCheckbox { margin-bottom: 0.12rem; }
    div.stButton > button:first-child { height: 2.6em; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Utility helpers
# -------------------------
def normalize_query(q: str) -> str:
    if not q:
        return q
    q_str = q.strip()
    if " OR " in q_str.upper():
        if q_str.startswith("(") and q_str.endswith(")"):
            return q_str
        return "(" + q_str + ")"
    return q_str

def request_gdelt(params, timeout=30):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(GDELT_URL, params=params, headers=headers, timeout=timeout)
    text = (r.text or "").strip()
    if not text:
        raise RuntimeError("Empty response from GDELT (no content). Check your network or parameters.")
    try:
        return r.json()
    except ValueError:
        snippet = text[:800] + ("..." if len(text) > 800 else "")
        hint = ""
        lowered = text.lower()
        if "must be surrounded by" in lowered or "queries containing or" in lowered or "surrounded by" in lowered:
            hint = "Hint: GDELT requires queries with OR to be wrapped in parentheses, e.g. (AI OR ChatGPT).\n"
        raise RuntimeError(
            "GDELT returned non-JSON response. API message (truncated):\n\n"
            f"{snippet}\n\n{hint}"
        )

# -------------------------
# Raw GDELT fetch helpers (fallback)
# -------------------------
def fetch_timeline_tone_raw(query, start_dt, end_dt):
    params = {
        "query": query,
        "format": "json",
        "mode": "TimelineTone",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S")
    }
    j = request_gdelt(params)
    timeline = j.get("timeline") or []
    if not timeline:
        return pd.DataFrame()
    data = timeline[0].get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df.rename(columns={"value": "tone_value", "count": "tone_sample_count"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def fetch_timeline_vol_raw(query, start_dt, end_dt):
    params = {
        "query": query,
        "format": "json",
        "mode": "TimelineVol",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S")
    }
    r = requests.get(GDELT_URL, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    text = r.text or ""
    try:
        j = r.json()
    except Exception as e:
        raise RuntimeError(f"Couldn't parse JSON from TimelineVol: {e}\nResponse snippet:\n{text[:800]}")
    timeline = j.get("timeline") or []
    if not timeline:
        return pd.DataFrame()
    data = timeline[0].get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# -------------------------
# gdeltdoc-based fetch (preferred when source_countries are provided)
# -------------------------
def fetch_with_gdeltdoc(raw_query, start_dt, end_dt, country_list):
    """
    Use gdeltdoc Filters to retrieve timeline tone and timeline vol filtered by source country list.
    Returns merged DataFrame with columns: date (datetime), tone_value, articles
    """
    if not _GDELTDOC_AVAILABLE:
        raise RuntimeError("gdeltdoc not installed. Add 'gdeltdoc' to requirements.txt to use source-country filtering.")

    # Build Filters. Filters expects dates as YYYY-MM-DD strings typically.
    # The Filters class in gdeltdoc accepts 'keyword', 'start_date', 'end_date', 'country'.
    filt = Filters(
        keyword = raw_query,
        start_date = start_dt.strftime("%Y-%m-%d"),
        end_date   = end_dt.strftime("%Y-%m-%d"),
        country    = country_list
    )

    gd = GdeltDoc()
    # Timeline calls: attempt timelinetone & timelinevolraw (client names may vary by package version)
    tone_df = pd.DataFrame()
    vol_df = pd.DataFrame()
    try:
        tone_df = gd.timeline_search("timelinetone", filt)
    except Exception:
        # try alternate name
        try:
            tone_df = gd.timeline_search("TimelineTone", filt)
        except Exception as e:
            raise RuntimeError(f"gdeltdoc timelinetone call failed: {e}")

    try:
        # try raw volume timeline variant first, then fallback
        vol_df = gd.timeline_search("timelinevolraw", filt)
    except Exception:
        try:
            vol_df = gd.timeline_search("timelinevol", filt)
        except Exception:
            # fallback to empty vol_df
            vol_df = pd.DataFrame()

    # Normalize tone_df
    if not tone_df.empty:
        # typical columns: 'date', 'value', 'count' — convert to our canonical names
        tone_df = tone_df.copy()
        if "value" in tone_df.columns and "tone_value" not in tone_df.columns:
            tone_df = tone_df.rename(columns={"value": "tone_value"})
        if "count" in tone_df.columns and "tone_sample_count" not in tone_df.columns:
            tone_df = tone_df.rename(columns={"count": "tone_sample_count"})
        if "date" in tone_df.columns:
            tone_df["date"] = pd.to_datetime(tone_df["date"], errors="coerce")
    # Normalize vol_df
    if not vol_df.empty:
        vol_df = vol_df.copy()
        # look for candidate columns similar to raw flow
        possible_cols = ["value", "articles", "count", "vol_sample_count", "value_vol", "count_vol"]
        found = None
        for c in possible_cols:
            if c in vol_df.columns:
                found = c
                break
        if found:
            vol_df = vol_df.rename(columns={found: "articles"})
            vol_df["articles"] = pd.to_numeric(vol_df["articles"].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)
        if "date" in vol_df.columns:
            vol_df["date"] = pd.to_datetime(vol_df["date"], errors="coerce")

    # Merge tone_df and vol_df
    merged = pd.merge(tone_df, vol_df, on="date", how="outer", suffixes=("_tone", "_vol")) if (not tone_df.empty or not vol_df.empty) else pd.DataFrame()
    if merged.empty:
        return merged
    if "articles" not in merged.columns:
        merged["articles"] = 0.0
    merged["articles"] = pd.to_numeric(merged["articles"], errors="coerce").fillna(0.0)
    # ensure tone_value exists
    if "tone_value" not in merged.columns:
        merged["tone_value"] = pd.to_numeric(merged.get("tone_value", pd.Series(dtype=float)), errors="coerce")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged

# -------------------------
# Fallback wrapper: choose gdeltdoc if countries provided and package installed, else raw
# -------------------------
def fetch_and_merge_choose(raw_query, start_dt, end_dt, source_countries=None):
    """
    If source_countries provided (non-empty list) and gdeltdoc is available, use it.
    Otherwise use the raw GDELT HTTP fetching functions.
    Returns merged DataFrame with date, tone_value, articles (floats).
    """
    if source_countries and len(source_countries) > 0:
        if not _GDELTDOC_AVAILABLE:
            raise RuntimeError("You selected source_country filtering but 'gdeltdoc' is not installed in this environment. Add 'gdeltdoc' to requirements.txt.")
        # use gdeltdoc
        merged = fetch_with_gdeltdoc(raw_query, start_dt, end_dt, source_countries)
        # If gdeltdoc returned empty DataFrame, fall back to raw as a safety? We'll return whatever we got.
        return merged
    else:
        # normalize query for raw calls
        query = normalize_query(raw_query)
        tone_df = fetch_timeline_tone_raw(query, start_dt, end_dt)
        vol_df = fetch_timeline_vol_raw(query, start_dt, end_dt)
        if tone_df.empty and vol_df.empty:
            return pd.DataFrame()
        # Normalize date/time and merge (same logic used throughout)
        for ddf in (tone_df, vol_df):
            if not ddf.empty:
                ddf["date"] = pd.to_datetime(ddf["date"], errors="coerce")
                try:
                    if pd.api.types.is_datetime64tz_dtype(ddf["date"].dtype):
                        ddf["date"] = ddf["date"].dt.tz_convert(None)
                except Exception:
                    pass
                ddf["date"] = ddf["date"].dt.round("1s")
        # detect volume column and rename
        possible_cols = ["value", "articles", "count", "vol_sample_count", "value_vol", "count_vol"]
        found = None
        if not vol_df.empty:
            for c in possible_cols:
                if c in vol_df.columns:
                    found = c
                    break
        if found:
            vol_df = vol_df.rename(columns={found: "articles"})
            vol_df["articles"] = pd.to_numeric(vol_df["articles"].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)
        else:
            if not vol_df.empty:
                vol_df["articles"] = 0.0
        merged = pd.merge(tone_df, vol_df, on="date", how="outer", suffixes=("_tone", "_vol"))
        if "articles" not in merged.columns:
            merged["articles"] = 0.0
        merged["articles"] = pd.to_numeric(merged["articles"], errors="coerce").fillna(0.0)
        merged = merged.sort_values("date").reset_index(drop=True)
        return merged

# -------------------------
# Aggregation and filler (same as before)
# -------------------------
def aggregate_sentiment_share(df, freq="monthly"):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().dropna(subset=["date"]).reset_index(drop=True)
    if "tone_value" not in df.columns:
        df["tone_value"] = pd.NA
    if freq == "monthly":
        df["period"] = df["date"].dt.to_period("M")
        df["period_str"] = df["period"].astype(str)
    else:
        week_start = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="d")
        df["period_str"] = week_start.dt.strftime("%Y-%m-%d")
    threshold = 0.4
    df["positive"] = df["tone_value"].apply(lambda x: 1 if pd.notna(x) and x >= threshold else 0)
    df["negative"] = df["tone_value"].apply(lambda x: 1 if pd.notna(x) and x < threshold else 0)
    df["articles"] = pd.to_numeric(df.get("articles", 0), errors="coerce").fillna(0).astype(float)
    basic = df.groupby("period_str", sort=True).agg({
        "positive": "sum",
        "negative": "sum",
        "articles": "sum"
    }).reset_index()

    def weighted_avg(subdf):
        mask = subdf["tone_value"].notna()
        if mask.any() and subdf.loc[mask, "articles"].sum() > 0:
            w = subdf.loc[mask, "articles"].astype(float)
            vals = subdf.loc[mask, "tone_value"].astype(float)
            return (vals * w).sum() / w.sum()
        vals = subdf["tone_value"].dropna()
        return vals.mean() if len(vals) else float("nan")

    tone_agg = (
        df.groupby("period_str")[["tone_value", "articles"]]
          .apply(lambda g: weighted_avg(g))
          .reset_index(name="avg_tone")
    )

    result = basic.merge(tone_agg, on="period_str", how="left")
    result["total_bins"] = result["positive"] + result["negative"]
    result["positive_share_%"] = result.apply(
        lambda r: (r["positive"] / r["total_bins"]) * 100 if r["total_bins"] > 0 else 0, axis=1
    )
    result["negative_share_%"] = result.apply(
        lambda r: (r["negative"] / r["total_bins"]) * 100 if r["total_bins"] > 0 else 0, axis=1
    )
    if freq == "monthly":
        result["period_dt"] = pd.to_datetime(result["period_str"] + "-01", errors="coerce")
    else:
        result["period_dt"] = pd.to_datetime(result["period_str"], errors="coerce")
    result = result.sort_values("period_dt").drop(columns=["period_dt"]).reset_index(drop=True)
    return result

def fill_missing_periods(agg_df, start_dt, end_dt, freq="monthly"):
    if agg_df is None or agg_df.empty:
        if freq == "monthly":
            all_periods = pd.date_range(start=start_dt.replace(day=1), end=end_dt, freq="MS")
            period_dt = all_periods
            period_strs = [d.strftime("%Y-%m") for d in period_dt]
        else:
            start_monday = (start_dt - pd.to_timedelta(start_dt.weekday(), unit="d")).date()
            period_dt = pd.date_range(start=pd.to_datetime(start_monday), end=end_dt, freq="W-MON")
            period_strs = [d.strftime("%Y-%m-%d") for d in period_dt]

        out = pd.DataFrame({
            "period_str": period_strs,
            "positive": 0,
            "negative": 0,
            "articles": 0.0,
            "avg_tone": [float("nan")] * len(period_strs),
        })
        out["total_bins"] = out["positive"] + out["negative"]
        out["positive_share_%"] = 0
        out["negative_share_%"] = 0
        return out

    df = agg_df.copy()
    for c in ["positive", "negative", "articles", "avg_tone"]:
        if c not in df.columns:
            df[c] = 0 if c in ("positive", "negative") else (0.0 if c == "articles" else float("nan"))

    if freq == "monthly":
        df["period_dt"] = pd.to_datetime(df["period_str"] + "-01", errors="coerce")
        full_index = pd.date_range(start=start_dt.replace(day=1), end=end_dt, freq="MS")
    else:
        df["period_dt"] = pd.to_datetime(df["period_str"], errors="coerce")
        start_monday = (start_dt - pd.to_timedelta(start_dt.weekday(), unit="d")).date()
        full_index = pd.date_range(start=pd.to_datetime(start_monday), end=end_dt, freq="W-MON")

    df = df.set_index("period_dt")
    df.index = pd.to_datetime(df.index)
    reindexed = df.reindex(full_index)

    for col in ["positive", "negative", "articles"]:
        if col in reindexed.columns:
            reindexed[col] = reindexed[col].fillna(0)

    reindexed["total_bins"] = reindexed.get("total_bins", reindexed.get("positive", 0) + reindexed.get("negative", 0))
    reindexed["total_bins"] = reindexed["total_bins"].fillna(reindexed.get("positive", 0) + reindexed.get("negative", 0))

    if "avg_tone" not in reindexed.columns:
        reindexed["avg_tone"] = float("nan")
    reindexed["avg_tone"] = pd.to_numeric(reindexed["avg_tone"], errors="coerce")
    try:
        reindexed["avg_tone"] = reindexed["avg_tone"].interpolate(method="time", limit_direction="both")
    except Exception:
        reindexed["avg_tone"] = reindexed["avg_tone"].interpolate(method="linear", limit_direction="both")

    reindexed["total_bins"] = reindexed["positive"].astype(float) + reindexed["negative"].astype(float)
    reindexed["positive_share_%"] = reindexed.apply(
        lambda r: (r["positive"] / r["total_bins"]) * 100 if r["total_bins"] > 0 else 0,
        axis=1
    )
    reindexed["negative_share_%"] = reindexed.apply(
        lambda r: (r["negative"] / r["total_bins"]) * 100 if r["total_bins"] > 0 else 0,
        axis=1
    )

    out = reindexed.reset_index().rename(columns={"index": "period_dt"})
    if freq == "monthly":
        out["period_str"] = out["period_dt"].dt.strftime("%Y-%m")
    else:
        out["period_str"] = out["period_dt"].dt.strftime("%Y-%m-%d")

    cols_order = ["period_str", "positive", "negative", "articles", "avg_tone", "total_bins", "positive_share_%", "negative_share_%"]
    cols_present = [c for c in cols_order if c in out.columns]
    return out[cols_present].reset_index(drop=True)

# -------------------------
# Plot helpers
# -------------------------
def plot_sentiment_share(df, query, freq_label):
    total_articles = int(df["articles"].sum()) if "articles" in df.columns else None
    fig, ax = plt.subplots(figsize=(10, 5))
    x_dates = pd.to_datetime(df["period_str"], errors="coerce")
    if freq_label.lower().startswith("monthly"):
        width = 20
        fmt = "%b %Y"
    else:
        width = 6
        fmt = "%d %b %Y"
    ax.bar(x_dates, df["positive_share_%"], width=width, label="Positive %", align="center", color="green")
    ax.bar(x_dates, df["negative_share_%"], width=width, bottom=df["positive_share_%"], align="center", color="red")
    title = f"{freq_label} Sentiment Share"
    if total_articles is not None:
        title += f" — Total articles: {total_articles:,}"
    ax.set_title(title)
    ax.set_xlabel("Period")
    ax.set_ylabel("Share (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_dual_axis(df, query, freq_label):
    total_articles = int(df["articles"].sum()) if "articles" in df.columns else None
    x_dates = pd.to_datetime(df["period_str"], errors="coerce")
    if freq_label.lower().startswith("monthly"):
        fmt = "%b %Y"
    else:
        fmt = "%d %b %Y"
    fig, ax1 = plt.subplots(figsize=(10,5))
    width = 20 if freq_label.lower().startswith("monthly") else 6
    ax1.bar(x_dates, df["articles"], label="Article Volume", alpha=0.6, width=width)
    ax1.set_ylabel("Articles")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax2 = ax1.twinx()
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Average Tone")
    ax2.plot(x_dates, df["avg_tone"], linewidth=2)
    for i, (xi, yi) in enumerate(zip(x_dates, df["avg_tone"])):
        color = "green" if pd.notna(yi) and yi >= 0 else "red"
        ax2.plot(xi, yi, marker="o", color=color)
    if total_articles is not None:
        ax1.text(0.99, 0.95, f"Total articles: {total_articles:,}", ha="right", va="top", transform=ax1.transAxes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    custom_line = plt.Line2D([], [], color="green", marker="o", label="Avg Tone ≥ 0")
    custom_line2 = plt.Line2D([], [], color="red", marker="o", label="Avg Tone < 0")
    ax1.legend(lines1 + [custom_line, custom_line2], labels1 + ["Avg Tone ≥ 0", "Avg Tone < 0"], loc="upper left")
    fig.suptitle(f"{freq_label} Volume and Sentiment")
    fig.tight_layout()
    return fig

# -------------------------
# Streamlit UI (sidebar-first selection)
# -------------------------
with st.sidebar:
    st.title("⚙️ App Controls")

    # ---- Query Section ----
    st.markdown("### 1️⃣ Query")
    raw_query = st.text_input("GDELT query (use OR for multiple terms)", value="AI OR ChatGPT")
    query = normalize_query(raw_query)
    if raw_query and raw_query.strip() != query:
        st.info(f"Normalized query to `{query}` for GDELT API compatibility.")

    st.markdown("---")

    # ---- Date Selection (compact: two columns) ----
    st.markdown("### 2️⃣ Dates")
    today = datetime.now().date()
    default_end = today
    default_start = today - timedelta(days=364)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=default_start)
    with col2:
        end_date = st.date_input("End", value=default_end)

    st.markdown("---")

    # ---- Aggregation Frequency ----
    st.markdown("### 3️⃣ Aggregation")
    freq = st.radio("Frequency", ("monthly", "weekly"), horizontal=True)

    st.markdown("---")

    # ---- Source country filter (NEW) ----
    st.markdown("### 4️⃣ Source country (optional)")
    st.caption("If you select 1+ countries, the app will attempt to use gdeltdoc to filter by source country. Leave empty for a full search.")
    # Provide a short list for convenience — users can type other ISO2 codes.
    source_countries = st.multiselect(
        "Source country ISO2 codes (e.g. US, GB, FR, DE)",
        options=["US", "GB", "FR", "DE", "CA", "AU", "IN"],
        default=[]
    )
    if source_countries and not _GDELTDOC_AVAILABLE:
        st.warning("gdeltdoc not installed in this environment. Add 'gdeltdoc' to requirements.txt to enable server-side country filtering.")

    st.markdown("---")

    # ---- Output Selection ----
    st.markdown("### 5️⃣ Outputs")
    show_share = st.checkbox("📊 Sentiment Share chart", value=True)
    show_dual = st.checkbox("📈 Volume & Avg Tone chart", value=True)
    show_table = st.checkbox("📋 Aggregated data table", value=True)
    download_csv = st.checkbox("💾 Aggregated CSV download", value=True)
    raw_download = st.checkbox("🗃 Raw timeline CSV download", value=False)
    allow_fig_download = st.checkbox("🖼 Allow figure PNG downloads", value=True)

    st.markdown("---")

    # ---- Fetch Data Button ----
    st.markdown("### 6️⃣ Fetch & Analyze")
    any_output_selected = any([show_share, show_dual, show_table, download_csv, raw_download])
    if not any_output_selected:
        st.warning("Select at least one output option above before fetching.")
        fetch_button = st.button("🚫 Fetch Data", disabled=True, use_container_width=True)
    else:
        # prettier button styling
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: #0E79B2;
                color: white;
                font-weight: 600;
                border-radius: 8px;
            }
            div.stButton > button:hover { filter: brightness(1.05); }
            </style>
            """,
            unsafe_allow_html=True
        )
        fetch_button = st.button("🚀 Fetch Data & Plot", use_container_width=True)

# --- Main: trigger fetch when clicked ---
if fetch_button:
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    if start_dt >= end_dt:
        st.error("Start date must be earlier than end date.")
    else:
        with st.spinner("Fetching data from GDELT (this may take a moment)..."):
            try:
                merged = fetch_and_merge_choose(raw_query, start_dt, end_dt, source_countries)
            except Exception as e:
                st.exception(e)
                merged = pd.DataFrame()

        if merged.empty:
            st.warning("No timeline data returned. Try widening the query or date range, or remove source-country filters.")
        else:
            st.success(f"Fetched {len(merged)} timeline rows.")
            agg = aggregate_sentiment_share(merged, freq=freq)
            agg = fill_missing_periods(agg, start_dt, end_dt, freq=freq)

            if agg.empty:
                st.warning("No aggregated data produced.")
            else:
                freq_label = "Monthly" if freq == "monthly" else "Weekly (week-start Mondays)"

                # Display outputs
                if show_share:
                    st.subheader(f"{freq_label} Sentiment Share")
                    fig1 = plot_sentiment_share(agg, query, freq_label)
                    st.pyplot(fig1)
                    if allow_fig_download:
                        buf1 = BytesIO()
                        fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
                        buf1.seek(0)
                        st.download_button(
                            label="📥 Download Sentiment Share chart (PNG)",
                            data=buf1,
                            file_name="sentiment_share.png",
                            mime="image/png"
                        )

                if show_dual:
                    st.subheader("Volume & Avg Tone")
                    fig2 = plot_dual_axis(agg, query, freq_label)
                    st.pyplot(fig2)
                    if allow_fig_download:
                        buf2 = BytesIO()
                        fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
                        buf2.seek(0)
                        st.download_button(
                            label="📥 Download Volume & Avg Tone chart (PNG)",
                            data=buf2,
                            file_name="volume_avg_tone.png",
                            mime="image/png"
                        )

                if show_table:
                    st.subheader("Aggregated table")
                    st.dataframe(agg)

                if download_csv:
                    csv_buf = io.StringIO()
                    agg.to_csv(csv_buf, index=False)
                    csv_bytes = csv_buf.getvalue().encode("utf-8")
                    st.download_button("Download aggregated CSV", csv_bytes, file_name="gdelt_sentiment_agg.csv", mime="text/csv")

                if raw_download:
                    buf = io.StringIO()
                    merged.to_csv(buf, index=False)
                    st.download_button("Download timeline CSV (raw)", buf.getvalue().encode("utf-8"), file_name="gdelt_timeline_raw.csv", mime="text/csv")

st.markdown("---")
st.caption("This tool fetches GDELT TimelineTone and TimelineVol (or uses gdeltdoc when source-country filtering is requested). It fills missing periods and interpolates average tone across time. The positive/negative threshold is 0.4.")
