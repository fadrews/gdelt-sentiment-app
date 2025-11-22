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
import gdeltdoc
# Try to import gdeltdoc (optional). If unavailable, we will fall back.
try:
    from gdeltdoc import GdeltDoc, Filters
    _GDELTDOC_AVAILABLE = True

    # PATCH: Fix gdeltdoc 1.5 bug - remove parentheses and quotes that cause API errors
    # This fixes the "Parentheses may only be used around OR'd statements" error
    original_init = Filters.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Remove parentheses around sourcecountry filter: (sourcecountry:US) -> sourcecountry:US
        for i, param in enumerate(self.query_params):
            if 'sourcecountry:' in str(param):
                cleaned = param.strip()
                if cleaned.startswith('(') and cleaned.endswith(')'):
                    self.query_params[i] = cleaned[1:-1] + ' '

        # Remove quotes around keyword: "election" -> election
        if len(self.query_params) > 0:
            keyword_param = self.query_params[0]
            if keyword_param.startswith('"') and keyword_param.strip().endswith('"'):
                self.query_params[0] = keyword_param.strip()[1:-1] + ' '

    Filters.__init__ = patched_init

    # Patch the query_string property to rebuild from modified query_params
    original_query_string = Filters.query_string.fget
    def patched_query_string(self):
        return ''.join(self.query_params)
    Filters.query_string = property(patched_query_string)
    print("✅ GDELTDOC PATCH APPLIED SUCCESSFULLY")
except ImportError:
    _GDELTDOC_AVAILABLE = False
except Exception as e:
    _GDELTDOC_AVAILABLE = False
    print(f"Error loading gdeltdoc: {e}")

# -------------------------
# Config / Defaults
# -------------------------
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Compact sidebar CSS (optional, reduces vertical spacing)
st.set_page_config(page_title="GDELT Sentiment Share Explorer", layout="wide")
st.markdown(
    """
    <style>
    /* Reduce space between widgets inside sidebar */
    section[data-testid="stSidebar"] .block-container { 
        padding: 0.5rem 1rem; 
    }
    div[data-testid="stVerticalBlock"] > div { 
        padding-top: 0rem; 
        padding-bottom: 0rem; 
        margin-top: 0rem;
        margin-bottom: 0rem;
    }
    h2, h3 { 
        margin-bottom: 0.2rem; 
        margin-top: 0.5rem; 
    }
    div.stCheckbox { 
        margin-bottom: 0rem;
        margin-top: 0rem;
    }
    div.stButton > button:first-child { 
        height: 2.4em;
        margin-top: 0.5rem;
    }
    div.stMarkdown {
        margin-bottom: 0rem;
    }
    /* Reduce spacing for radio buttons */
    div[data-testid="stRadio"] {
        margin-top: 0rem;
        margin-bottom: 0rem;
    }
    /* Reduce spacing for multiselect */
    div[data-testid="stMultiSelect"] {
        margin-top: 0rem;
        margin-bottom: 0rem;
    }
    /* Reduce caption spacing */
    .stCaption {
        margin-top: 0rem;
        margin-bottom: 0.2rem;
    }
    /* Make horizontal rules thinner and less spaced */
    section[data-testid="stSidebar"] hr {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# -------------------------
# Utility helpers
# -------------------------

def normalize_query(q: str) -> str:
    """Normalize query for GDELT API - wrap OR queries in parentheses."""
    if not q:
        return q
    q_str = q.strip()
    if " OR " in q_str.upper():
        if q_str.startswith("(") and q_str.endswith(")"):
            return q_str
        return "(" + q_str + ")"
    return q_str


def request_gdelt(params, timeout=30):
    """Make request to GDELT API with error handling."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(GDELT_URL, params=params, headers=headers, timeout=timeout)
        text = (r.text or "").strip()

        if not text:
            raise RuntimeError("Empty response from GDELT (no content). Check your network or parameters.")

        if r.status_code != 200:
            snippet = text[:800] + ("..." if len(text) > 800 else "")
            raise RuntimeError(f"GDELT returned status {r.status_code}. Response: {snippet}")

        return r.json()

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request timed out after {timeout} seconds. Try widening your date range or simplifying your query.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error occurred: {str(e)}")
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

def fetch_timeline_tone(query, start_dt, end_dt):
    """Fetch timeline tone data from GDELT API."""
    params = {
        "query": query,
        "format": "json",
        "mode": "timelinetone",
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


def fetch_timeline_vol(query, start_dt, end_dt):
    """Fetch timeline volume data from GDELT API."""
    params = {
        "query": query,
        "format": "json",
        "mode": "timelinevol",
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
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# Compatibility wrappers for older call sites
def fetch_timeline_tone_raw(query, start_dt, end_dt, *args, **kwargs):
    """Backwards-compatible wrapper for fetch_timeline_tone."""
    return fetch_timeline_tone(query, start_dt, end_dt)


def fetch_timeline_vol_raw(query, start_dt, end_dt, *args, **kwargs):
    """Backwards-compatible wrapper for fetch_timeline_vol."""
    return fetch_timeline_vol(query, start_dt, end_dt)


# -------------------------
# gdeltdoc-based fetch (preferred when source_countries are provided)
# -------------------------

def fetch_with_gdeltdoc(raw_query, start_dt, end_dt, country_list, verbose=False):
    """
    Robust gdeltdoc fetch that probes available mode names and tries multiple candidates.
    Returns merged DataFrame with columns: date (datetime), tone_value, articles.
    Raises RuntimeError if gdeltdoc not available or all probes fail.
    """
    if not _GDELTDOC_AVAILABLE:
        raise RuntimeError(
            "gdeltdoc not installed. Add 'gdeltdoc' to requirements.txt to use source-country filtering."
        )

    # Strip parentheses that were added for raw GDELT API, BUT keep them for OR queries
    clean_query = raw_query.strip()
    if clean_query.startswith("(") and clean_query.endswith(")"):
        # Check if it contains OR - if yes, keep the parentheses
        if " OR " not in clean_query.upper():
            clean_query = clean_query[1:-1]  # Remove outer parentheses only if no OR
    # If query has OR but no parentheses, add them
    elif " OR " in clean_query.upper():
        clean_query = f"({clean_query})"

    # Prepare Filters (string dates)
    filt = Filters(
        keyword=clean_query,  # ← Use cleaned query
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        country=country_list
    )
    # DEBUG: Print what query is actually being sent
    if verbose:
        print(f"[DEBUG] Query being sent to GDELT API:")
        print(f"[DEBUG]   Raw query: {raw_query}")
        print(f"[DEBUG]   Clean query: {clean_query}")
        print(f"[DEBUG]   Filter query string: {filt.query_string}")
        print(f"[DEBUG]   Filter query params: {filt.query_params}")
    gd = GdeltDoc()

    # Helper to discover available modes on the client object
    client_modes = set()
    for attr in ("modes", "available_modes", "mode_list", "_modes", "MODES"):
        try:
            val = getattr(gd, attr)
            if callable(val):
                val = val()
            if isinstance(val, (list, set, tuple)):
                client_modes.update([str(x).lower() for x in val])
        except Exception:
            pass

    # Add some likely timeline modes into a probe list (lowercase)
    tone_candidates = ["timelinetone", "TimelineTone", "timeline_tone", "tone", "timelinetone_raw"]
    vol_candidates = ["timelinevol", "timelinevolraw", "timelinevol_raw", "timeline_vol", "vol"]

    # Normalize client modes to lowercase for comparison
    client_modes_lc = set([m.lower() for m in client_modes])

    def pick_candidate(candidates):
        """Return the first candidate that matches client_modes_lc, else None."""
        for c in candidates:
            if c.lower() in client_modes_lc:
                return c
        return None

    chosen_tone_mode = pick_candidate(tone_candidates)
    chosen_vol_mode = pick_candidate(vol_candidates)

    tried = {"tone": [], "vol": []}
    tone_df = pd.DataFrame()
    vol_df = pd.DataFrame()

    # Try tone candidates (best-first)
    tone_attempts = [chosen_tone_mode] + tone_candidates if chosen_tone_mode else tone_candidates
    for mode in tone_attempts:
        if not mode or mode in tried["tone"]:
            continue
        tried["tone"].append(mode)
        try:
            if verbose:
                print(f"[gdeltdoc] trying tone mode: {mode}")
            tone_df = gd.timeline_search(mode, filt)
            # ADD THESE DEBUG LINES:
            if verbose:
                print(
                    f"[DEBUG] tone_df type: {type(tone_df)}, empty: {tone_df.empty if isinstance(tone_df, pd.DataFrame) else 'N/A'}")
                if isinstance(tone_df, pd.DataFrame):
                    print(f"[DEBUG] tone_df shape: {tone_df.shape}, columns: {list(tone_df.columns)}")
            # sanity check: expect a DataFrame with a 'date' column
            if isinstance(tone_df, pd.DataFrame) and not tone_df.empty and (
                    "date" in tone_df.columns or "datetime" in tone_df.columns):
                if verbose:
                    print(f"[gdeltdoc] tone mode succeeded: {mode}")
                break
            else:
                tone_df = pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"[gdeltdoc] tone mode '{mode}' failed: {e}")
            tone_df = pd.DataFrame()

    # Try vol candidates (best-first)
    vol_attempts = [chosen_vol_mode] + vol_candidates if chosen_vol_mode else vol_candidates
    for mode in vol_attempts:
        if not mode:
            continue
        if mode in tried["vol"]:
            continue
        tried["vol"].append(mode)
        try:
            if verbose:
                print(f"[gdeltdoc] trying vol mode: {mode}")
            vol_df = gd.timeline_search(mode, filt)
            if verbose:
                print(
                    f"[DEBUG] vol_df type: {type(vol_df)}, empty: {vol_df.empty if isinstance(vol_df, pd.DataFrame) else 'N/A'}")
                if isinstance(vol_df, pd.DataFrame):
                    print(f"[DEBUG] vol_df shape: {vol_df.shape}, columns: {list(vol_df.columns)}")
            if isinstance(vol_df, pd.DataFrame) and not vol_df.empty and (
                    "date" in vol_df.columns or "datetime" in vol_df.columns):
                if verbose:
                    print(f"[gdeltdoc] vol mode succeeded: {mode}")
                break
            else:
                vol_df = pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"[gdeltdoc] vol mode '{mode}' failed: {e}")
            vol_df = pd.DataFrame()
    # If both are empty, raise informative error so caller can fallback
    if (tone_df is None or tone_df.empty) and (vol_df is None or vol_df.empty):
        raise RuntimeError(
            f"No data found with country filter."
        )
    # Normalize column names
    if isinstance(tone_df, pd.DataFrame) and not tone_df.empty:
        if "datetime" in tone_df.columns and "date" not in tone_df.columns:
            tone_df = tone_df.rename(columns={"datetime": "date"})
        if "Average Tone" in tone_df.columns:
            tone_df = tone_df.rename(columns={"Average Tone": "tone_value"})

    if isinstance(vol_df, pd.DataFrame) and not vol_df.empty:
        if "datetime" in vol_df.columns and "date" not in vol_df.columns:
            vol_df = vol_df.rename(columns={"datetime": "date"})
        if "Article Count" in vol_df.columns:
            vol_df = vol_df.rename(columns={"Article Count": "articles"})
        elif "Volume Intensity" in vol_df.columns:
            vol_df = vol_df.rename(columns={"Volume Intensity": "articles"})

    # If both are empty, raise informative error
    if (tone_df is None or tone_df.empty) and (vol_df is None or vol_df.empty):
        raise RuntimeError(
            f"gdeltdoc: no timeline modes succeeded. Tried tone modes: {tried['tone']}; vol modes: {tried['vol']}. "
            "This usually means no articles matched your query with the specified filters. "
            "Try widening your date range, simplifying your query, or removing source-country filters."
        )

    # Normalize dataframes into canonical shape
    if isinstance(tone_df, pd.DataFrame) and not tone_df.empty:
        tone_df = tone_df.copy()
        if "value" in tone_df.columns and "tone_value" not in tone_df.columns:
            tone_df = tone_df.rename(columns={"value": "tone_value"})
        if "count" in tone_df.columns and "tone_sample_count" not in tone_df.columns:
            tone_df = tone_df.rename(columns={"count": "tone_sample_count"})
        if "date" in tone_df.columns:
            tone_df["date"] = pd.to_datetime(tone_df["date"], errors="coerce")

    if isinstance(vol_df, pd.DataFrame) and not vol_df.empty:
        vol_df = vol_df.copy()
        possible_cols = ["value", "articles", "count", "vol_sample_count", "value_vol", "count_vol"]
        found = None
        for c in possible_cols:
            if c in vol_df.columns:
                found = c
                break
        if found:
            vol_df = vol_df.rename(columns={found: "articles"})
            vol_df["articles"] = pd.to_numeric(
                vol_df["articles"].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0.0)
        if "date" in vol_df.columns:
            vol_df["date"] = pd.to_datetime(vol_df["date"], errors="coerce")

    # Merge and return
    if not tone_df.empty and not vol_df.empty:
        merged = pd.merge(tone_df, vol_df, on="date", how="outer", suffixes=("_tone", "_vol"))
    elif not tone_df.empty:
        merged = tone_df
    elif not vol_df.empty:
        merged = vol_df
    else:
        merged = pd.DataFrame()

    if merged.empty:
        return merged

    if "articles" not in merged.columns:
        merged["articles"] = 0.0
    merged["articles"] = pd.to_numeric(merged["articles"], errors="coerce").fillna(0.0)

    if "tone_value" not in merged.columns:
        merged["tone_value"] = pd.to_numeric(
            merged.get("tone_value", pd.Series(dtype=float)), errors="coerce"
        )

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


# -------------------------
# Fallback wrapper: choose gdeltdoc if countries provided, else raw
# -------------------------

def fetch_and_merge_choose(raw_query, start_dt, end_dt, source_countries=None):
    """
    If source_countries provided (non-empty list) and gdeltdoc is available, use it.
    Otherwise use the raw GDELT HTTP fetching functions.
    Returns merged DataFrame with date, tone_value, articles (floats).
    """
    if source_countries and len(source_countries) > 0:
        if not _GDELTDOC_AVAILABLE:
            raise RuntimeError(
                "You selected source_country filtering but 'gdeltdoc' is not installed in this environment. "
                "Add 'gdeltdoc' to requirements.txt."
            )

        # Try gdeltdoc with fallback to raw on failure
        try:
            merged = fetch_with_gdeltdoc(raw_query, start_dt, end_dt, source_countries)
            if not merged.empty:
                return merged
            # If empty, fall through to raw method
            country_names = ", ".join(source_countries)
            st.info(
                f"ℹ️ No articles found from {country_names} sources. "
                f"Showing worldwide results instead."
            )
        except RuntimeError as e:
            country_names = ", ".join(source_countries)
            st.info(
                f"ℹ️ No articles found from {country_names} sources for this query and date range. "
                f"Showing worldwide results instead."
            )
    # Raw GDELT fallback (no country filtering)
    query = normalize_query(raw_query)

    try:
        tone_df = fetch_timeline_tone_raw(query, start_dt, end_dt)
        vol_df = fetch_timeline_vol_raw(query, start_dt, end_dt)
    except Exception as e:
        st.error(f"Failed to fetch from GDELT API: {str(e)}")
        return pd.DataFrame()

    if tone_df.empty and vol_df.empty:
        return pd.DataFrame()

    # Normalize date/time and merge
    for ddf in (tone_df, vol_df):
        if not ddf.empty:
            ddf["date"] = pd.to_datetime(ddf["date"], errors="coerce")
            try:
                if pd.api.types.is_datetime64tz_dtype(ddf["date"].dtype):
                    ddf["date"] = ddf["date"].dt.tz_convert(None)
            except Exception:
                pass
            ddf["date"] = ddf["date"].dt.round("1s")

    # Detect volume column and rename
    possible_cols = ["value", "articles", "count", "vol_sample_count", "value_vol", "count_vol"]
    found = None
    if not vol_df.empty:
        for c in possible_cols:
            if c in vol_df.columns:
                found = c
                break
    if found:
        vol_df = vol_df.rename(columns={found: "articles"})
        vol_df["articles"] = pd.to_numeric(
            vol_df["articles"].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0.0)
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
# Aggregation and filler
# -------------------------

def aggregate_sentiment_share(df, freq="monthly"):
    """Aggregate timeline data by frequency with sentiment classification."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy().dropna(subset=["date"]).reset_index(drop=True)

    if "tone_value" not in df.columns:
        df["tone_value"] = pd.NA

    if freq == "monthly":
        # Convert to period, removing timezone first to avoid warning
        if pd.api.types.is_datetime64tz_dtype(df["date"]):
            df["date"] = df["date"].dt.tz_localize(None)
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
    """Fill missing time periods with zeros and interpolate average tone."""
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

    cols_order = ["period_str", "positive", "negative", "articles", "avg_tone", "total_bins",
                  "positive_share_%", "negative_share_%"]
    cols_present = [c for c in cols_order if c in out.columns]
    return out[cols_present].reset_index(drop=True)


# -------------------------
# Plot helpers
# -------------------------

def plot_sentiment_share(df, query, freq_label):
    """Create sentiment share stacked bar chart."""
    total_articles = int(df["articles"].sum()) if "articles" in df.columns else None
    fig, ax = plt.subplots(figsize=(10, 5))
    x_dates = pd.to_datetime(df["period_str"], errors="coerce")

    if freq_label.lower().startswith("monthly"):
        width = 20

        # Format as Year-Quarter (e.g., 2024-Q3)
        def format_quarter(x, pos):
            if pd.notna(x):
                dt = mdates.num2date(x)
                quarter = (dt.month - 1) // 3 + 1
                return f"{dt.year}-Q{quarter}"
            return ""

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_quarter))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every quarter
    else:
        width = 6

        # For weekly, show Year-Quarter
        def format_quarter(x, pos):
            if pd.notna(x):
                dt = mdates.num2date(x)
                quarter = (dt.month - 1) // 3 + 1
                return f"{dt.year}-Q{quarter}"
            return ""

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_quarter))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax.bar(x_dates, df["positive_share_%"], width=width, label="Positive %",
           align="center", color="green")
    ax.bar(x_dates, df["negative_share_%"], width=width, bottom=df["positive_share_%"],
           align="center", color="red", label="Negative %")

    title = f"{freq_label} Sentiment Share"
    if total_articles is not None:
        title += f" — Total articles: {total_articles:,}"
    ax.set_title(title)
    ax.set_xlabel("Period")
    ax.set_ylabel("Share (%)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_dual_axis(df, query, freq_label):
    """Create dual-axis chart with volume bars and tone line."""
    total_articles = int(df["articles"].sum()) if "articles" in df.columns else None
    x_dates = pd.to_datetime(df["period_str"], errors="coerce")

    if freq_label.lower().startswith("monthly"):
        width = 20
        # Format as Year-Quarter
        def format_quarter(x, pos):
            if pd.notna(x):
                dt = mdates.num2date(x)
                quarter = (dt.month - 1) // 3 + 1
                return f"{dt.year}-Q{quarter}"
            return ""
    else:
        width = 6
        # For weekly, also show Year-Quarter
        def format_quarter(x, pos):
            if pd.notna(x):
                dt = mdates.num2date(x)
                quarter = (dt.month - 1) // 3 + 1
                return f"{dt.year}-Q{quarter}"
            return ""

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x_dates, df["articles"], label="Article Volume", alpha=0.6, width=width)
    ax1.set_ylabel("Articles")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_quarter))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every quarter
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2 = ax1.twinx()

    # Calculate statistics for tone
    tone_values = df["avg_tone"].dropna()
    if len(tone_values) > 0:
        mean_tone = tone_values.mean()
        std_tone = tone_values.std()

        # Add reference lines
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Zero")
        ax2.axhline(mean_tone, color="blue", linestyle="-", linewidth=1.5, alpha=0.7, label=f"Mean ({mean_tone:.2f})")
        ax2.axhline(mean_tone + 2 * std_tone, color="black", linestyle=":", linewidth=1.2, alpha=0.6,
                    label=f"+2 SD ({mean_tone + 2 * std_tone:.2f})")
        ax2.axhline(mean_tone - 2 * std_tone, color="black", linestyle=":", linewidth=1.2, alpha=0.6,
                    label=f"-2 SD ({mean_tone - 2 * std_tone:.2f})")
    else:
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax2.set_ylabel("Average Tone")
    ax2.plot(x_dates, df["avg_tone"], linewidth=1, color="black", label="Avg Tone")

    for i, (xi, yi) in enumerate(zip(x_dates, df["avg_tone"])):
        color = "green" if pd.notna(yi) and yi >= 0 else "red"
        ax2.plot(xi, yi, marker="o", color=color, markersize=4)

    if total_articles is not None:
        ax1.text(0.99, 0.95, f"Total articles: {total_articles:,}",
                 ha="right", va="top", transform=ax1.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    fig.suptitle(f"{freq_label} Volume and Sentiment")
    fig.tight_layout()
    return fig
# -------------------------
# Streamlit UI (sidebar-first selection)
# -------------------------

with st.sidebar:
    st.title("Search Entry")

    # ---- Query Section ----
    st.markdown("### 1️⃣ Query")
    raw_query = st.text_input("GDELT query (use OR for multiple terms)", value="AI OR ChatGPT")

    # Add helpful hint
    if raw_query and " " in raw_query and " OR " not in raw_query.upper() and " AND " not in raw_query.upper():
        st.caption(
            "💡 Tip: Searching for articles containing ALL these words. Use 'OR' (e.g., 'AI OR ChatGPT') to find articles with ANY of these words.")
    query = normalize_query(raw_query)
    if raw_query and raw_query.strip() != query:
        st.info(f"Normalized query to `{query}`.")

    st.markdown("")

    # ---- Date Selection (compact: two columns) ----
    st.markdown("### 2️⃣ Date range")
    today = datetime.now().date()
    default_end = today
    default_start = today - timedelta(days=364)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=default_start)
    with col2:
        end_date = st.date_input("End", value=default_end)

    st.markdown("")

    # ---- Aggregation Frequency ----
    st.markdown("### 3️⃣ Aggregation")
    freq = st.radio("Frequency", ("monthly", "weekly"), horizontal=True)

    st.markdown("")

    # ---- Source country filter ----
    st.markdown("### 4️⃣ Source country (optional)")
    st.caption("If you select 1+ countries. Leave empty for a full search.")
    source_countries = st.multiselect(
        "Source country (e.g. US, GB, FR, DE)",
        options=["US", "GB", "FR", "DE", "CA", "AU", "IN", "JP", "CN", "RU", "BR"],
        default=[]
    )
    if source_countries and not _GDELTDOC_AVAILABLE:
        st.warning("⚠️ gdeltdoc not installed in this environment. "
                  "Add 'gdeltdoc' to requirements.txt to enable server-side country filtering.")

    st.markdown("")

    # ---- Output Selection ----
    st.markdown("### 5️⃣ Outputs")
    show_share = st.checkbox("Sentiment Share chart", value=True)
    show_dual = st.checkbox("Volume & Avg Tone chart", value=True)
 #   show_table = st.checkbox("Aggregated data table", value=True)
 #   download_csv = st.checkbox("Aggregated CSV download", value=True)
 #   raw_download = st.checkbox("Raw timeline CSV download", value=False)
 #   allow_fig_download = st.checkbox("Allow figure PNG downloads", value=True)

    st.markdown("---")

    # ---- Fetch Data Button ----
    st.markdown("### 6️⃣ Fetch & Analyze")
    any_output_selected = any([show_share, show_dual])
    if not any_output_selected:
        st.warning("Select at least one chart option above before fetching.")
        fetch_button = st.button("🚫 Fetch Data", disabled=True, use_container_width=True)
    else:
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
        fetch_button = st.button("Fetch Data & Plot", use_container_width=True)

# --- Main: trigger fetch when clicked ---
if fetch_button:
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    if start_dt >= end_dt:
        st.error("❌ Start date must be earlier than end date.")
    else:
        with st.spinner("Fetching data from GDELT (this may take a moment)..."):
            try:
                merged = fetch_and_merge_choose(raw_query, start_dt, end_dt, source_countries)
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.exception(e)
                merged = pd.DataFrame()

        if merged is None or merged.empty:
            st.warning(
                "⚠️ No timeline data returned. Try:\n"
                "- Widening your date range\n"
                "- Simplifying your query\n"
                "- Removing source-country filters\n"
                "- Using a more common search term"
            )
        else:
            st.success(f"✅ Fetched {len(merged)} timeline rows.")

            with st.spinner("Aggregating and processing data..."):
                agg = aggregate_sentiment_share(merged, freq=freq)
                agg = fill_missing_periods(agg, start_dt, end_dt, freq=freq)

            if agg.empty:
                st.warning("⚠️ No aggregated data produced.")
            else:
                freq_label = "Monthly" if freq == "monthly" else "Weekly (week-start Mondays)"

                # Display selected charts
                if show_share:
                    st.subheader(f"{freq_label} Sentiment Share")
                    fig1 = plot_sentiment_share(agg, query, freq_label)
                    st.pyplot(fig1)
                    buf1 = BytesIO()
                    fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
                    buf1.seek(0)
                    st.download_button(
                        label="📥 Download Chart (PNG)",
                        data=buf1,
                        file_name="sentiment_share.png",
                        mime="image/png"
                    )

                if show_dual:
                    st.subheader("Volume & Avg Tone")
                    fig2 = plot_dual_axis(agg, query, freq_label)
                    st.pyplot(fig2)
                    buf2 = BytesIO()
                    fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
                    buf2.seek(0)
                    st.download_button(
                        label="📥 Download Chart (PNG)",
                        data=buf2,
                        file_name="volume_avg_tone.png",
                        mime="image/png"
                    )

                # Always show table and downloads
                st.markdown("---")
                st.subheader("Data & Downloads")

                # Show table first
                with st.expander("📋 View Aggregated Data Table", expanded=True):
                    st.dataframe(agg, use_container_width=True, height=400)

                # Download buttons in vertical lines
                st.markdown("**Download Options:**")

                csv_buf = io.StringIO()
                agg.to_csv(csv_buf, index=False)
                csv_bytes = csv_buf.getvalue().encode("utf-8")
                st.download_button(
                    "📥 Download Aggregated Data (CSV)",
                    csv_bytes,
                    file_name="gdelt_sentiment_agg.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                buf = io.StringIO()
                merged.to_csv(buf, index=False)
                st.download_button(
                    "📥 Download Raw Timeline Data (CSV)",
                    buf.getvalue().encode("utf-8"),
                    file_name="gdelt_timeline_raw.csv",
                    mime="text/csv",
                    use_container_width=True
                )

st.markdown("")
st.caption(
    "This app fetches GDELT TimelineTone and TimelineVol data (or uses gdeltdoc when source-country "
    "filtering is requested). It fills missing periods and interpolates average tone across time. "
    "The positive/negative threshold is 0.4."
)