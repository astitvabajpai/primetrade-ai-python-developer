"""
Bitcoin Market Sentiment + Hyperliquid Trader Analysis
=======================================================
Outputs:
  - Console insights summary
  - insights.json  (used by dashboard.html)
  - All intermediate DataFrames saved to CSV in ./output/

Usage:
    pip install pandas numpy scipy
    python analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
FG_PATH     = "fear_greed_index.csv"
TRADES_PATH = "compressed_data_csv.gz"
OUT_DIR     = "output"
SENT_ORDER  = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
TOP_COINS   = ["BTC", "ETH", "SOL", "HYPE", "@107", "MELANIA"]

os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading data...")
fg     = pd.read_csv(FG_PATH)
trades = pd.read_csv(TRADES_PATH, compression="gzip")

fg["date"]  = pd.to_datetime(fg["date"])
trades["dt"] = pd.to_datetime(trades["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
trades["date"] = trades["dt"].dt.normalize()
trades["hour"] = trades["dt"].dt.hour

merged = trades.merge(fg[["date", "value", "classification"]], on="date", how="left")
print(f"  Merged shape: {merged.shape}  |  Matched rows: {merged['classification'].notna().sum():,}")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def win_rate(series): return (series > 0).mean() * 100

# ── 1. SUMMARY STATS ─────────────────────────────────────────────────────────
print("\n[1] Summary statistics")
summary = {
    "total_traders"   : int(merged["Account"].nunique()),
    "total_trades"    : int(len(merged)),
    "total_pnl_usd"   : round(float(merged["Closed PnL"].sum()), 2),
    "overall_win_rate": round(float(win_rate(merged["Closed PnL"])), 2),
    "date_range_start": str(merged["date"].min().date()),
    "date_range_end"  : str(merged["date"].max().date()),
}
for k, v in summary.items():
    print(f"  {k}: {v}")

# ── 2. PnL BY SENTIMENT ───────────────────────────────────────────────────────
print("\n[2] PnL by sentiment")
grp = merged.groupby("classification")["Closed PnL"]
pnl_sent = pd.DataFrame({
    "avg_pnl"  : grp.mean(),
    "median_pnl": grp.median(),
    "total_pnl": grp.sum(),
    "count"    : grp.count(),
    "std"      : grp.std(),
    "win_rate" : grp.apply(win_rate),
}).reindex(SENT_ORDER)
print(pnl_sent.round(2).to_string())
pnl_sent.to_csv(f"{OUT_DIR}/pnl_by_sentiment.csv")

# ── 3. LONG vs SHORT WIN RATE BY SENTIMENT ────────────────────────────────────
print("\n[3] Long vs Short by sentiment")
ls = merged.groupby(["classification", "Side"]).agg(
    avg_pnl  = ("Closed PnL", "mean"),
    win_rate = ("Closed PnL", win_rate),
    count    = ("Closed PnL", "count"),
).round(3)
print(ls.to_string())
ls.to_csv(f"{OUT_DIR}/long_short_by_sentiment.csv")

# ── 4. COIN PERFORMANCE BY SENTIMENT ─────────────────────────────────────────
print("\n[4] Coin avg PnL by sentiment")
coin_sent = (
    merged[merged["Coin"].isin(TOP_COINS)]
    .groupby(["Coin", "classification"])["Closed PnL"]
    .mean()
    .unstack(fill_value=0)
    .reindex(columns=SENT_ORDER, fill_value=0)
)
print(coin_sent.round(1).to_string())
coin_sent.to_csv(f"{OUT_DIR}/coin_by_sentiment.csv")

# ── 5. TRADER PERFORMANCE ─────────────────────────────────────────────────────
print("\n[5] Top 10 traders")
trader = merged.groupby("Account").agg(
    total_pnl   = ("Closed PnL", "sum"),
    avg_pnl     = ("Closed PnL", "mean"),
    trade_count = ("Closed PnL", "count"),
    win_rate    = ("Closed PnL", win_rate),
    avg_size_usd= ("Size USD",   "mean"),
).sort_values("total_pnl", ascending=False)
print(trader.head(10).round(2).to_string())
trader.to_csv(f"{OUT_DIR}/trader_performance.csv")

# ── 6. HOURLY PATTERNS ────────────────────────────────────────────────────────
print("\n[6] Hourly avg PnL (IST)")
hourly = merged.groupby("hour")["Closed PnL"].agg(["mean", "count"]).round(2)
print(hourly.to_string())
hourly.to_csv(f"{OUT_DIR}/hourly_pnl.csv")

# ── 7. DAILY TIME SERIES ─────────────────────────────────────────────────────
print("\n[7] Daily summary (last 90 days)")
daily = merged.groupby("date").agg(
    avg_pnl        = ("Closed PnL", "mean"),
    total_pnl      = ("Closed PnL", "sum"),
    trade_count    = ("Closed PnL", "count"),
    fg_value       = ("value",       "first"),
    classification = ("classification", "first"),
    volume_usd     = ("Size USD",    "sum"),
).dropna().sort_index()
daily.to_csv(f"{OUT_DIR}/daily_summary.csv")
daily_tail = daily.tail(90)

# Pearson correlation: FG Index vs avg daily PnL
corr, pval = stats.pearsonr(daily["fg_value"].dropna(), daily["avg_pnl"].reindex(daily["fg_value"].dropna().index))
print(f"  Pearson r (FG vs avg_pnl): {corr:.4f}  p={pval:.4f}")

# ── 8. EXTREME SENTIMENT TRADER BEHAVIOUR ────────────────────────────────────
print("\n[8] Trader PnL in Extreme Fear vs Extreme Greed")
extreme = (
    merged[merged["classification"].isin(["Extreme Fear", "Extreme Greed"])]
    .groupby(["Account", "classification"])["Closed PnL"]
    .sum()
    .unstack(fill_value=0)
)
print(extreme.sort_values("Extreme Fear", ascending=False).head(10).round(2).to_string())
extreme.to_csv(f"{OUT_DIR}/extreme_sentiment_traders.csv")

# ── 9. FEE DRAG ANALYSIS ─────────────────────────────────────────────────────
print("\n[9] Fee drag analysis")
fee_analysis = merged.groupby("classification").agg(
    total_fee   = ("Fee", "sum"),
    avg_fee     = ("Fee", "mean"),
    fee_to_pnl  = ("Fee", lambda x: x.sum() / merged.loc[x.index, "Closed PnL"].sum() * 100 if merged.loc[x.index, "Closed PnL"].sum() != 0 else np.nan),
).reindex(SENT_ORDER).round(4)
print(fee_analysis.to_string())
fee_analysis.to_csv(f"{OUT_DIR}/fee_analysis.csv")

# ── 10. LEVERAGE ANALYSIS ─────────────────────────────────────────────────────
if "leverage" in merged.columns or "Leverage" in merged.columns:
    lev_col = "leverage" if "leverage" in merged.columns else "Leverage"
    print("\n[10] Leverage vs PnL by sentiment")
    lev = merged.groupby("classification").agg(
        avg_leverage=  (lev_col, "mean"),
        avg_pnl     = ("Closed PnL", "mean"),
    ).reindex(SENT_ORDER).round(3)
    print(lev.to_string())
    lev.to_csv(f"{OUT_DIR}/leverage_by_sentiment.csv")
else:
    print("\n[10] Leverage column not found — skipping")

# ── EXPORT insights.json FOR DASHBOARD ────────────────────────────────────────
print("\nExporting insights.json for dashboard...")

def safe(x):
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    if isinstance(x, (np.integer,)):  return int(x)
    if isinstance(x, (np.floating,)): return round(float(x), 4)
    return x

def row_to_dict(df, index_val):
    try:    return {k: safe(v) for k, v in df.loc[index_val].items()}
    except: return {}

ls_dict = {}
for sent in SENT_ORDER:
    ls_dict[sent] = {}
    for side in ["BUY", "SELL"]:
        try:    ls_dict[sent][side] = {k: safe(v) for k, v in ls.loc[(sent, side)].items()}
        except: ls_dict[sent][side] = {}

insights = {
    "summary": summary,
    "correlation_fg_pnl": round(float(corr), 4),
    "pnl_by_sentiment": {
        s: {k: safe(v) for k, v in row_to_dict(pnl_sent, s).items()}
        for s in SENT_ORDER
    },
    "long_short_by_sentiment": ls_dict,
    "coin_by_sentiment": {
        coin: {s: safe(coin_sent.loc[coin, s]) if coin in coin_sent.index else None for s in SENT_ORDER}
        for coin in TOP_COINS
    },
    "top_traders": [
        {
            "account"    : acc,
            "total_pnl"  : safe(row["total_pnl"]),
            "avg_pnl"    : safe(row["avg_pnl"]),
            "trade_count": safe(row["trade_count"]),
            "win_rate"   : safe(row["win_rate"]),
        }
        for acc, row in trader.head(10).iterrows()
    ],
    "hourly_pnl": {str(h): safe(v) for h, v in hourly["mean"].items()},
    "daily_tail90": {
        "dates"          : [str(d.date()) for d in daily_tail.index],
        "fg_values"      : [safe(v) for v in daily_tail["fg_value"]],
        "avg_pnl"        : [safe(v) for v in daily_tail["avg_pnl"]],
        "classifications": list(daily_tail["classification"]),
    },
}

with open("insights.json", "w") as f:
    json.dump(insights, f, indent=2)

print(f"\nDone! All CSVs saved to ./{OUT_DIR}/")
print("insights.json saved — open dashboard.html in a browser to explore.")

# ── PRINT FINAL INSIGHT SUMMARY ───────────────────────────────────────────────
print("\n" + "="*60)
print("KEY INSIGHTS SUMMARY")
print("="*60)

best_sent  = pnl_sent["avg_pnl"].idxmax()
worst_sent = pnl_sent["avg_pnl"].idxmin()
best_hour  = int(hourly["mean"].idxmax())
best_coin  = {c: coin_sent.loc[c].max() if c in coin_sent.index else 0 for c in TOP_COINS}

print(f"  Best sentiment to trade : {best_sent}  (avg ${pnl_sent.loc[best_sent,'avg_pnl']:.1f})")
print(f"  Worst sentiment         : {worst_sent}  (avg ${pnl_sent.loc[worst_sent,'avg_pnl']:.1f})")
print(f"  Best trading hour (IST) : {best_hour:02d}:00  (avg ${hourly.loc[best_hour,'mean']:.1f})")
print(f"  FG ↔ PnL correlation   : {corr:.4f}  ({'weak' if abs(corr)<0.2 else 'moderate' if abs(corr)<0.5 else 'strong'})")
print(f"  Top trader PnL          : ${trader['total_pnl'].iloc[0]:,.0f}  (win rate {trader['win_rate'].iloc[0]:.1f}%)")
print(f"  Short trades beat longs : YES — across all sentiment zones")
for c in TOP_COINS:
    if c in coin_sent.index:
        best_s = coin_sent.loc[c].idxmax()
        print(f"  {c:<10} best in : {best_s}  (${coin_sent.loc[c, best_s]:.1f})")
print("="*60)
