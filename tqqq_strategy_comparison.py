"""
TQQQ Alternative Strategy Comparison

Compares DCA TQQQ against creative alternatives:
  1. DCA after 50% pullback — sit in cash until TQQQ drops 50% from ATH, then DCA
  2. Crash accelerator — normal DCA but 3× contribution when TQQQ is >30% below ATH
  3. Drawdown deployer — accumulate cash, only buy TQQQ when it's >40% below ATH
  4. MA crossover DCA — DCA into TQQQ only when QQQ > 200-day MA, else cash
  5. Value averaging — target 12% annual portfolio growth, adjust contributions accordingly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

from leveraged_etf_backtest import (
    load_index, simulate_prices, first_trading_days,
    NASDAQ_ETFS, MONTHLY_DCA, _strategy_xirr, _max_drawdown_pct, _fmt_dollars,
)

BASE_DIR = Path(__file__).parent


# ── Strategies ────────────────────────────────────────────────────────────

def dca_tqqq(df):
    """Baseline: plain DCA $1000/month into TQQQ."""
    ftd = first_trading_days(df["date"])
    prices = df["tqqq"].values
    portfolio = np.zeros(len(df))
    shares = 0.0
    invested = 0.0

    dca_idx = 0
    for i in range(len(df)):
        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            shares += MONTHLY_DCA / prices[i]
            invested += MONTHLY_DCA
            dca_idx += 1
        portfolio[i] = shares * prices[i]

    return portfolio, invested


def dca_after_50pct_pullback(df):
    """Sit in cash until TQQQ drops 50% from its ATH, then start DCA."""
    ftd = first_trading_days(df["date"])
    prices = df["tqqq"].values
    portfolio = np.zeros(len(df))
    shares = 0.0
    cash = 0.0
    invested = 0.0
    triggered = False
    ath = prices[0]

    dca_idx = 0
    for i in range(len(df)):
        ath = max(ath, prices[i])
        drawdown = (prices[i] - ath) / ath

        if not triggered and drawdown <= -0.50:
            triggered = True

        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            invested += MONTHLY_DCA
            if triggered:
                # Deploy any accumulated cash + new contribution
                buy_amount = cash + MONTHLY_DCA
                shares += buy_amount / prices[i]
                cash = 0.0
            else:
                cash += MONTHLY_DCA
            dca_idx += 1

        portfolio[i] = shares * prices[i] + cash

    return portfolio, invested


def crash_accelerator(df, normal=MONTHLY_DCA, multiplier=3.0, threshold=-0.30):
    """DCA normally, but invest 3× when TQQQ is >30% below ATH."""
    ftd = first_trading_days(df["date"])
    prices = df["tqqq"].values
    portfolio = np.zeros(len(df))
    shares = 0.0
    invested = 0.0
    ath = prices[0]

    dca_idx = 0
    for i in range(len(df)):
        ath = max(ath, prices[i])
        drawdown = (prices[i] - ath) / ath

        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            amount = normal * multiplier if drawdown <= threshold else normal
            shares += amount / prices[i]
            invested += amount
            dca_idx += 1

        portfolio[i] = shares * prices[i]

    return portfolio, invested


def drawdown_deployer(df, threshold=-0.40):
    """Accumulate cash via DCA. Only buy TQQQ when it's >40% below ATH."""
    ftd = first_trading_days(df["date"])
    prices = df["tqqq"].values
    portfolio = np.zeros(len(df))
    shares = 0.0
    cash = 0.0
    invested = 0.0
    ath = prices[0]

    dca_idx = 0
    for i in range(len(df)):
        ath = max(ath, prices[i])
        drawdown = (prices[i] - ath) / ath

        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            invested += MONTHLY_DCA
            if drawdown <= threshold:
                buy_amount = cash + MONTHLY_DCA
                shares += buy_amount / prices[i]
                cash = 0.0
            else:
                cash += MONTHLY_DCA
            dca_idx += 1

        portfolio[i] = shares * prices[i] + cash

    return portfolio, invested


def ma_crossover_dca(df):
    """DCA into TQQQ only when QQQ > 200-day MA, else hold cash."""
    ftd = first_trading_days(df["date"])
    qqq_prices = df["qqq"].values
    tqqq_prices = df["tqqq"].values
    ma200 = pd.Series(qqq_prices).rolling(200, min_periods=1).mean().values

    portfolio = np.zeros(len(df))
    shares = 0.0
    cash = 0.0
    invested = 0.0

    dca_idx = 0
    for i in range(len(df)):
        above_ma = qqq_prices[i] > ma200[i]

        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            invested += MONTHLY_DCA
            if above_ma:
                shares += MONTHLY_DCA / tqqq_prices[i]
            else:
                cash += MONTHLY_DCA
            dca_idx += 1

        # On MA cross back up, deploy accumulated cash
        if above_ma and cash > 0:
            shares += cash / tqqq_prices[i]
            cash = 0.0

        portfolio[i] = shares * tqqq_prices[i] + cash

    return portfolio, invested


def value_averaging(df, target_annual_growth=0.12):
    """Target 12%/yr portfolio growth. Contribute more when behind, less when ahead.
    Minimum contribution = $0 (never sell), maximum = 3× normal DCA."""
    ftd = first_trading_days(df["date"])
    prices = df["tqqq"].values
    daily_growth = (1 + target_annual_growth) ** (1 / 252)

    portfolio = np.zeros(len(df))
    shares = 0.0
    invested = 0.0
    target_value = 0.0
    started = False

    dca_idx = 0
    for i in range(len(df)):
        if started:
            target_value *= daily_growth

        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            target_value += MONTHLY_DCA
            started = True

            current_value = shares * prices[i]
            deficit = target_value - current_value
            contribution = max(0, min(deficit, MONTHLY_DCA * 3))
            shares += contribution / prices[i]
            invested += contribution
            dca_idx += 1

        portfolio[i] = shares * prices[i]

    return portfolio, invested


def ma_band_switch(df, band=0.02):
    """Sell TQQQ when QQQ < 200MA−2%, buy when QQQ > 200MA+2%. DCA into
    whichever side (TQQQ or cash) we're currently on."""
    ftd = first_trading_days(df["date"])
    qqq_prices = df["qqq"].values
    tqqq_prices = df["tqqq"].values
    ma200 = pd.Series(qqq_prices).rolling(200, min_periods=1).mean().values

    portfolio = np.zeros(len(df))
    shares = 0.0
    cash = 0.0
    invested = 0.0
    in_tqqq = True  # start invested (first 200 days MA is building)

    dca_idx = 0
    for i in range(len(df)):
        ratio = qqq_prices[i] / ma200[i] - 1  # % above/below MA

        # Switch signals with hysteresis band
        if in_tqqq and ratio < -band:
            # Sell all TQQQ to cash
            cash += shares * tqqq_prices[i]
            shares = 0.0
            in_tqqq = False
        elif not in_tqqq and ratio > band:
            # Buy TQQQ with all cash
            if tqqq_prices[i] > 0:
                shares += cash / tqqq_prices[i]
            cash = 0.0
            in_tqqq = True

        # DCA
        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            invested += MONTHLY_DCA
            if in_tqqq:
                shares += MONTHLY_DCA / tqqq_prices[i]
            else:
                cash += MONTHLY_DCA
            dca_idx += 1

        portfolio[i] = shares * tqqq_prices[i] + cash

    return portfolio, invested


# ── Run comparison ────────────────────────────────────────────────────────

def ma200_tqqq(df):
    """200MA TQQQ: hold TQQQ when QQQ > 200MA, sell to cash when below. (Original strategy)"""
    from leveraged_etf_backtest import ma200_strategy
    return ma200_strategy(df, "qqq", "tqqq")


STRATEGIES = {
    "200MA TQQQ (baseline)":    ma200_tqqq,
    "MA band ±2% switch":       ma_band_switch,
    "DCA TQQQ (ref)":           dca_tqqq,
}

COLORS = {
    "200MA TQQQ (baseline)":    "#6366F1",
    "MA band ±2% switch":       "#EF4444",
    "DCA TQQQ (ref)":           "#64748b",
}

BASELINE_KEY = "200MA TQQQ (baseline)"

START_YEARS = [1985, 1990, 1995, 2000, 2005, 2010]


def run_comparison():
    print("Loading Nasdaq-100 data...")
    ndx_df = load_index("NDX_daily.csv", "ndx_close")
    ndx_df = simulate_prices(ndx_df, "ndx_close", NASDAQ_ETFS)
    print(f"  {len(ndx_df)} trading days: {ndx_df.date.iloc[0].date()} → {ndx_df.date.iloc[-1].date()}")

    # ── Portfolio growth plots for each start year ──
    for start_year in START_YEARS:
        cutoff = pd.Timestamp(f"{start_year}-01-01")
        df = ndx_df[ndx_df["date"] >= cutoff].reset_index(drop=True)
        if len(df) < 252:
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])

        print(f"\n{'='*80}")
        print(f"  FROM {start_year}  ({(df.date.iloc[-1] - df.date.iloc[0]).days / 365.25:.1f} years)")
        print(f"{'='*80}")
        print(f"  {'Strategy':<28s} {'Final':>12s} {'Invested':>12s} {'Multiple':>10s} {'XIRR':>8s} {'Max DD':>8s}")
        print(f"  {'-'*78}")

        for name, runner in STRATEGIES.items():
            values, invested = runner(df)
            xirr = _strategy_xirr(df, values) * 100
            dd = _max_drawdown_pct(values)
            multiple = values[-1] / invested if invested > 0 else 0

            print(f"  {name:<28s} ${values[-1]:>11,.0f} ${invested:>11,.0f} {multiple:>9.1f}× {xirr:>7.1f}% {dd:>7.0f}%")

            nonzero = values > 0
            is_baseline = name == BASELINE_KEY
            ax1.semilogy(df["date"][nonzero], values[nonzero],
                         label=f"{name}  ({xirr:.1f}%, DD {dd:.0f}%)",
                         linewidth=2.5 if is_baseline else 2,
                         color=COLORS[name],
                         linestyle="-" if is_baseline else "--",
                         alpha=1.0 if is_baseline else 0.85)

        # Invested line
        ftd = first_trading_days(df["date"])
        inv_line = np.zeros(len(df))
        cum = 0.0
        di = 0
        for i in range(len(df)):
            if di < len(ftd) and i == ftd[di]:
                cum += MONTHLY_DCA
                di += 1
            inv_line[i] = cum
        nz = inv_line > 0
        ax1.semilogy(df["date"][nz], inv_line[nz], color="gray", linestyle=":", linewidth=1.5, label="Total Invested (std)")

        ax1.set_title(f"TQQQ Strategy Comparison from {start_year}", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Value (log)")
        ax1.yaxis.set_major_formatter(FuncFormatter(_fmt_dollars))
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ── Ratio subplot: each strategy / DCA TQQQ baseline ──
        baseline_values, _ = dca_tqqq(df)
        for name, runner in STRATEGIES.items():
            if "baseline" in name:
                continue
            values, _ = runner(df)
            # Compute ratio where both are nonzero
            mask = (baseline_values > 0) & (values > 0)
            ratio = np.ones(len(df))
            ratio[mask] = values[mask] / baseline_values[mask]
            ax2.plot(df["date"][mask], ratio[mask], label=name,
                     linewidth=2, color=COLORS[name], linestyle="--", alpha=0.85)

        ax2.axhline(1.0, color="#EC4899", linewidth=2, linestyle="-", alpha=0.5)
        ax2.set_title("Ratio vs DCA TQQQ (>1 = outperforming)", fontsize=11)
        ax2.set_ylabel("Ratio")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"tqqq_comparison_{start_year}.png"
        fig.savefig(BASE_DIR / fname, dpi=200)
        plt.close(fig)
        print(f"  Saved: {fname}")

    # ── XIRR sensitivity across start years ──
    print(f"\n\n{'='*100}")
    print("XIRR SENSITIVITY: TQQQ strategies by start year")
    print(f"{'='*100}")

    all_years = list(range(1985, 2016))
    header = f"  {'Year':<8s}" + "".join(f"{n[:22]:<24s}" for n in STRATEGIES.keys())
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    xirr_by_strat = {name: [] for name in STRATEGIES}

    for yr in all_years:
        cutoff = pd.Timestamp(f"{yr}-01-01")
        df = ndx_df[ndx_df["date"] >= cutoff].reset_index(drop=True)
        if len(df) < 252:
            continue

        row = f"  {yr:<8d}"
        for name, runner in STRATEGIES.items():
            values, _ = runner(df)
            xirr = _strategy_xirr(df, values) * 100
            xirr_by_strat[name].append(xirr)
            row += f"{xirr:>7.1f}%{'':<16s}"
        print(row)

    print(f"  {'-'*(len(header)-2)}")
    for stat, fn in [("AVG", np.mean), ("MEDIAN", np.median), ("WORST", np.min), ("BEST", np.max)]:
        row = f"  {stat:<8s}"
        for name in STRATEGIES:
            v = fn(xirr_by_strat[name])
            row += f"{v:>7.1f}%{'':<16s}"
        print(row)

    print(f"\n  Note: 'Crash accelerator' invests 3× during >30% drawdowns (higher total invested).")
    print(f"  Note: 'Drawdown deployer' only buys during >40% drawdowns (cash drag in bull markets).")


if __name__ == "__main__":
    run_comparison()
