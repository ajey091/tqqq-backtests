"""
Generate JSON data for the Plotly dashboard.

Imports all simulation/strategy logic from leveraged_etf_backtest.py,
runs XIRR + drawdown sensitivity analysis and portfolio time series,
and writes dashboard/data.json.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from leveraged_etf_backtest import (
    load_index, simulate_prices, dca_strategy, ninesig_strategy,
    ma200_strategy, _strategy_xirr, _max_drawdown_pct, first_trading_days,
    SP500_ETFS, NASDAQ_ETFS, MONTHLY_DCA,
)

BASE_DIR = Path(__file__).parent

# Strategy definitions (same as cagr_sensitivity)
SP_STRAT_DEFS = [
    ("DCA SPY",        lambda df: dca_strategy(df, "spy")),
    ("DCA SSO",        lambda df: dca_strategy(df, "sso")),
    ("DCA UPRO",       lambda df: dca_strategy(df, "upro")),
    ("9sig UPRO+cash", lambda df: ninesig_strategy(df, "upro")),
    ("200MA SSO",      lambda df: ma200_strategy(df, "spy", "sso")),
    ("200MA UPRO",     lambda df: ma200_strategy(df, "spy", "upro")),
]
NDX_STRAT_DEFS = [
    ("DCA QQQ",        lambda df: dca_strategy(df, "qqq")),
    ("DCA QLD",        lambda df: dca_strategy(df, "qld")),
    ("DCA TQQQ",       lambda df: dca_strategy(df, "tqqq")),
    ("9sig TQQQ+cash", lambda df: ninesig_strategy(df, "tqqq")),
    ("200MA QLD",      lambda df: ma200_strategy(df, "qqq", "qld")),
    ("200MA TQQQ",     lambda df: ma200_strategy(df, "qqq", "tqqq")),
]
ALL_STRAT_NAMES = [n for n, _ in SP_STRAT_DEFS] + [n for n, _ in NDX_STRAT_DEFS]

# Colors matching the matplotlib backtest
STRATEGY_COLORS = {
    "DCA SPY":        "#1A56DB",
    "DCA SSO":        "#E04F39",
    "DCA UPRO":       "#16A34A",
    "9sig UPRO+cash": "#CA8A04",
    "200MA SSO":      "#B45309",
    "200MA UPRO":     "#9333EA",
    "DCA QQQ":        "#06B6D4",
    "DCA QLD":        "#F97316",
    "DCA TQQQ":       "#EC4899",
    "9sig TQQQ+cash": "#84CC16",
    "200MA QLD":      "#0E7490",
    "200MA TQQQ":     "#6366F1",
}

PORTFOLIO_START_YEARS = [1986, 1990, 1995, 2000, 2005, 2010]


def load_data():
    """Load and simulate all ETF prices."""
    print("Loading S&P 500 data...")
    sp_df = load_index("GSPC_daily.csv", "gspc_close")
    sp_df = simulate_prices(sp_df, "gspc_close", SP500_ETFS)
    print(f"  {len(sp_df)} trading days")

    print("Loading Nasdaq-100 data...")
    ndx_df = load_index("NDX_daily.csv", "ndx_close")
    ndx_df = simulate_prices(ndx_df, "ndx_close", NASDAQ_ETFS)
    print(f"  {len(ndx_df)} trading days")

    return sp_df, ndx_df


def compute_xirr_drawdown(sp_df, ndx_df, start_years=range(1985, 2016)):
    """Compute XIRR and max drawdown for all strategies across start years."""
    print("Computing XIRR & drawdown sensitivity...")
    xirr_data = {name: [] for name in ALL_STRAT_NAMES}
    dd_data = {name: [] for name in ALL_STRAT_NAMES}
    years = []

    for yr in start_years:
        cutoff = pd.Timestamp(f"{yr}-01-01")
        sp_cut = sp_df[sp_df["date"] >= cutoff].reset_index(drop=True)
        ndx_cut = ndx_df[ndx_df["date"] >= cutoff].reset_index(drop=True)

        if len(sp_cut) < 252 or len(ndx_cut) < 252:
            continue

        years.append(yr)
        print(f"  {yr}...", end=" ", flush=True)

        for name, runner in SP_STRAT_DEFS:
            values, _ = runner(sp_cut)
            xirr_data[name].append(round(_strategy_xirr(sp_cut, values) * 100, 2))
            dd_data[name].append(round(_max_drawdown_pct(values), 1))

        for name, runner in NDX_STRAT_DEFS:
            values, _ = runner(ndx_cut)
            xirr_data[name].append(round(_strategy_xirr(ndx_cut, values) * 100, 2))
            dd_data[name].append(round(_max_drawdown_pct(values), 1))

    print("done.")
    return years, xirr_data, dd_data


def compute_portfolio_series(sp_df, ndx_df, start_years=PORTFOLIO_START_YEARS):
    """Compute monthly-sampled portfolio value time series for select start years."""
    print("Computing portfolio time series...")
    series = {}

    from leveraged_etf_backtest import build_invested_line

    for yr in start_years:
        print(f"  {yr}...", end=" ", flush=True)
        cutoff = pd.Timestamp(f"{yr}-01-01")
        sp_cut = sp_df[sp_df["date"] >= cutoff].reset_index(drop=True)
        ndx_cut = ndx_df[ndx_df["date"] >= cutoff].reset_index(drop=True)

        if len(sp_cut) < 252 or len(ndx_cut) < 252:
            continue

        # Align both to the later start date so all strategies share the same range
        common_start = max(sp_cut["date"].iloc[0], ndx_cut["date"].iloc[0])
        sp_cut = sp_cut[sp_cut["date"] >= common_start].reset_index(drop=True)
        ndx_cut = ndx_cut[ndx_cut["date"] >= common_start].reset_index(drop=True)

        # Use a single shared date axis (S&P trading days, which is a superset)
        ftd = first_trading_days(sp_cut["date"])
        dates_monthly = [sp_cut["date"].iloc[idx].strftime("%Y-%m-%d") for idx in ftd]
        last_day = sp_cut["date"].iloc[-1].strftime("%Y-%m-%d")
        if dates_monthly[-1] != last_day:
            dates_monthly.append(last_day)
            ftd = np.append(ftd, len(sp_cut) - 1)

        # Nasdaq sampling: find closest index in ndx_cut for each sp_cut ftd date
        ndx_ftd = []
        for idx in ftd:
            target_date = sp_cut["date"].iloc[idx]
            # Find closest date in ndx_cut
            diffs = (ndx_cut["date"] - target_date).abs()
            ndx_ftd.append(diffs.idxmin())
        ndx_ftd = np.array(ndx_ftd)

        yr_data = {"dates": dates_monthly, "strategies": {}}

        for name, runner in SP_STRAT_DEFS:
            values, total_inv = runner(sp_cut)
            sampled = [round(float(values[idx]), 2) for idx in ftd]
            yr_data["strategies"][name] = {
                "values": sampled,
                "total_invested": round(total_inv, 2),
                "final_value": round(float(values[-1]), 2),
            }

        for name, runner in NDX_STRAT_DEFS:
            values, total_inv = runner(ndx_cut)
            sampled = [round(float(values[idx]), 2) for idx in ndx_ftd]
            yr_data["strategies"][name] = {
                "values": sampled,
                "total_invested": round(total_inv, 2),
                "final_value": round(float(values[-1]), 2),
            }

        # DCA SPY reference for this start year
        spy_values, spy_inv = dca_strategy(sp_cut, "spy")
        sampled_spy = [round(float(spy_values[idx]), 2) for idx in ftd]
        yr_data["dca_spy_ref"] = {
            "values": sampled_spy,
            "total_invested": round(spy_inv, 2),
            "final_value": round(float(spy_values[-1]), 2),
        }

        # Total invested line
        inv_line = build_invested_line(sp_cut)
        sampled_inv = [round(float(inv_line[idx]), 2) for idx in ftd]
        yr_data["invested_line"] = sampled_inv

        series[str(yr)] = yr_data

    print("done.")
    return series


def compute_summary_stats(sp_df, ndx_df, start_year=1990):
    """Compute summary stats table for a single start year."""
    cutoff = pd.Timestamp(f"{start_year}-01-01")
    sp_cut = sp_df[sp_df["date"] >= cutoff].reset_index(drop=True)
    ndx_cut = ndx_df[ndx_df["date"] >= cutoff].reset_index(drop=True)

    stats = []
    for name, runner in SP_STRAT_DEFS:
        values, total_inv = runner(sp_cut)
        xirr = _strategy_xirr(sp_cut, values) * 100
        dd = _max_drawdown_pct(values)
        multiple = values[-1] / total_inv if total_inv > 0 else 0
        stats.append({
            "name": name,
            "final_value": round(float(values[-1]), 0),
            "total_invested": round(total_inv, 0),
            "multiple": round(multiple, 1),
            "xirr": round(xirr, 1),
            "max_drawdown": round(dd, 0),
        })

    for name, runner in NDX_STRAT_DEFS:
        values, total_inv = runner(ndx_cut)
        xirr = _strategy_xirr(ndx_cut, values) * 100
        dd = _max_drawdown_pct(values)
        multiple = values[-1] / total_inv if total_inv > 0 else 0
        stats.append({
            "name": name,
            "final_value": round(float(values[-1]), 0),
            "total_invested": round(total_inv, 0),
            "multiple": round(multiple, 1),
            "xirr": round(xirr, 1),
            "max_drawdown": round(dd, 0),
        })

    return stats


def main():
    sp_df, ndx_df = load_data()

    years, xirr_data, dd_data = compute_xirr_drawdown(sp_df, ndx_df)
    portfolio_series = compute_portfolio_series(sp_df, ndx_df)
    summary_stats = compute_summary_stats(sp_df, ndx_df, start_year=1990)

    data = {
        "strategy_names": ALL_STRAT_NAMES,
        "strategy_colors": STRATEGY_COLORS,
        "sensitivity": {
            "years": years,
            "xirr": xirr_data,
            "drawdown": dd_data,
        },
        "portfolio_series": portfolio_series,
        "summary_stats": summary_stats,
        "config": {
            "monthly_dca": MONTHLY_DCA,
            "portfolio_start_years": PORTFOLIO_START_YEARS,
        },
    }

    out_path = BASE_DIR / "dashboard" / "data.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
