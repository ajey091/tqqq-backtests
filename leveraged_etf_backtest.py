"""
Leveraged ETF Backtest: S&P 500 + Nasdaq-100

Simulates leveraged ETF prices from index data and backtests 5 DCA strategies:
  S&P 500: SPY (1×), SSO (2×), UPRO (3×)  from ^GSPC (1928–2026)
  Nasdaq:  QQQ (1×), QLD (2×), TQQQ (3×)  from ^NDX  (1985–2026)

Note: Both indices are price-only (no dividends), understating returns by ~2%/yr.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Configuration ──────────────────────────────────────────────────────────
MONTHLY_DCA = 1000.0
BORROW_COST = 0.015   # 1.5% annual borrow cost on leveraged portion
SIGNAL_GROWTH = 0.09  # 9% per quarter for 9sig strategy (per Jason Kelly's methodology)
LINE_WIDTH = 2.0
DPI = 300

# ETF definitions: (name, leverage, annual_expense_ratio)
SP500_ETFS = [
    ("SPY",  1, 0.0009),
    ("SSO",  2, 0.0089),
    ("UPRO", 3, 0.0091),
]
NASDAQ_ETFS = [
    ("QQQ",  1, 0.0020),
    ("QLD",  2, 0.0095),
    ("TQQQ", 3, 0.0086),
]


# ── Risk-free rate (fed funds) ────────────────────────────────────────────
_FF_CACHE = None

def load_fed_funds_rate():
    """Load daily fed funds rate and return as a Series indexed by date."""
    global _FF_CACHE
    if _FF_CACHE is not None:
        return _FF_CACHE
    ff_path = BASE_DIR / "fed_funds_rate.csv"
    if not ff_path.exists():
        _FF_CACHE = pd.Series(dtype=float)
        return _FF_CACHE
    ff = pd.read_csv(ff_path)
    ff.columns = ["date", "rate"]
    ff["date"] = pd.to_datetime(ff["date"])
    ff["rate"] = pd.to_numeric(ff["rate"], errors="coerce")
    ff = ff.dropna().set_index("date")["rate"]
    _FF_CACHE = ff
    return ff


def build_daily_cash_rate(dates):
    """Build array of daily cash multipliers (1 + r_daily) aligned to dates."""
    ff = load_fed_funds_rate()
    daily_mult = np.ones(len(dates))
    if ff.empty:
        return daily_mult
    # Reindex ff to trading dates using forward-fill
    ff_aligned = ff.reindex(dates, method="ffill")
    # Convert annual % rate to daily multiplier
    valid = ff_aligned.notna()
    daily_mult[valid] = 1 + ff_aligned[valid].values / 100 / 365
    return daily_mult


# ── Load index data ───────────────────────────────────────────────────────
def load_index(filename, col_name):
    df = pd.read_csv(BASE_DIR / filename, parse_dates=["Date"])
    df = df[["Date", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
    df.columns = ["date", col_name]
    return df


# ── Simulate leveraged ETF prices ─────────────────────────────────────────
def simulate_prices(df, index_col, etf_defs):
    idx = df[index_col].values
    daily_ret = np.diff(idx) / idx[:-1]
    n = len(idx)

    start_price = idx[0] / 10  # arbitrary starting price

    for name, leverage, er in etf_defs:
        daily_cost = (er + BORROW_COST * max(leverage - 1, 0)) / 252
        lev_ret = leverage * daily_ret - daily_cost
        if leverage > 1:
            lev_ret = np.maximum(lev_ret, -1.0)  # floor at -100%

        prices = np.empty(n)
        prices[0] = start_price
        prices[1:] = start_price * np.cumprod(1 + lev_ret)
        df[name.lower()] = prices

    return df


# ── First trading day of each month ───────────────────────────────────────
def first_trading_days(dates):
    s = pd.Series(dates)
    ym = s.dt.to_period("M")
    mask = ym != ym.shift(1)
    return s[mask].index.values


# ── Strategy helpers ──────────────────────────────────────────────────────
def dca_strategy(df, price_col):
    """Simple DCA: buy price_col on first trading day of each month."""
    ftd = first_trading_days(df["date"])
    prices = df[price_col].values

    portfolio_value = np.zeros(len(df))
    shares = 0.0
    total_invested = 0.0

    dca_idx = 0
    for i in range(len(df)):
        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            shares += MONTHLY_DCA / prices[i]
            total_invested += MONTHLY_DCA
            dca_idx += 1
        portfolio_value[i] = shares * prices[i]

    return portfolio_value, total_invested


def ninesig_strategy(df, lev3_col):
    """9-signal: DCA into cash pool, quarterly rebalance 3× ETF vs cash.

    Signal line is a running target that compounds at 9% per quarter daily
    and gets $1k added on each DCA day. Cash earns the fed funds rate.
    """
    ftd = first_trading_days(df["date"])
    dates = df["date"]
    prices = df[lev3_col].values
    cash_mult = build_daily_cash_rate(dates)

    portfolio_value = np.zeros(len(df))
    cash = 0.0
    shares = 0.0
    total_invested = 0.0
    signal = 0.0  # running signal line value

    daily_growth_factor = (1 + SIGNAL_GROWTH) ** (1 / 63)  # ~63 trading days per quarter
    months = dates.dt.month.values
    years = dates.dt.year.values

    dca_idx = 0
    last_rebal_quarter = None
    started = False

    for i in range(len(df)):
        # Compound signal daily
        if started:
            signal *= daily_growth_factor

        # Cash earns risk-free rate daily
        if cash > 0:
            cash *= cash_mult[i]

        # DCA into cash pool on first trading day of month
        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            cash += MONTHLY_DCA
            total_invested += MONTHLY_DCA
            signal += MONTHLY_DCA  # new contribution adds to signal target
            started = True
            dca_idx += 1

        # Quarterly rebalance
        current_quarter = (years[i], (months[i] - 1) // 3)
        if current_quarter != last_rebal_quarter and started:
            last_rebal_quarter = current_quarter

            etf_value = shares * prices[i]

            if etf_value < signal:
                # ETF below signal — buy with cash to bring ETF up to signal
                deficit = signal - etf_value
                buy_amount = min(deficit, cash)
                if buy_amount > 0 and prices[i] > 0:
                    shares += buy_amount / prices[i]
                    cash -= buy_amount
            elif etf_value > signal:
                # ETF above signal — sell down to signal, proceeds to cash
                surplus = etf_value - signal
                if prices[i] > 0:
                    sell_shares = min(surplus / prices[i], shares)
                    shares -= sell_shares
                    cash += sell_shares * prices[i]

        portfolio_value[i] = shares * prices[i] + cash

    return portfolio_value, total_invested


def ma200_strategy(df, base_col, lev3_col):
    """200-day MA: hold 3× ETF when base > 200-day SMA, else cash (earning fed funds rate)."""
    ftd = first_trading_days(df["date"])
    base_prices = df[base_col].values
    lev3_prices = df[lev3_col].values
    cash_mult = build_daily_cash_rate(df["date"])

    base_sma200 = pd.Series(base_prices).rolling(200, min_periods=1).mean().values

    portfolio_value = np.zeros(len(df))
    shares = 0.0
    cash = 0.0
    total_invested = 0.0
    invested = True  # default to invested for first 200 days

    dca_idx = 0
    for i in range(len(df)):
        should_hold = base_prices[i] > base_sma200[i]

        if should_hold and not invested:
            if cash > 0 and lev3_prices[i] > 0:
                shares += cash / lev3_prices[i]
                cash = 0.0
            invested = True
        elif not should_hold and invested:
            if shares > 0:
                cash += shares * lev3_prices[i]
                shares = 0.0
            invested = False

        # Cash earns risk-free rate daily
        if cash > 0:
            cash *= cash_mult[i]

        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            total_invested += MONTHLY_DCA
            if invested:
                if lev3_prices[i] > 0:
                    shares += MONTHLY_DCA / lev3_prices[i]
            else:
                cash += MONTHLY_DCA
            dca_idx += 1

        portfolio_value[i] = shares * lev3_prices[i] + cash

    return portfolio_value, total_invested


def ninesig_bond_strategy(df, lev3_col, bond_col):
    """9-signal with bonds: same as ninesig_strategy but holds bonds instead of cash.

    The 'cash' portion is invested in a bond fund (e.g. VBMFX/AGG).
    Quarterly rebalancing sells/buys between the leveraged ETF and bond fund.
    """
    ftd = first_trading_days(df["date"])
    dates = df["date"]
    prices = df[lev3_col].values
    bond_prices = df[bond_col].values

    portfolio_value = np.zeros(len(df))
    bond_shares = 0.0
    etf_shares = 0.0
    total_invested = 0.0
    signal = 0.0

    daily_growth_factor = (1 + SIGNAL_GROWTH) ** (1 / 63)  # ~63 trading days per quarter
    months = dates.dt.month.values
    years = dates.dt.year.values

    dca_idx = 0
    last_rebal_quarter = None
    started = False

    for i in range(len(df)):
        if started:
            signal *= daily_growth_factor

        # DCA into bonds on first trading day of month
        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            total_invested += MONTHLY_DCA
            signal += MONTHLY_DCA
            if bond_prices[i] > 0:
                bond_shares += MONTHLY_DCA / bond_prices[i]
            started = True
            dca_idx += 1

        # Quarterly rebalance
        current_quarter = (years[i], (months[i] - 1) // 3)
        if current_quarter != last_rebal_quarter and started:
            last_rebal_quarter = current_quarter

            etf_value = etf_shares * prices[i]
            bond_value = bond_shares * bond_prices[i]

            if etf_value < signal:
                # Buy ETF with bond proceeds
                deficit = signal - etf_value
                buy_amount = min(deficit, bond_value)
                if buy_amount > 0:
                    if bond_prices[i] > 0:
                        bond_shares -= buy_amount / bond_prices[i]
                    if prices[i] > 0:
                        etf_shares += buy_amount / prices[i]
            elif etf_value > signal:
                # Sell ETF surplus into bonds
                surplus = etf_value - signal
                if prices[i] > 0:
                    sell_shares = min(surplus / prices[i], etf_shares)
                    etf_shares -= sell_shares
                    proceeds = sell_shares * prices[i]
                    if bond_prices[i] > 0:
                        bond_shares += proceeds / bond_prices[i]

        portfolio_value[i] = etf_shares * prices[i] + bond_shares * bond_prices[i]

    return portfolio_value, total_invested


def ma200_9sig_strategy(df, base_col, lev3_col):
    """200MA + 9sig hybrid: run 9sig rebalancing only when base > 200MA, else all cash.

    When base index is above 200MA: DCA flows into the 9sig cash/ETF pool and
    quarterly rebalancing occurs normally.
    When base drops below 200MA: sell all ETF shares to cash, suspend rebalancing.
    DCA contributions still accumulate in cash. Signal line keeps compounding.
    When base crosses back above 200MA: resume 9sig with existing cash + signal.
    Cash earns the fed funds rate.
    """
    ftd = first_trading_days(df["date"])
    dates = df["date"]
    base_prices = df[base_col].values
    prices = df[lev3_col].values
    cash_mult = build_daily_cash_rate(dates)
    base_sma200 = pd.Series(base_prices).rolling(200, min_periods=1).mean().values

    portfolio_value = np.zeros(len(df))
    cash = 0.0
    shares = 0.0
    total_invested = 0.0
    signal = 0.0

    daily_growth_factor = (1 + SIGNAL_GROWTH) ** (1 / 63)  # ~63 trading days per quarter
    months = dates.dt.month.values
    years = dates.dt.year.values

    dca_idx = 0
    last_rebal_quarter = None
    started = False
    above_ma = True  # start invested (like ma200_strategy)

    for i in range(len(df)):
        # Compound signal daily regardless of MA state
        if started:
            signal *= daily_growth_factor

        # Cash earns risk-free rate daily
        if cash > 0:
            cash *= cash_mult[i]

        # Check MA crossover
        new_above = base_prices[i] > base_sma200[i]
        if above_ma and not new_above:
            # Crossed below — liquidate ETF to cash
            if shares > 0:
                cash += shares * prices[i]
                shares = 0.0
            above_ma = False
        elif not above_ma and new_above:
            above_ma = True
            # Don't immediately buy — let next rebalance handle allocation

        # DCA into cash pool on first trading day of month
        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            cash += MONTHLY_DCA
            total_invested += MONTHLY_DCA
            signal += MONTHLY_DCA
            started = True
            dca_idx += 1

        # Quarterly rebalance — only when above 200MA
        if above_ma and started:
            current_quarter = (years[i], (months[i] - 1) // 3)
            if current_quarter != last_rebal_quarter:
                last_rebal_quarter = current_quarter
                etf_value = shares * prices[i]

                if etf_value < signal:
                    deficit = signal - etf_value
                    buy_amount = min(deficit, cash)
                    if buy_amount > 0 and prices[i] > 0:
                        shares += buy_amount / prices[i]
                        cash -= buy_amount
                elif etf_value > signal:
                    surplus = etf_value - signal
                    if prices[i] > 0:
                        sell_shares = min(surplus / prices[i], shares)
                        shares -= sell_shares
                        cash += sell_shares * prices[i]

        portfolio_value[i] = shares * prices[i] + cash

    return portfolio_value, total_invested


# ── Summary stats ─────────────────────────────────────────────────────────
def compute_stats(name, portfolio_values, total_invested, dates):
    final_val = portfolio_values[-1]
    multiple = final_val / total_invested if total_invested > 0 else 0

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    first_nonzero = portfolio_values[portfolio_values > 0]
    cagr = (final_val / first_nonzero[0]) ** (1 / years) - 1 if years > 0 and len(first_nonzero) > 0 else 0

    running_max = np.maximum.accumulate(portfolio_values)
    running_max[running_max == 0] = 1
    drawdowns = (portfolio_values - running_max) / running_max
    max_dd = drawdowns.min()

    return {
        "Strategy": name,
        "Final Value": f"${final_val:,.0f}",
        "Total Invested": f"${total_invested:,.0f}",
        "Multiple": f"{multiple:.1f}×",
        "~CAGR": f"{cagr:.1%}",
        "Max Drawdown": f"{max_dd:.1%}",
    }


def print_summary(title, strats, dates):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    headers = ["Strategy", "Final Value", "Total Invested", "Multiple", "~CAGR", "Max Drawdown"]
    widths = [20, 20, 16, 10, 10, 14]
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for name, (values, total_inv) in strats.items():
        s = compute_stats(name, values, total_inv, dates)
        print("".join(str(s[h]).ljust(w) for h, w in zip(headers, widths)))
    print("=" * 90)


def build_invested_line(df):
    ftd = first_trading_days(df["date"])
    invested_line = np.zeros(len(df))
    cumulative = 0.0
    dca_idx = 0
    for i in range(len(df)):
        if dca_idx < len(ftd) and i == ftd[dca_idx]:
            cumulative += MONTHLY_DCA
            dca_idx += 1
        invested_line[i] = cumulative
    return invested_line


# ── Run backtest for one index family ─────────────────────────────────────
def run_backtest(df, label, etf_defs, base_col, lev2_col, lev3_col):
    etf_names = [e[0] for e in etf_defs]
    print(f"\nSimulating {' / '.join(etf_names)} prices...")
    df = simulate_prices(df, base_col + "_close", etf_defs)

    for name, _, _ in etf_defs:
        col = name.lower()
        print(f"  {name}: {df[col].iloc[0]:.4f} → {df[col].iloc[-1]:.4f}")

    # ── Plot 1: Price history ──
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    for name, lev, _ in etf_defs:
        col = name.lower()
        ax1.semilogy(df["date"], df[col], label=f"{name} ({lev}×)", linewidth=LINE_WIDTH)
    ax1.set_title(f"Simulated {' / '.join(etf_names)} Price History (log scale)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (log)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    price_file = f"leveraged_etf_prices_{label}.png"
    fig1.savefig(BASE_DIR / price_file, dpi=DPI)
    plt.close(fig1)
    print(f"Saved: {price_file}")

    # ── Run strategies ──
    base_1x = etf_defs[0][0].lower()
    lev2 = etf_defs[1][0].lower()
    lev3 = etf_defs[2][0].lower()

    print(f"\nRunning 7 strategies ($1,000/month DCA)...")
    strats = {}
    print(f"  1/7 DCA {etf_names[0]}...")
    strats[f"DCA {etf_names[0]}"] = dca_strategy(df, base_1x)
    print(f"  2/7 DCA {etf_names[1]}...")
    strats[f"DCA {etf_names[1]}"] = dca_strategy(df, lev2)
    print(f"  3/7 DCA {etf_names[2]}...")
    strats[f"DCA {etf_names[2]}"] = dca_strategy(df, lev3)
    print(f"  4/7 9sig {etf_names[2]}+cash...")
    strats[f"9sig {etf_names[2]}+cash"] = ninesig_strategy(df, lev3)
    print(f"  5/7 200MA {etf_names[1]}...")
    strats[f"200MA {etf_names[1]}"] = ma200_strategy(df, base_1x, lev2)
    print(f"  6/7 200MA {etf_names[2]}...")
    strats[f"200MA {etf_names[2]}"] = ma200_strategy(df, base_1x, lev3)
    print(f"  7/7 DCA SPY (ref)...")

    # ── Plot 2: Portfolio values ──
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for (name, (values, _)), color in zip(strats.items(), colors):
        nonzero = values > 0
        ax2.semilogy(df["date"][nonzero], values[nonzero], label=name,
                     linewidth=LINE_WIDTH, color=color)

    # SPY reference line (simulated from GSPC for both; for NDX we load GSPC separately)
    spy_values, spy_invested = _get_spy_dca(df)
    if spy_values is not None:
        nonzero = spy_values > 0
        ax2.semilogy(df["date"][nonzero], spy_values[nonzero], label="DCA SPY (ref)",
                     linewidth=LINE_WIDTH, color="gray", linestyle="-.", alpha=0.7)

    # Total invested line
    invested_line = build_invested_line(df)
    nonzero = invested_line > 0
    ax2.semilogy(df["date"][nonzero], invested_line[nonzero], label="Total Invested",
                 linewidth=LINE_WIDTH, color="black", linestyle="--")

    ax2.set_title(f"Portfolio Values: {label.upper()} Strategies ($1,000/month, log scale)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Value (log)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    strat_file = f"leveraged_etf_strategies_{label}.png"
    fig2.savefig(BASE_DIR / strat_file, dpi=DPI)
    plt.close(fig2)
    print(f"Saved: {strat_file}")

    print_summary(f"{label.upper()} STRATEGY SUMMARY", strats, df["date"])
    return df, strats


# ── SPY reference for strategy plots ──────────────────────────────────────
_spy_cache = {}

def _get_spy_dca(df):
    """Get DCA SPY portfolio values aligned to df's date range."""
    if "spy" in df.columns:
        return dca_strategy(df, "spy")
    # For NDX: simulate SPY from GSPC, then align to NDX dates
    if "spy_aligned" not in _spy_cache:
        try:
            gspc_df = load_index("GSPC_daily.csv", "gspc_close")
            gspc_df = simulate_prices(gspc_df, "gspc_close", [("spy", 1, 0.0009)])
            _spy_cache["gspc_df"] = gspc_df
        except FileNotFoundError:
            return None, None

    gspc_df = _spy_cache.get("gspc_df")
    if gspc_df is None:
        return None, None

    # Merge SPY prices onto df's dates
    merged = df[["date"]].merge(gspc_df[["date", "spy"]], on="date", how="left")
    merged["spy"] = merged["spy"].ffill()

    if merged["spy"].isna().all():
        return None, None

    temp_df = df.copy()
    temp_df["spy_ref"] = merged["spy"].values
    return dca_strategy(temp_df, "spy_ref")


# ── Combined 1990+ plot ───────────────────────────────────────────────────
def _cagr(portfolio_values, dates):
    """Return raw CAGR float for a portfolio series."""
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    first_nonzero = portfolio_values[portfolio_values > 0]
    if years <= 0 or len(first_nonzero) == 0:
        return 0.0
    return (portfolio_values[-1] / first_nonzero[0]) ** (1 / years) - 1


def _fmt_dollars(x, _):
    """Format y-axis tick as $XXk, $XXM, $XXB."""
    if x >= 1e9:
        return f"${x / 1e9:.0f}B"
    elif x >= 1e6:
        return f"${x / 1e6:.0f}M"
    elif x >= 1e3:
        return f"${x / 1e3:.0f}k"
    else:
        return f"${x:.0f}"


def _total_return_pct(values, total_invested):
    """Total return as percentage: (final - invested) / invested."""
    if total_invested <= 0:
        return 0.0
    return (values[-1] - total_invested) / total_invested * 100


def _max_drawdown_pct(values):
    """Max drawdown as a positive percentage."""
    running_max = np.maximum.accumulate(values)
    running_max[running_max == 0] = 1
    dd = (values - running_max) / running_max
    return dd.min() * 100  # negative number


def _fmt_final(val):
    """Format final balance compactly for annotation."""
    if val >= 1e9:
        return f"${val / 1e9:.1f}B"
    elif val >= 1e6:
        return f"${val / 1e6:.1f}M"
    elif val >= 1e3:
        return f"${val / 1e3:.0f}k"
    else:
        return f"${val:,.0f}"


def _plot_combined(ax, sp90, sp_strats, ndx90, ndx_strats, log_scale,
                   total_invested, spy_total_return_pct, start_year=1990):
    """Plot all strategies on a given axis."""
    from matplotlib.ticker import FuncFormatter

    # 12 distinct colors — no repeats
    # S&P: solid lines
    sp_styles = {
        "DCA SPY":        {"color": "#1A56DB"},  # strong blue
        "DCA SSO":        {"color": "#E04F39"},  # red-orange
        "DCA UPRO":       {"color": "#16A34A"},  # green
        "9sig UPRO+cash": {"color": "#CA8A04"},  # dark gold
        "200MA SSO":      {"color": "#B45309"},  # brown
        "200MA UPRO":     {"color": "#9333EA"},  # purple
    }
    # Nasdaq: markers every ~250 days
    ndx_styles = {
        "DCA QQQ":          {"color": "#06B6D4", "marker": "o"},  # cyan
        "DCA QLD":          {"color": "#F97316", "marker": "s"},  # orange
        "DCA TQQQ":         {"color": "#EC4899", "marker": "^"},  # pink
        "9sig TQQQ+cash":   {"color": "#84CC16", "marker": "D"},  # lime
        "200MA QLD":        {"color": "#0E7490", "marker": "P"},  # dark teal
        "200MA TQQQ":       {"color": "#6366F1", "marker": "v"},  # indigo
    }

    plot_fn = ax.semilogy if log_scale else ax.plot
    mark_every = 250

    # Plot S&P strategies
    for name, (values, inv) in sp_strats.items():
        ret = _total_return_pct(values, inv)
        dd = _max_drawdown_pct(values)
        final = _fmt_final(values[-1])
        label = f"{name}  ({ret:,.0f}%, DD {dd:.0f}%) → {final}"
        nonzero = values > 0
        plot_fn(sp90["date"][nonzero], values[nonzero], label=label,
                linewidth=LINE_WIDTH, color=sp_styles[name]["color"], linestyle="-")

    # Plot Nasdaq strategies
    for name, (values, inv) in ndx_strats.items():
        ret = _total_return_pct(values, inv)
        dd = _max_drawdown_pct(values)
        final = _fmt_final(values[-1])
        label = f"{name}  ({ret:,.0f}%, DD {dd:.0f}%) → {final}"
        sty = ndx_styles.get(name, {"color": "#888888", "marker": "x"})
        nonzero = values > 0
        plot_fn(ndx90["date"][nonzero], values[nonzero], label=label,
                linewidth=LINE_WIDTH, color=sty["color"], linestyle="-",
                marker=sty["marker"], markevery=mark_every, markersize=5,
                markerfacecolor=sty["color"], markeredgecolor="white",
                markeredgewidth=0.5)

    # Dollar formatting on y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_dollars))

    # Subtitle with total invested and SPY return
    scale_label = "log" if log_scale else "linear"
    ax.set_title(
        f"All Strategies from {start_year}: S&P 500 (solid) vs Nasdaq-100 (markers) — {scale_label} scale\n"
        f"Total Invested: ${total_invested:,.0f}  |  SPY price return over period: {spy_total_return_pct:,.0f}%",
        fontsize=13)


def _run_and_plot_combined(sp_df, ndx_df, start_year):
    """Re-run all 10 strategies from start_year, plot log + linear, print summary."""
    print(f"\nRunning all strategies from {start_year} for combined plot...")
    cutoff = pd.Timestamp(f"{start_year}-01-01")
    tag = str(start_year)

    sp_cut = sp_df[sp_df["date"] >= cutoff].reset_index(drop=True)
    ndx_cut = ndx_df[ndx_df["date"] >= cutoff].reset_index(drop=True)

    sp_strats = {}
    sp_strats["DCA SPY"] = dca_strategy(sp_cut, "spy")
    sp_strats["DCA SSO"] = dca_strategy(sp_cut, "sso")
    sp_strats["DCA UPRO"] = dca_strategy(sp_cut, "upro")
    sp_strats["9sig UPRO+cash"] = ninesig_strategy(sp_cut, "upro")
    sp_strats["200MA SSO"] = ma200_strategy(sp_cut, "spy", "sso")
    sp_strats["200MA UPRO"] = ma200_strategy(sp_cut, "spy", "upro")

    ndx_strats = {}
    ndx_strats["DCA QQQ"] = dca_strategy(ndx_cut, "qqq")
    ndx_strats["DCA QLD"] = dca_strategy(ndx_cut, "qld")
    ndx_strats["DCA TQQQ"] = dca_strategy(ndx_cut, "tqqq")
    ndx_strats["9sig TQQQ+cash"] = ninesig_strategy(ndx_cut, "tqqq")
    ndx_strats["200MA QLD"] = ma200_strategy(ndx_cut, "qqq", "qld")
    ndx_strats["200MA TQQQ"] = ma200_strategy(ndx_cut, "qqq", "tqqq")

    total_invested = list(sp_strats.values())[0][1]
    spy_start = sp_cut["spy"].iloc[0]
    spy_end = sp_cut["spy"].iloc[-1]
    spy_total_return_pct = (spy_end - spy_start) / spy_start * 100

    for log_scale in [True, False]:
        fig, ax = plt.subplots(figsize=(16, 9))
        _plot_combined(ax, sp_cut, sp_strats, ndx_cut, ndx_strats,
                       log_scale=log_scale, total_invested=total_invested,
                       spy_total_return_pct=spy_total_return_pct,
                       start_year=start_year)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.legend(loc="upper left", fontsize=7.5, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        scale = "log" if log_scale else "linear"
        fname = f"leveraged_etf_combined_{tag}_{scale}.png"
        fig.savefig(BASE_DIR / fname, dpi=DPI)
        plt.close(fig)
        print(f"Saved: {fname}")

    print_summary(f"COMBINED {tag}+ S&P 500 STRATEGIES", sp_strats, sp_cut["date"])
    print_summary(f"COMBINED {tag}+ NASDAQ-100 STRATEGIES", ndx_strats, ndx_cut["date"])


# ── XIRR for proper DCA return measurement ──────────────────────────────
def _xirr(cf_dates, cf_amounts):
    """Annualized IRR via binary search. Negative=outflow, positive=inflow."""
    if len(cf_dates) < 2:
        return 0.0
    d0 = cf_dates[0]
    day_fracs = np.array([(d - d0).days / 365.25 for d in cf_dates])
    amounts = np.array(cf_amounts, dtype=float)

    def npv(rate):
        return np.sum(amounts / (1 + rate) ** day_fracs)

    lo, hi = -0.5, 5.0
    try:
        if npv(lo) * npv(hi) > 0:
            return 0.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if npv(mid) > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    except (OverflowError, FloatingPointError):
        return 0.0


def _strategy_xirr(df, portfolio_values):
    """Compute XIRR for a DCA strategy: -$1000 on each first-of-month, +final_value at end."""
    ftd = first_trading_days(df["date"])
    dates_list = [df["date"].iloc[idx] for idx in ftd]
    amounts_list = [-MONTHLY_DCA] * len(ftd)
    # Final cash flow: sell everything on the last day
    dates_list.append(df["date"].iloc[-1])
    amounts_list.append(float(portfolio_values[-1]))
    return _xirr(dates_list, amounts_list)


# ── CAGR sensitivity by start year ───────────────────────────────────────
def cagr_sensitivity(sp_df, ndx_df, start_years=range(1985, 2016)):
    """Run all strategies from each start year, compute XIRR, print table + averages."""
    print("\n\n" + "=" * 140)
    print("CAGR (XIRR) SENSITIVITY ANALYSIS: All strategies by start year (through end of data)")
    print("Each cell = annualized IRR accounting for $1,000/month DCA timing")
    print("=" * 140)

    sp_strat_defs = [
        ("DCA SPY",        lambda df: dca_strategy(df, "spy")),
        ("DCA SSO",        lambda df: dca_strategy(df, "sso")),
        ("DCA UPRO",       lambda df: dca_strategy(df, "upro")),
        ("9sig UPRO+cash", lambda df: ninesig_strategy(df, "upro")),
        ("200MA SSO",      lambda df: ma200_strategy(df, "spy", "sso")),
        ("200MA UPRO",     lambda df: ma200_strategy(df, "spy", "upro")),
    ]
    ndx_strat_defs = [
        ("DCA QQQ",          lambda df: dca_strategy(df, "qqq")),
        ("DCA QLD",          lambda df: dca_strategy(df, "qld")),
        ("DCA TQQQ",         lambda df: dca_strategy(df, "tqqq")),
        ("9sig TQQQ+cash",   lambda df: ninesig_strategy(df, "tqqq")),
        ("200MA QLD",        lambda df: ma200_strategy(df, "qqq", "qld")),
        ("200MA TQQQ",       lambda df: ma200_strategy(df, "qqq", "tqqq")),
    ]

    all_names = [n for n, _ in sp_strat_defs] + [n for n, _ in ndx_strat_defs]
    xirr_results = {name: [] for name in all_names}
    dd_results = {name: [] for name in all_names}
    year_list = []

    for yr in start_years:
        cutoff = pd.Timestamp(f"{yr}-01-01")
        sp_cut = sp_df[sp_df["date"] >= cutoff].reset_index(drop=True)
        ndx_cut = ndx_df[ndx_df["date"] >= cutoff].reset_index(drop=True)

        if len(sp_cut) < 252 or len(ndx_cut) < 252:
            continue

        year_list.append(yr)
        print(f"  {yr}...", end=" ", flush=True)

        for name, runner in sp_strat_defs:
            values, _ = runner(sp_cut)
            xirr_results[name].append(_strategy_xirr(sp_cut, values))
            dd_results[name].append(_max_drawdown_pct(values))

        for name, runner in ndx_strat_defs:
            values, _ = runner(ndx_cut)
            xirr_results[name].append(_strategy_xirr(ndx_cut, values))
            dd_results[name].append(_max_drawdown_pct(values))

    print("done.\n")

    def _print_table(title, data, fmt="{:.1%}", raw_pct=False):
        """Print a table of results. If raw_pct, values are already in % (e.g. -75.3)."""
        print(f"\n{'─'*140}")
        print(f"  {title}")
        print(f"{'─'*140}")
        col_w = 10
        name_w = 14
        header = "Start".ljust(name_w) + "".join(n[:col_w - 1].ljust(col_w) for n in all_names)
        print(header)
        print("-" * len(header))

        for i, yr in enumerate(year_list):
            row = str(yr).ljust(name_w)
            for name in all_names:
                v = data[name][i]
                if raw_pct:
                    row += f"{v:.0f}%".ljust(col_w)
                else:
                    row += f"{v:.1%}".ljust(col_w)
            print(row)

        print("-" * len(header))
        for stat_name, fn in [("AVERAGE", np.mean), ("MEDIAN", np.median),
                               ("WORST", np.min), ("BEST", np.max)]:
            row = stat_name.ljust(name_w)
            for name in all_names:
                v = fn(data[name])
                if raw_pct:
                    row += f"{v:.0f}%".ljust(col_w)
                else:
                    row += f"{v:.1%}".ljust(col_w)
            print(row)

    _print_table("ANNUALIZED RETURN (XIRR) by Start Year", xirr_results)
    _print_table("MAX DRAWDOWN by Start Year", dd_results, raw_pct=True)

    print(f"\n{'═'*140}")
    print("Note: XIRR = annualized internal rate of return accounting for monthly DCA cash flow timing.")
    print("      Max drawdown = largest peak-to-trough decline in portfolio value.")
    print("      Both indices are price-only (no dividends); real returns ~2%/yr higher.")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # ── S&P 500 ──
    print("=" * 60)
    print("S&P 500 LEVERAGED ETF BACKTEST")
    print("=" * 60)
    print("Loading GSPC data...")
    gspc_df = load_index("GSPC_daily.csv", "gspc_close")
    print(f"  {len(gspc_df)} trading days: {gspc_df.date.iloc[0].date()} to {gspc_df.date.iloc[-1].date()}")
    sp_df, sp_strats = run_backtest(gspc_df, "sp500", SP500_ETFS, "gspc", "sso", "upro")

    # ── Nasdaq-100 ──
    print("\n\n" + "=" * 60)
    print("NASDAQ-100 LEVERAGED ETF BACKTEST")
    print("=" * 60)
    ndx_file = BASE_DIR / "NDX_daily.csv"
    if not ndx_file.exists():
        print("NDX_daily.csv not found — downloading from Yahoo Finance...")
        download_ndx()

    print("Loading NDX data...")
    ndx_df = load_index("NDX_daily.csv", "ndx_close")
    print(f"  {len(ndx_df)} trading days: {ndx_df.date.iloc[0].date()} to {ndx_df.date.iloc[-1].date()}")
    ndx_df_out, ndx_strats = run_backtest(ndx_df, "nasdaq", NASDAQ_ETFS, "ndx", "qld", "tqqq")

    # ── Combined plots (re-run strategies fresh from each start year) ──
    _run_and_plot_combined(sp_df, ndx_df_out, 1990)
    _run_and_plot_combined(sp_df, ndx_df_out, 2000)

    # ── CAGR sensitivity analysis ──
    cagr_sensitivity(sp_df, ndx_df_out)

    print("\n\nNote: Both indices are price-only (no dividends). Real returns ~2%/yr higher.")
    print("      Great Depression (S&P) and dot-com bust (Nasdaq) cause massive leveraged drawdowns.")
    print("      DCA recovers via new contributions at depressed prices.")


def download_ndx():
    """Download NDX daily data via Yahoo v8 API."""
    import requests, csv, time
    from datetime import datetime

    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENDX"
    params = {"period1": "0", "period2": str(int(time.time())), "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]

    rows = []
    for i, ts in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        close = quotes["close"][i]
        if close is not None:
            rows.append([dt, quotes["open"][i], quotes["high"][i],
                         quotes["low"][i], close, quotes["volume"][i], close])

    with open(BASE_DIR / "NDX_daily.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"])
        writer.writerows(rows)
    print(f"  Downloaded {len(rows)} rows of NDX data")


if __name__ == "__main__":
    main()
