"""
Monte Carlo Stress Test for Momentum v2 (1x and leveraged)
Author: 代码熊 🐻

Bootstrap resampling + extreme scenario analysis.
"""
import numpy as np
import pandas as pd
from momentum_leveraged import run_leveraged
from momentum_utils import download_data, backtest_metrics, RISK_FREE_RATE

SEED = 42
N_SIMS = 1000
SIM_YEARS = 10
SIM_MONTHS = SIM_YEARS * 12


def bootstrap_paths(monthly_returns, n_sims=N_SIMS, n_months=SIM_MONTHS, seed=SEED):
    """Generate bootstrap resampled return paths."""
    rng = np.random.RandomState(seed)
    rets = monthly_returns.values
    indices = rng.randint(0, len(rets), size=(n_sims, n_months))
    return rets[indices]  # shape: (n_sims, n_months)


def path_metrics(paths, rf=RISK_FREE_RATE):
    """Compute Sharpe, MaxDD, final return for each simulated path."""
    sharpes, maxdds, finals = [], [], []
    for path in paths:
        cum = np.cumprod(1 + path)
        total_years = len(path) / 12
        cagr = cum[-1] ** (1 / total_years) - 1
        ann_ret = np.mean(path) * 12
        ann_vol = np.std(path) * np.sqrt(12)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
        
        running_max = np.maximum.accumulate(cum)
        dd = cum / running_max - 1
        maxdd = dd.min()
        
        sharpes.append(sharpe)
        maxdds.append(maxdd)
        finals.append(cum[-1])
    return np.array(sharpes), np.array(maxdds), np.array(finals)


def print_distribution(label, sharpes, maxdds, finals):
    pcts = [5, 25, 50, 75, 95]
    print(f"\n{'=' * 60}")
    print(f"  Monte Carlo: {label} ({len(sharpes)} paths × {SIM_YEARS}yr)")
    print(f"{'=' * 60}")
    
    print(f"\n  Sharpe Distribution:")
    for p in pcts:
        print(f"    P{p:02d}: {np.percentile(sharpes, p):.3f}")
    
    print(f"\n  MaxDD Distribution:")
    for p in pcts:
        print(f"    P{p:02d}: {np.percentile(maxdds, p)*100:.1f}%")
    
    pos_pct = (finals > 1.0).mean() * 100
    print(f"\n  Positive return paths: {pos_pct:.1f}%")
    print(f"  Median final value (per $1): ${np.median(finals):.2f}")


def scenario_test(scenario_name, monthly_rets, leverage=1.0, financing_rate=0.05):
    """Run a hand-crafted scenario and report MaxDD + recovery time."""
    rets = np.array(monthly_rets) * leverage
    if leverage > 1.0:
        rets -= (leverage - 1.0) * financing_rate / 12
    
    cum = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cum)
    dd = cum / running_max - 1
    maxdd = dd.min()
    
    # Recovery time: months from max DD to new high
    trough_idx = np.argmin(dd)
    recovered = np.where(cum[trough_idx:] >= running_max[trough_idx])[0]
    recovery_months = recovered[0] if len(recovered) > 0 else None
    
    rec_str = f"{recovery_months} months" if recovery_months is not None else "NOT recovered"
    print(f"  {scenario_name:40s} MaxDD={maxdd*100:7.2f}%  Recovery={rec_str}")
    return maxdd, recovery_months


def main():
    print("Downloading data...")
    prices = download_data()
    
    # Get historical monthly returns for 1x and best leveraged
    ret_1x = run_leveraged(prices, 1.0, False, False)
    ret_15x = run_leveraged(prices, 1.5, True, True)  # with financing + CB
    ret_20x = run_leveraged(prices, 2.0, True, True)
    
    # Bootstrap
    for label, rets in [("v2 1.0x", ret_1x), ("v2 1.5x (fin+CB)", ret_15x), ("v2 2.0x (fin+CB)", ret_20x)]:
        paths = bootstrap_paths(rets)
        sharpes, maxdds, finals = path_metrics(paths)
        print_distribution(label, sharpes, maxdds, finals)
    
    # Extreme scenarios
    print(f"\n{'=' * 60}")
    print(f"  Extreme Scenario Tests")
    print(f"{'=' * 60}")
    
    scenario_a = [-0.05] * 12  # 12 months of -5%
    scenario_b = [0, -0.10, -0.15, -0.20, -0.15, -0.10, 0.05, 0.10, 0.08, 0.05, 0.03, 0.02]
    scenario_c = [-0.05, -0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    
    for lev_label, lev in [("1.0x", 1.0), ("1.5x", 1.5), ("2.0x", 2.0)]:
        print(f"\n  --- Leverage {lev_label} ---")
        scenario_test(f"A: Persistent Bear ({lev_label})", scenario_a, lev)
        scenario_test(f"B: 2008 GFC ({lev_label})", scenario_b, lev)
        scenario_test(f"C: 2020 COVID ({lev_label})", scenario_c, lev)
    
    # Visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, (label, rets) in enumerate([("1.0x", ret_1x), ("1.5x fin+CB", ret_15x), ("2.0x fin+CB", ret_20x)]):
            paths = bootstrap_paths(rets, seed=SEED + idx)
            sharpes, _, _ = path_metrics(paths)
            axes[idx].hist(sharpes, bins=50, alpha=0.7, color=['steelblue', 'orange', 'red'][idx])
            axes[idx].axvline(np.median(sharpes), color='black', linestyle='--', label=f'Median={np.median(sharpes):.2f}')
            axes[idx].set_title(f'Sharpe Distribution ({label})')
            axes[idx].set_xlabel('Sharpe Ratio')
            axes[idx].legend()
        
        plt.tight_layout()
        out = '/root/.openclaw/workspace/wombat-quant-lab/hedge/codebear/mc_sharpe_dist.png'
        plt.savefig(out, dpi=150)
        print(f"\n  Saved histogram: {out}")
    except Exception as e:
        print(f"\n  (matplotlib unavailable: {e})")
    
    return ret_1x, ret_15x


if __name__ == '__main__':
    main()
