"""
Run all momentum rotation variants and generate report.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from momentum_utils import download_data, backtest_metrics, spy_benchmark, balanced_60_40, walk_forward_split, print_results
import momentum_v1, momentum_v2, momentum_v3

def composite_score(m):
    """Sharpe×0.4 + Calmar×0.4 + CAGR/100×0.2"""
    return m.get('Sharpe',0)*0.4 + m.get('Calmar',0)*0.4 + m.get('CAGR',0)/100*0.2

def main():
    prices = download_data()
    print(f"Assets: {list(prices.columns)}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    results = {}
    for name, mod in [('v1', momentum_v1), ('v2', momentum_v2), ('v3', momentum_v3)]:
        print(f"\n{'#'*60}")
        print(f"  Running Momentum {name}")
        print(f"{'#'*60}")
        strat_ret = mod.run_strategy(prices)
        is_ret, oos_ret = walk_forward_split(strat_ret)
        
        full = backtest_metrics(strat_ret)
        is_m = backtest_metrics(is_ret)
        oos_m = backtest_metrics(oos_ret)
        
        print_results(f"Momentum {name} (Full)", full)
        print_results(f"Momentum {name} (IS)", is_m)
        print_results(f"Momentum {name} (OOS)", oos_m)
        
        deg = 1 - oos_m.get('Sharpe',0)/is_m['Sharpe'] if is_m.get('Sharpe',0) > 0 else 999
        wf_pass = deg < 0.3
        print(f"OOS degradation: {deg*100:.1f}% - {'PASS' if wf_pass else 'FAIL'}")
        
        results[name] = {
            'full': full, 'is': is_m, 'oos': oos_m,
            'wf_pass': wf_pass, 'degradation': deg,
            'composite': composite_score(full)
        }
    
    # Benchmarks
    spy_ret = spy_benchmark(prices)
    bal_ret = balanced_60_40(prices)
    spy_m = backtest_metrics(spy_ret.iloc[13:])
    bal_m = backtest_metrics(bal_ret.iloc[13:])
    print_results("SPY Buy&Hold", spy_m)
    print_results("60/40", bal_m)
    
    # Find best
    best_name = max(results, key=lambda k: results[k]['composite'])
    best = results[best_name]
    print(f"\n🏆 Best: Momentum {best_name} (composite: {best['composite']:.3f})")
    
    # Generate report
    report = generate_report(results, spy_m, bal_m, prices)
    
    report_path = os.path.join(os.path.dirname(__file__), '..', '..', 'STRATEGY_REPORT.md')
    with open(report_path, 'r') as f:
        existing = f.read()
    with open(report_path, 'w') as f:
        f.write(existing + '\n' + report)
    
    print(f"\nReport appended to STRATEGY_REPORT.md")
    return results, best_name

def generate_report(results, spy_m, bal_m, prices):
    lines = [
        "\n---\n",
        "## 宽资产动量轮动策略 (2026-02-19)",
        "",
        f"**资产宇宙**: {len(prices.columns)} ETFs | **数据**: {prices.index[0].date()} ~ {prices.index[-1].date()}",
        "",
        "### 策略对比",
        "",
        "| 策略 | CAGR | MaxDD | Sharpe | Calmar | 胜率 | OOS Sharpe | WF Pass | Composite |",
        "|------|------|-------|--------|--------|------|------------|---------|-----------|",
    ]
    
    for name, r in results.items():
        f = r['full']
        o = r['oos']
        lines.append(
            f"| Momentum {name} | {f['CAGR']}% | {f['MaxDD']}% | **{f['Sharpe']}** | {f['Calmar']} | {f['WinRate']}% | {o['Sharpe']} | {'✅' if r['wf_pass'] else '❌'} | {r['composite']:.3f} |"
        )
    
    lines.append(f"| SPY B&H | {spy_m['CAGR']}% | {spy_m['MaxDD']}% | {spy_m['Sharpe']} | {spy_m['Calmar']} | {spy_m['WinRate']}% | - | - | - |")
    lines.append(f"| 60/40 | {bal_m['CAGR']}% | {bal_m['MaxDD']}% | {bal_m['Sharpe']} | {bal_m['Calmar']} | {bal_m['WinRate']}% | - | - | - |")
    
    lines.extend([
        "",
        "### 策略说明",
        "- **v1**: 等权 Top5 + 绝对动量过滤（负动量→SHY）",
        "- **v2**: 风险平价(inverse vol) Top5 + 绝对动量过滤",
        "- **v3**: v2 + 波动率择时（高vol时降仓50%→SHY）",
        "- 动量得分：1M/3M/6M/12M 等权加权",
        "- 手续费 0.05%/边 + 滑点 0.1%，无风险利率 4%",
        "- Walk-forward: 60% IS / 40% OOS，OOS Sharpe 降幅<30%为PASS",
        "",
    ])
    
    return '\n'.join(lines)

if __name__ == '__main__':
    main()
