#!/usr/bin/env python3
"""Compare your agent's performance against baseline"""

import json

def compare_results():
    """Compare advanced agent vs baseline"""
    
    # Load baseline results
    try:
        with open('./example/evaluation_results_track2_goodreads.json', 'r') as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print("❌ Baseline results not found")
        baseline = {
            "top_1_hit_rate": 0.11,
            "top_3_hit_rate": 0.2925,
            "top_5_hit_rate": 0.4475
        }
    
    # Load your results
    try:
        with open('./evaluation_results_advanced_goodreads.json', 'r') as f:
            advanced = json.load(f)
    except FileNotFoundError:
        print("❌ Advanced agent results not found yet")
        print("   Run: python run_agent.py --tasks 50 --workers 5")
        return
    
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print()
    
    # Compare each metric
    metrics = [
        ("HR@1", "top_1_hit_rate"),
        ("HR@3", "top_3_hit_rate"),
        ("HR@5", "top_5_hit_rate")
    ]
    
    for name, key in metrics:
        base_val = baseline.get(key, 0)
        adv_val = advanced.get(key, 0)
        improvement = ((adv_val - base_val) / base_val * 100) if base_val > 0 else 0
        
        print(f"{name}:")
        print(f"  Baseline:  {base_val:.4f} ({base_val*100:.2f}%)")
        print(f"  Advanced:  {adv_val:.4f} ({adv_val*100:.2f}%)")
        
        if improvement > 0:
            print(f"  📈 Improvement: +{improvement:.1f}%")
        elif improvement < 0:
            print(f"  📉 Decrease: {improvement:.1f}%")
        else:
            print(f"  ➡️  No change")
        print()
    
    # Overall assessment
    avg_baseline = sum(baseline.get(k, 0) for _, k in metrics) / len(metrics)
    avg_advanced = sum(advanced.get(k, 0) for _, k in metrics) / len(metrics)
    overall_improvement = ((avg_advanced - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0
    
    print("=" * 70)
    print(f"OVERALL IMPROVEMENT: {overall_improvement:+.1f}%")
    print("=" * 70)
    
    if overall_improvement > 10:
        print("🎉 Excellent! Significant improvement!")
    elif overall_improvement > 0:
        print("✅ Good progress! Keep optimizing.")
    else:
        print("⚠️  Consider tuning hyperparameters")

if __name__ == "__main__":
    compare_results()
