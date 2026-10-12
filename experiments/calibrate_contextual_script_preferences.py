"""Preference evaluation statistics shared with comparator training."""
import math

def summarize(rows):
    """Count failures in the denominator and compare paired majority predictions."""
    all_rows = rows
    rows = [r for r in rows if r['human_p_a'] != .5]
    n = len(rows)
    if not n:
        raise ValueError('Empty calibration')
    majority = sum(r['human_p_a'] > .5 for r in rows) > n / 2
    wins = losses = correct = 0
    class_correct = {True: 0, False: 0}
    class_total = {True: 0, False: 0}
    brier = 0.
    for r in rows:
        gold = r['human_p_a'] > .5
        class_total[gold] += 1
        valid = r.get('decided', False)
        ok = valid and (r['probability_a'] > .5) == gold
        base = majority == gold
        correct += ok
        class_correct[gold] += ok
        wins += ok and not base
        losses += base and not ok
        p = r.get('probability_a', .5)
        brier += (p - r['human_p_a']) ** 2
    discordant = wins + losses
    p_value = sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / 2 ** discordant if discordant else 1.
    accuracy = correct / n
    z = 1.96
    lower = (accuracy + z*z/(2*n) - z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))) / (1+z*z/n)
    return {'n': n, 'attempted_rows': len(all_rows), 'human_tie_rows': len(all_rows)-n,
            'correct_all_pairs': correct, 'accuracy_all_pairs': accuracy,
            'balanced_accuracy': sum(class_correct[c]/class_total[c] if class_total[c] else 0 for c in (True, False))/2,
            'order_stability': sum(r.get('order_stable', False) for r in rows)/n,
            'decided_coverage': sum(r.get('decided', False) for r in rows)/n,
            'majority_constant_accuracy': max(class_total.values())/n,
            'paired_majority_baseline_one_sided_exact_p': p_value,
            'accuracy_wilson95_lower': lower,
            'soft_target_brier': sum((r.get('probability_a',.5)-r['human_p_a'])**2 for r in all_rows)/len(all_rows),
            'failures_in_denominator': True}
