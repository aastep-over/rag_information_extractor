from typing import Dict, Any, Tuple

# ---------- helpers ----------

def _group_leaf_counts(group: Dict[str, Any]) -> Tuple[int, int]:
    """Return (correct, total) over all leaf fields in a group, excluding 'Context' key."""
    correct = 0
    total = 0
    for sub_name, sub in group.items():
        if sub_name == "Context":
            continue
        if isinstance(sub, dict):
            for _, v in sub.items():
                if isinstance(v, int):  # 0/1
                    total += 1
                    correct += int(v)
    return correct, total

def _company_leaf_counts(match_data: Dict[str, Any]) -> Tuple[int, int]:
    """Return (correct, total) across all groups for a company."""
    c_sum = t_sum = 0
    for group in match_data.values():
        c, t = _group_leaf_counts(group)
        c_sum += c
        t_sum += t
    return c_sum, t_sum


# ---------- metrics: accuracy ----------

def accuracy_per_company(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return per-company accuracy:
    - overall accuracy (correct/total)
    - per-group accuracy dict
    """
    per_group = {}
    for group_name, group in match_data.items():
        c, t = _group_leaf_counts(group)
        per_group[group_name] = {
            "correct": c,
            "total": t,
            "accuracy": (c / t) if t else 0.0,
        }
    c_all, t_all = _company_leaf_counts(match_data)
    return {
        "overall": {"correct": c_all, "total": t_all, "accuracy": (c_all / t_all) if t_all else 0.0},
        "per_group": per_group,
    }

def accuracy_overall(companies: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate accuracy across companies:
    - overall (sum correct / sum total)
    - per-group (aggregated across companies)
    - per-company results (useful for reporting)
    """
    per_company = {name: accuracy_per_company(md) for name, md in companies.items()}

    # Aggregate overall
    total_correct = total_count = 0
    for res in per_company.values():
        total_correct += res["overall"]["correct"]
        total_count += res["overall"]["total"]
    overall = {
        "correct": total_correct,
        "total": total_count,
        "accuracy": (total_correct / total_count) if total_count else 0.0,
    }

    # Aggregate per-group across companies
    group_sums: Dict[str, Dict[str, int]] = {}
    for res in per_company.values():
        for g, stats in res["per_group"].items():
            group_sums.setdefault(g, {"correct": 0, "total": 0})
            group_sums[g]["correct"] += stats["correct"]
            group_sums[g]["total"] += stats["total"]

    per_group = {
        g: {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": (s["correct"] / s["total"]) if s["total"] else 0.0,
        }
        for g, s in group_sums.items()
    }

    return {"overall": overall, "per_group": per_group, "per_company": per_company}





