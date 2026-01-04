from typing import Dict, Any, Tuple

# ---------- metrics: run time ----------
def unformat_time(t: str) -> float:
    """Returns '0.00 s' -> 0.00 """
    return float(t[:-2])
def format_time(t: float) -> str:
    """Return 0.00 -> '0.000 s' """
    return f"{t:.3f} s"

def runtime_per_company(runtimes_data: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Any]:
    per_group = {}
    overall = 0
    total_leafs = 0
    for group_name, group in runtimes_data.items():
        total_time_subgroups = sum([unformat_time(sg['overall']) for _, sg in group.items()])
        num_subgroups = len(group.keys())
        overall += total_time_subgroups
        total_leafs += num_subgroups
        per_group[group_name] = format_time(total_time_subgroups / num_subgroups)
    return {
        "overall": format_time(overall / total_leafs),
        "per_group": per_group
    }

def runtime_overall(companies_runtimes: Dict[str, Any]) -> Dict[str, Any]:
    per_company = {name: runtime_per_company(md) for name, md in companies_runtimes.items()}

    # Aggregate overall
    overall = 0
    for res in per_company.values():
        overall += unformat_time(res['overall'])
    overall = f"{overall / len(per_company):.3f} s"

    # Aggregate per-group across companies
    group_sums: Dict[str, float] = {} 
    for res in per_company.values():
        for g, t in res["per_group"].items(): # t is avg overall time per group g for company res
            group_sums.setdefault(g, 0)
            group_sums[g] += unformat_time(t)
    
    per_group = {g: f"{s / len(per_company):.3f} s" for g, s in group_sums.items()}

    # per_company_formatted = {}
    # for name, t in per_company.items():
    #     per_company_formatted[name] = {"overall": f"{t['overall']:.3f} s", "per_group": {k: f"{v:.3f} s" for k, v in t['per_group'].items()}}

    return {"overall": overall, "per_group": per_group, "per_company": per_company} #"per_company": per_company_formatted
