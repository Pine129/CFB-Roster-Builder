#attributes_integration.py
import json
import math
import numpy as np
from copy import deepcopy

# ---------- Helpers to load schema ----------
def load_schema(path="schema.json"):
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    # Ensure attr_order is present
    if "attr_order" not in schema:
        raise ValueError("schema.json missing 'attr_order'")
    return schema

# ---------- Build covariance matrix ----------
def build_covariance_matrix(schema):
    attrs = schema["attr_order"]
    n = len(attrs)
    # start with small independent variance
    cov = np.eye(n) * 4.0
    # insert block covariances
    for block in schema.get("covariance_blocks", {}).values():
        block_attrs = block["attrs"]
        block_cov = np.array(block["cov"], dtype=float)
        # map indices
        idxs = [attrs.index(a) for a in block_attrs if a in attrs]
        for i, ii in enumerate(idxs):
            for j, jj in enumerate(idxs):
                cov[ii, jj] = block_cov[i, j] * 4.0  # scale base variance
    # small regularization to keep matrix positive definite
    cov += np.eye(n) * 1e-6
    return cov

# ---------- Generate one player's attributes ----------
def generate_attributes_for_player(player, schema, cov_matrix=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    attrs = schema["attr_order"]
    n = len(attrs)

    # base vector
    pos = player.get("Position", "default")
    pos_bases = schema.get("positions", {}).get(pos, {})
    base_value = schema.get("defaults", {}).get("base_value", 50)
    mu = np.array([pos_bases.get(a, base_value) for a in attrs], dtype=float)

    # archetype modifiers
    arche = player.get("Archetype")
    if arche and arche in schema.get("archetypes", {}):
        for k, v in schema["archetypes"][arche].items():
            if k in attrs:
                mu[attrs.index(k)] += v

    # year modifier (small)
    year = player.get("Year", "JR")
    year_mod_map = {"FR": -2.0, "SO": -1.0, "JR": 0.0, "SR": 1.0}
    year_mod = year_mod_map.get(year, 0.0)
    # apply small year shift to awareness and stamina primarily
    if "AWR" in attrs:
        mu[attrs.index("AWR")] += year_mod
    if "STA" in attrs:
        mu[attrs.index("STA")] += year_mod

    # dev/potential modifiers
    dev = player.get("DevTrait", "Normal")
    pot = player.get("Potential", "Medium")
    dev_mod = schema.get("dev_potential_modifiers", {}).get("DevTrait", {}).get(dev, {})
    pot_mod = schema.get("dev_potential_modifiers", {}).get("Potential", {}).get(pot, {})

    variance_scale = float(dev_mod.get("variance_scale", 1.0)) * float(pot_mod.get("variance_scale", 1.0))
    mean_shift = float(pot_mod.get("mean_shift", 0.0)) + float(dev_mod.get("ceiling_bonus", 0.0)) * 0.0

    # apply mean_shift globally
    mu = mu + mean_shift

    # sample correlated noise
    if cov_matrix is None:
        cov_matrix = build_covariance_matrix(schema)
    # scale covariance by variance_scale
    cov = cov_matrix * variance_scale

    # draw sample
    try:
        sample = rng.multivariate_normal(mu, cov)
    except Exception:
        # fallback to independent normal if covariance fails
        sample = mu + rng.normal(0, 4.0, size=n)

    # clamp and round
    min_attr = schema.get("defaults", {}).get("min_attr", 20)
    max_attr = schema.get("defaults", {}).get("max_attr", 99)
    sample = np.clip(np.round(sample), min_attr, max_attr).astype(int)

    # convert to dict
    attr_dict = {a: int(sample[i]) for i, a in enumerate(attrs)}
    return attr_dict

# ---------- Compute weighted overall ----------
def compute_overall_from_attributes(attr_dict, schema, position):
    weights_map = schema.get("weights_for_overall", {})
    # try exact position weights, fallback to default, fallback to uniform
    weights = weights_map.get(position) or weights_map.get("default") or {}
    if not weights:
        # uniform average if no weights
        vals = list(attr_dict.values())
        return float(sum(vals) / len(vals))
    # compute weighted sum and normalize by sum of weights
    total = 0.0
    weight_sum = 0.0
    for k, w in weights.items():
        if k in attr_dict:
            total += attr_dict[k] * float(w)
            weight_sum += float(w)
    if weight_sum <= 0:
        return float(sum(attr_dict.values()) / len(attr_dict))
    # scale to 0-100 like a typical overall
    overall = total / weight_sum
    # clamp to 20-99
    overall = max(20.0, min(99.0, overall))
    return float(overall)

# ---------- Adjust attributes to match target overall ----------
def scale_attributes_to_target(attr_dict, target_overall, schema, position, max_iter=6):
    """
    Small linear scaling toward baseline to nudge computed overall to target_overall.
    Keeps relative differences between attributes.
    """
    attrs = schema["attr_order"]
    baseline = schema.get("defaults", {}).get("base_value", 50)
    # compute current overall
    current = compute_overall_from_attributes(attr_dict, schema, position)
    if abs(current - target_overall) < 0.25:
        return attr_dict, current

    # linear scaling factor s where new = baseline + (old - baseline) * s
    denom = (current - baseline) if abs(current - baseline) > 1e-6 else 1.0
    s = (target_overall - baseline) / denom
    # clamp s to reasonable range to avoid extreme jumps
    s = max(0.7, min(1.3, s))

    new_attrs = {}
    for a in attrs:
        old = attr_dict.get(a, baseline)
        new_val = baseline + (old - baseline) * s
        new_attrs[a] = int(round(max(schema["defaults"]["min_attr"], min(schema["defaults"]["max_attr"], new_val))))

    new_overall = compute_overall_from_attributes(new_attrs, schema, position)
    # small iterative correction if needed
    for _ in range(max_iter):
        if abs(new_overall - target_overall) < 0.25:
            break
        denom = (new_overall - baseline) if abs(new_overall - baseline) > 1e-6 else 1.0
        s *= (target_overall - baseline) / denom
        s = max(0.7, min(1.3, s))
        for a in attrs:
            old = attr_dict.get(a, baseline)
            new_val = baseline + (old - baseline) * s
            new_attrs[a] = int(round(max(schema["defaults"]["min_attr"], min(schema["defaults"]["max_attr"], new_val))))
        new_overall = compute_overall_from_attributes(new_attrs, schema, position)

    return new_attrs, new_overall

# ---------- Main batch function ----------
def generate_attributes_for_roster(roster, schema_path="schema.json", rng_seed=None, match_target_overall=True):
    schema = load_schema(schema_path)
    rng = np.random.default_rng(rng_seed)
    cov_matrix = build_covariance_matrix(schema)
    attrs_order = schema["attr_order"]

    new_roster = []
    for player in roster:
        # generate attributes
        attr_dict = generate_attributes_for_player(player, schema, cov_matrix=cov_matrix, rng=rng)
        # compute attribute-based overall
        pos = player.get("Position", "default")
        computed_overall = compute_overall_from_attributes(attr_dict, schema, pos)

        # if roster already has a target Overall, scale attributes to match it
        target = player.get("Overall")
        if match_target_overall and target is not None:
            try:
                target_val = float(target)
                attr_dict, new_overall = scale_attributes_to_target(attr_dict, target_val, schema, pos)
                computed_overall = new_overall
            except Exception:
                pass

        # attach attributes to player dict
        player_copy = deepcopy(player)
        for a in attrs_order:
            player_copy[a] = int(attr_dict.get(a, schema["defaults"]["base_value"]))
        # also attach computed overall for verification
        player_copy["_ComputedOverall"] = round(computed_overall, 2)
        new_roster.append(player_copy)

    return new_roster

# ---------- Example usage ----------
if __name__ == "__main__":
    # quick test
    sample_player = {"Position": "WR", "Archetype": "Route Runner", "Year": "JR", "DevTrait": "Impact", "Potential": "High", "Overall": 78}
    schema = load_schema("schema.json")
    attrs = generate_attributes_for_player(sample_player, schema)
    overall = compute_overall_from_attributes(attrs, schema, "WR")
    scaled_attrs, new_overall = scale_attributes_to_target(attrs, 78, schema, "WR")
    print("Computed overall:", overall, "Scaled overall:", new_overall)

