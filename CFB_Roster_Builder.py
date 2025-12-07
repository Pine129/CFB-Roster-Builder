
Requirements:
 - pandas, numpy, matplotlib, seaborn, openpyxl, scikit-learn
Files required in working directory:
 - CFB26_attribute_overall_analysis.xlsx
 - CFB26_AllPlayers_withOVR_withArchetype.xlsx
 - first_names.txt
 - last_names.txt
"""

import os
import re
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from numpy.linalg import cholesky, LinAlgError
from sklearn.covariance import LedoitWolf

# ---------- Config ----------
ANALYSIS_FILE = "CFB26_attribute_overall_analysis.xlsx"
RAW_FILE = "CFB26_AllPlayers_withOVR_withArchetype.xlsx"
FIRST_NAMES_FILE = "first_names.txt"
LAST_NAMES_FILE = "last_names.txt"

RATIOS_SHEET = "Ratios_Pos_Arch"
CORRS_SHEET = "Correlations_Pos_Arch"
RIDGE_SHEET = "RidgeWeights_Pos_Arch"

USE_ANALYSIS_ARCH = True
TOTAL_PLAYERS = 11062

SHRINK_GRID = [0.35, 0.45, 0.55]
NOISE_GRID = [0.35, 0.45, 0.55]
SAMPLE_N = 2000
GEN_DATA_DIR = "GEN_Data"
DIAGNOSTIC_XLSX = "correlation_diagnostics.xlsx"
os.makedirs(GEN_DATA_DIR, exist_ok=True)

POSITIONS = {
    "QB": 5, "HB": 6, "FB": 1, "WR": 9, "TE": 5,
    "LT": 3, "LG": 3, "C": 3, "RG": 3, "RT": 3,
    "LE": 4, "RE": 4, "DT": 6, "ROLB": 4, "MLB": 4,
    "LOLB": 4, "CB": 8, "FS": 4, "SS": 3, "K": 2, "P": 1
}
POSITION_ORDER = ["QB","RB","FB","WR","TE","LT","LG","C","RG","RT","LE","RE","DT","SAM","MIKE","WILL","CB","FS","SS","K","P"]
POSITION_INDEX = {pos: i for i, pos in enumerate(POSITION_ORDER)}
ACADEMIC_YEARS = ["FR","SO","JR","SR"]

DEV_TRAITS = ["Normal","Impact","Star","Elite"]
# default dev weights (used as fallback)
DEV_WEIGHTS = [0.70,0.20,0.08,0.02]
# year-aware dev weights (younger players less likely to be Star/Elite)
DEV_WEIGHTS_BY_YEAR = {
    "FR": [0.85, 0.12, 0.03, 0.00],
    "SO": [0.78, 0.16, 0.05, 0.01],
    "JR": [0.65, 0.22, 0.10, 0.03],
    "SR": [0.55, 0.25, 0.15, 0.05]
}

POTENTIALS = ["Low","Medium","High"]
POT_WEIGHTS = [0.20,0.60,0.20]

HANDEDNESS_BY_POSITION = {
    "QB":0.90,"RB":0.85,"WR":0.85,"TE":0.88,
    "LT":0.95,"LG":0.95,"C":0.95,"RG":0.95,"RT":0.95,
    "LE":0.95,"RE":0.95,"DT":0.95,
    "MIKE":0.90,"SAM":0.90,"WILL":0.90,
    "CB":0.90,"FS":0.90,"SS":0.90,
    "K":0.80,"P":0.80
}

# ---------- Analysis position mapping fixes ----------
# Clear, unambiguous mapping helpers
ANALYSIS_POSITION_MAP = {
    "RB": "HB", "MIKE": "MLB", "SAM": "LOLB", "WILL": "ROLB",
    "HB": "RB", "MLB": "MIKE", "LOLB": "SAM", "ROLB": "WILL"
}



def provided_to_analysis_position(pos: str) -> str:
    """Map generator/provided position (RB, MIKE, SAM, WILL) to analysis labels (HB, MLB, LOLB, ROLB)."""
    if pos is None:
        return pos
    p = str(pos).strip().upper()
    return ANALYSIS_POSITION_MAP.get(p, p)

def analysis_to_provided_position(pos: str) -> str:
    """Map analysis position label (HB, MLB, LOLB, ROLB) back to the generator/provided name (RB, MIKE, SAM, WILL)."""
    if pos is None:
        return pos
    p = str(pos).strip().upper()
    # If the map contains both directions, this returns the opposite mapping
    return ANALYSIS_POSITION_MAP.get(p, p)
# Backwards-compatible aliases for older code
# Keep these so existing calls to map_to_analysis_position / map_analysis_position_to_provided still work
def map_to_analysis_position(pos: str) -> str:
    return provided_to_analysis_position(pos)

def map_analysis_position_to_provided(pos: str) -> str:
    return analysis_to_provided_position(pos)




# Year bias to shift mean overall by academic year (conservative)
YEAR_OVERALL_BIAS = {"FR": -4, "SO": -2, "JR": +1, "SR": +3}

# ---------- Aggregate attributes (must remain blank for everyone) ----------
AGGREGATE_ATTRS = ["Ballcarrier","Blocking","Passing","Defense","Receiving","Kicking","General"]
AGGREGATE_ATTR_KEYS = [re.sub(r'[^a-z0-9 ]','', a.lower()).strip() for a in AGGREGATE_ATTRS]

# ---------- Positional archetypes (user-provided) ----------
ARCHETYPES = {
    "QB": ["Pocket Passer", "Backfield Creator", "Dual Threat", "Pure Runner"],
    "RB": ["Contact Seeker", "East/West Playmaker", "Backfield Threat", "North/South Receiver", "Elusive Bruiser"],
    "FB": ["Blocking", "Utility"],
    "WR": ["Route Artist", "Speedster", "Physical Route Runner", "Elusive Route Runner", "Gritty Possession", "Gadget", "Contested Specialist"],
    "TE": ["Physical Route Runner", "Vertical Threat", "Pure Blocker", "Gritty Possession", "Pure Possession"],
    "LT": ["Pass Protector", "Raw Strength", "Well Rounded", "Agile"],
    "LG": ["Pass Protector", "Raw Strength", "Well Rounded", "Agile"],
    "C":  ["Pass Protector", "Raw Strength", "Well Rounded", "Agile"],
    "RG": ["Pass Protector", "Raw Strength", "Well Rounded", "Agile"],
    "RT": ["Pass Protector", "Raw Strength", "Well Rounded", "Agile"],
    "LE": ["Speed Rusher", "Power Rusher", "Pure Power", "Edge Setter"],
    "RE": ["Speed Rusher", "Power Rusher", "Pure Power", "Edge Setter"],
    "DT": ["Speed Rusher", "Power Rusher", "Pure Power", "Gap Specialist"],
    "MIKE": ["Signal Caller", "Lurker", "Thumper"],
    "SAM": ["Signal Caller", "Lurker", "Thumper"],
    "WILL": ["Signal Caller", "Lurker", "Thumper"],
    "CB": ["Bump and Run", "Boundary", "Zone", "Field"],
    "FS": ["Coverage Specialist", "Hybrid", "Box Specialist"],
    "SS": ["Coverage Specialist", "Hybrid", "Box Specialist"],
    "K": ["Accuracy", "Power"],
    "P": ["Power", "Accurate"]
}
def add_previous_redshirt_column(df: pd.DataFrame,
                                 col_name: str = "previous redshirt",
                                 yes_prob: float = 0.15,
                                 overwrite: bool = False,
                                 random_seed: int | None = None) -> pd.DataFrame:
    """
    Add a 'previous redshirt' column to df with 'Yes'/'No' values.
    - yes_prob: probability of 'Yes' (e.g., 0.15)
    - overwrite: if False and column exists, leave it unchanged
    - random_seed: optional seed for reproducibility
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if col_name in df.columns and not overwrite:
        # leave existing column alone
        return df

    n = len(df)
    choices = np.random.choice(["Yes", "No"], size=n, p=[yes_prob, 1 - yes_prob])
    df[col_name] = choices
    return df

_ARCH_KEYS = {pos: [re.sub(r'[^a-z0-9 ]','', a.lower()).strip() for a in archs] for pos, archs in ARCHETYPES.items()}

def clean_arch_name(raw: str) -> str:
    """
    Normalize archetype strings by removing truncation, special characters, and stray symbols.
    Example: 'Pocket Pas:' -> 'pocket passer'
             'Dual Threa®' -> 'dual threat'
             'Backfield C' -> 'backfield creator'
    """
    if raw is None:
        return ""
    # Lowercase, strip whitespace, remove non-alphanumeric
    cleaned = re.sub(r'[^a-z0-9 ]', '', str(raw).lower()).strip()
    return cleaned
def get_arch_list_for_pos(pos: str):
    return ARCHETYPES.get(pos, ["Default"])

def map_analysis_archetype_to_provided(raw_arch: str, pos: str):
    if raw_arch is None:
        return None
    raw_key = clean_arch_name(raw_arch)
    pos_keys = _ARCH_KEYS.get(pos, [])
    if raw_key in pos_keys:
        return ARCHETYPES[pos][pos_keys.index(raw_key)]
    best = None; best_len = 0
    for i, k in enumerate(pos_keys):
        if k and (k in raw_key or raw_key in k):
            if len(k) > best_len:
                best = ARCHETYPES[pos][i]; best_len = len(k)
    if best:
        return best
    for i, k in enumerate(pos_keys):
        if raw_key.startswith(k) or k.startswith(raw_key):
            return ARCHETYPES[pos][i]
    if pos in ARCHETYPES and len(ARCHETYPES[pos])>0:
        return ARCHETYPES[pos][0]
    return ANALYSIS_POSITION_MAP.get(pos, pos)  # ensure the arrow and function compile



#for pos in ["RB","MIKE","SAM","WILL"]:
    #print(pos, ARCHETYPES[pos])


# ---------- Utilities ----------

# normalize_positions should produce provided names
def normalize_positions(roster: pd.DataFrame) -> pd.DataFrame:
    if "Position" not in roster.columns:
        raise KeyError("Roster DataFrame must contain a 'Position' column.")
    roster = roster.copy()
    # convert analysis labels (HB/MLB/LOLB/ROLB) to provided names (RB/MIKE/SAM/WILL)
    roster["NormPosition"] = roster["Position"].apply(analysis_to_provided_position)
    return roster


def normalize_attr_name(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]','', str(s).lower()).strip()

def clamp(v, lo, hi):
    return int(max(lo, min(hi, round(v))))

def pick_dev_trait_by_year(year: str):
    weights = DEV_WEIGHTS_BY_YEAR.get(year, DEV_WEIGHTS)
    return random.choices(DEV_TRAITS, weights=weights, k=1)[0]

def pick_potential():
    return random.choices(POTENTIALS, weights=POT_WEIGHTS, k=1)[0]

def pick_handedness(position: str) -> str:
    prob_right = HANDEDNESS_BY_POSITION.get(position, 0.9)
    return "Right" if random.random() < prob_right else "Left"

def ensure_overall_column(df: pd.DataFrame) -> pd.DataFrame:
    # If 'Overall' is missing but 'OVR' exists, rename it
    if "Overall" not in df.columns and "OVR" in df.columns:
        df = df.rename(columns={"OVR": "Overall"})
    return df

def normalize_roster_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw roster DataFrame column names to expected schema.
    - Rename POS_SHORT -> Position
    - Rename OVR -> Overall
    - Keep Archetype as-is
    """
    df = df.copy()
    if "POS_SHORT" in df.columns and "Position" not in df.columns:
        df = df.rename(columns={"POS_SHORT": "Position"})
    if "OVR" in df.columns and "Overall" not in df.columns:
        df = df.rename(columns={"OVR": "Overall"})
    return df

def project_to_target_overall(sample, w, b, target_ovr, Sigma=None, eps=1e-8):
    """
    Adjust sample (numpy array) so that w^T sample + b == target_ovr.
    Sigma: covariance matrix (numpy) or None.
    Returns adjusted sample (numpy array).
    """
    import numpy as _np
    sample = _np.asarray(sample, dtype=float)
    w = _np.asarray(w, dtype=float)
    pred = float(_np.dot(w, sample) + b)
    delta = float(target_ovr - pred)
    if abs(delta) < 1e-8:
        return sample
    # Covariance-aware projection
    if Sigma is not None:
        Sigma = _np.asarray(Sigma, dtype=float)
        denom = float(_np.dot(w, Sigma.dot(w)))
        if denom is None or abs(denom) < eps:
            # fallback to Euclidean
            denom = float(_np.dot(w, w))
            if abs(denom) < eps:
                return sample
            return sample + (delta / denom) * w
        return sample + (delta / denom) * (Sigma.dot(w))
    else:
        denom = float(_np.dot(w, w))
        if abs(denom) < eps:
            return sample
        return sample + (delta / denom) * w

# --- Helper: add hs star rating column ---

def hs_star_rating(df: pd.DataFrame,
                   prompt: str,
                   col_name: str = "hs star rating",
                   overwrite: bool = False) -> pd.DataFrame:
    """
    Parse a single-line prompt of the form:
      "majority <1-5> [seed=<int>]"
    and add a column with values "one","two","three","four","five".
    - prompt: e.g. "majority 3 seed=42" or "majority 1"
    - overwrite: if False and column exists, leave it unchanged
    Returns the modified DataFrame (column contains scalar strings only).
    """
    if col_name in df.columns and not overwrite:
        return df

    # default probability mapping for majority 1..5 (one,two,three,four,five)
    majority_map = {
        1: [0.70, 0.20, 0.08, 0.02, 0.00],
        2: [0.25, 0.45, 0.20, 0.08, 0.02],
        3: [0.10, 0.30, 0.40, 0.15, 0.05],
        4: [0.02, 0.08, 0.40, 0.40, 0.10],
        5: [0.01, 0.04, 0.40, 0.40, 0.15]
    }

    # parse prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string like: 'majority 3' or 'majority 4 seed=7'")

    s = prompt.strip()
    m = re.match(r'^\s*majority\s+([1-5])(?:\s*[,\s]\s*seed\s*=\s*(\d+))?\s*$', s, flags=re.IGNORECASE)
    if not m:
        # allow "majority 3 seed=42" or "majority 3, seed=42"
        m2 = re.match(r'^\s*majority\s+([1-5])(?:.*seed\s*=\s*(\d+))?', s, flags=re.IGNORECASE)
        if not m2:
            raise ValueError("Unrecognized prompt. Use: 'majority <1-5> [seed=<int>]'.")
        m = m2

    majority = int(m.group(1))
    seed = int(m.group(2)) if m.group(2) is not None else None

    probs = majority_map.get(majority)
    if probs is None:
        raise ValueError("Majority must be an integer 1 through 5.")

    n = len(df)
    labels = ["one", "two", "three", "four", "five"]

    if n == 0:
        df[col_name] = pd.Series(dtype="object")
        return df

    # reproducible randomness if seed provided
    if seed is not None:
        rng = np.random.default_rng(seed)
        choices = rng.choice(labels, size=n, p=probs)
    else:
        choices = np.random.choice(labels, size=n, p=probs)

    # ensure scalar strings only
    df[col_name] = choices.astype(str)
    return df

def read_majority_prompt_cli(prompt_text="Enter star distribution (e.g. 'majority 3' or 'majority 3 seed=42'): "):
    """
    Read and validate a single-line prompt from the user.
    Returns canonical_prompt string like "majority 3" or "majority 3 seed=42".
    Raises ValueError on invalid input.
    """
    s = input(prompt_text).strip()
    if not s:
        # fallback default
        return "majority 3"
    # Accept "majority 3", "majority 3 seed=42", "majority 3, seed=42"
    m = re.match(r'^\s*majority\s+([1-5])(?:\s*[,\s]\s*seed\s*=\s*(\d+))?\s*$', s, flags=re.IGNORECASE)
    if not m:
        m2 = re.match(r'^\s*majority\s+([1-5])(?:.*seed\s*=\s*(\d+))?', s, flags=re.IGNORECASE)
        if not m2:
            raise ValueError("Unrecognized prompt. Use: 'majority <1-5> [seed=<int>]'.")
        m = m2
    majority = int(m.group(1))
    seed = int(m.group(2)) if m.group(2) is not None else None
    return f"majority {majority}" + (f" seed={seed}" if seed is not None else "")

# Insert this near your existing skew helpers (parse_skew_direction / parse_skew_amount / compute_mode)
import re

def display_skew_table():
    """
    Prints a compact table showing skew directions, example percentages,
    and a short human explanation of the effect on the triangular sampling mode.
    """
    rows = [
        ("Direction", "Example %", "Effect (mode shift)"),
        ("center", "0%", "Mode at midpoint (lo+hi)/2 — symmetric"),
        ("higher", "10% / 30% / 60%", "Mode moves toward upper bound; larger % -> stronger upper skew"),
        ("lower",  "10% / 30% / 60%", "Mode moves toward lower bound; larger % -> stronger lower skew"),
    ]
    # simple aligned print (avoids external libs)
    col_widths = [max(len(r[i]) for r in rows) for i in range(3)]
    print("\nSkew selection examples (how the triangular mode shifts):")
    print("-" * (sum(col_widths) + 6))
    for i, r in enumerate(rows):
        print(f"{r[0].ljust(col_widths[0])}   {r[1].ljust(col_widths[1])}   {r[2].ljust(col_widths[2])}")
        if i == 0:
            print("-" * (sum(col_widths) + 6))
    print("-" * (sum(col_widths) + 6))
    print("Notes:")
    print(" - 'higher' moves the distribution toward the upper bound (more high overalls).")
    print(" - 'lower' moves the distribution toward the lower bound (more low overalls).")
    print(" - percentage controls how far the mode moves from center toward the bound.\n")

def read_skew_prompt_cli(prompt_text="Enter skew (e.g. 'higher 30%' or 'center 0'): "):
    """
    Interactive prompt that shows the skew table, accepts user input, and returns
    (direction, amount01) where amount01 is a float in [0.0, 1.0].
    Acceptable inputs:
      - 'center' or 'center 0'
      - 'higher 30' or 'higher 30%'  -> returns ('higher', 0.30)
      - 'lower 10' or 'lower 10%'
    If the user presses Enter, returns default ('center', 0.0).
    Raises ValueError for invalid input.
    """
    display_skew_table()
    s = input(prompt_text).strip()
    if s == "":
        return "center", 0.0

    # Accept forms: "higher 30", "higher 30%", "center", "lower 10"
    m = re.match(r'^\s*(higher|lower|center)\s*(\d{1,3}\s*%?)?\s*$', s, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Invalid skew input. Use 'higher 30%', 'lower 10', or 'center'.")
    direction = m.group(1).lower()
    pct = m.group(2)
    if direction == "center":
        return "center", 0.0
    if not pct:
        # default small shift if user omitted percentage
        return direction, 0.25
    # parse percentage
    pct_val = float(str(pct).replace("%", "").strip())
    pct_val = max(0.0, min(100.0, pct_val))
    return direction, pct_val / 100.0
# Example integration in your pipeline (synchronous CLI)


def add_dealbreaker_column(df: pd.DataFrame,
                           col_name: str = "dealbreaker",
                           choices: list | None = None,
                           equal: bool = True,
                           overwrite: bool = False,
                           seed: int | None = None) -> pd.DataFrame:
    """
    Add a 'dealbreaker' column populated equally with the provided choices.
    - choices: list of string values to use (default set below).
    - equal: if True, produce an (approximately) exact equal count of each choice
             by repeating and shuffling; if False, sample uniformly at random.
    - overwrite: if False and column exists, leave it unchanged.
    - seed: optional integer for reproducible results.
    Returns the modified DataFrame (scalar strings only).
    """
    if choices is None:
        choices = [
            "Playing Time", "Playing Style", "Championship Contender",
            "Proximity to Home", "Brand Exposure", "Conference Prestige",
            "Coach Prestige", "Pro Potential", "None"
        ]

    if col_name in df.columns and not overwrite:
        return df

    n = len(df)
    if n == 0:
        df[col_name] = pd.Series(dtype="object")
        return df

    # deterministic RNG if seed provided
    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
    else:
        rng = random.Random()
        np_rng = np.random.default_rng()

    k = len(choices)
    if equal:
        # create a list that repeats each choice enough times, trim to n, then shuffle
        reps = math.ceil(n / k)
        pool = choices * reps
        # trim to exact length n
        pool = pool[:n]
        rng.shuffle(pool)
        assigned = np.array(pool, dtype=object)
    else:
        # uniform random sampling with equal probability
        assigned = np_rng.choice(choices, size=n)

    df[col_name] = pd.Series(assigned, index=df.index).astype(str)
    return df


def add_height_weight_with_presets(df: pd.DataFrame,
                                   position_col: str = "Position",
                                   height_in_col: str = "Height_in",
                                   height_str_col: str = "Height",
                                   weight_col: str = "Weight_lb",
                                   preset: str | None = None,
                                   prompt_for_preset: bool = True,
                                   seed: int | None = None,
                                   overwrite: bool = False) -> pd.DataFrame:
    """
    Add height (inches + formatted string) and weight (lbs) based on position,
    with team-wide size presets the user can choose from.

    Presets (from smallest to biggest):
      1. Tiny               (smallest)
      2. High School        (smaller)
      3. Small Side         (small)
      4. Average            (medium)  [default]
      5. Corn Fed           (big)
      6. Giants             (bigger)
      7. Physical Specimens (biggest)
      8. Chode              (small and heavy)
      9. Waluigi            (tall and thin)

    Parameters
    - df: roster DataFrame (must contain position_col or will use empty string)
    - preset: optional preset name or number (case-insensitive). If None and prompt_for_preset True,
              the function prints the presets table and reads user input via input().
    - prompt_for_preset: if False and preset is None, uses "average".
    - seed: optional integer for reproducible results
    - overwrite: if False and any of the target columns exist, the function returns df unchanged

    Returns modified DataFrame with three new columns:
      - height_in_col (float inches)
      - weight_col (int lbs)
      - height_str_col (formatted like 6'2")
    Also sets df.attrs["height_weight_preset"] and writes a column "size_preset".
    """

    # If any target columns exist and overwrite not requested, do nothing
    if (height_in_col in df.columns or height_str_col in df.columns or weight_col in df.columns or "size_preset" in df.columns) and not overwrite:
        return df

    # Preset scale mapping (height_scale, weight_scale)
    # Chode: slightly smaller height but substantially heavier
    # Waluigi: taller but noticeably thinner
    preset_map = {
        "tiny":               (0.94, 0.92),
        "high school":        (0.97, 0.96),
        "small side":         (0.99, 0.98),
        "average":            (1.00, 1.00),
        "corn fed":           (1.04, 1.06),
        "giants":             (1.08, 1.12),
        "physical specimens": (1.12, 1.18),
        "chode":              (0.96, 1.14),  # small but heavy
        "waluigi":            (1.12, 0.92)   # tall but thin
    }

    # Friendly display order (updated to include the two new presets)
    display_order = [
        ("Tiny", "smallest"),
        ("High School", "smaller"),
        ("Small Side", "small"),
        ("Average", "medium"),
        ("Corn Fed", "big"),
        ("Giants", "bigger"),
        ("Physical Specimens", "biggest"),
        ("Chode", "small and heavy"),
        ("Waluigi", "tall and thin")
    ]

    # Normalize and accept synonyms and numeric choices
    synonyms = {
        # numeric keys
        "1": "tiny", "2": "high school", "3": "small side", "4": "average", "5": "corn fed", "6": "giants", "7": "physical specimens", "8": "chode", "9": "waluigi",
        # names and common variants
        "tiny": "tiny", "smallest": "tiny",
        "highschool": "high school", "high school": "high school", "hs": "high school",
        "small side": "small side", "small": "small side",
        "average": "average", "medium": "average",
        "corn fed": "corn fed", "cornfed": "corn fed",
        "giants": "giants", "giant": "giants",
        "physical specimens": "physical specimens", "physical": "physical specimens", "specimens": "physical specimens",
        "chode": "chode", "small and heavy": "chode", "shortheavy": "chode",
        "waluigi": "waluigi", "walugi": "waluigi", "tall and thin": "waluigi", "tallthin": "waluigi"
    }

    # Determine chosen preset
    chosen_norm = None
    if preset is None and prompt_for_preset:
        # interactive prompt
        print("Select a team size preset (type the name or number):")
        for i, (name, desc) in enumerate(display_order, start=1):
            print(f"  {i}. {name:18s} — {desc}")
        raw = input(f"Preset (name or 1-{len(display_order)}) [Average]: ").strip()
        if raw == "":
            chosen_norm = "average"
        else:
            key = raw.strip().lower().replace("-", " ").replace("_", " ")
            chosen_norm = synonyms.get(key, key)
    else:
        # programmatic preset provided or no prompt requested
        if preset is None:
            chosen_norm = "average"
        else:
            key = str(preset).strip().lower().replace("-", " ").replace("_", " ")
            chosen_norm = synonyms.get(key, key)

    # fallback to average if unrecognized
    if chosen_norm not in preset_map:
        chosen_norm = "average"

    height_scale, weight_scale = preset_map[chosen_norm]

    # RNG
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # Base per-position stats (means, stds, min/max). These are baseline "Average" values.
    pos_stats = {
        "QB":  {"h_mean": 75, "h_std": 2.0, "w_mean": 220, "w_std": 12, "h_min":72, "h_max":79, "w_min":190, "w_max":260},
        "RB":  {"h_mean": 71, "h_std": 1.8, "w_mean": 210, "w_std": 15, "h_min":67, "h_max":75, "w_min":180, "w_max":240},
        "FB":  {"h_mean": 71, "h_std": 1.5, "w_mean": 245, "w_std": 18, "h_min":69, "h_max":74, "w_min":220, "w_max":280},
        "WR":  {"h_mean": 72, "h_std": 2.5, "w_mean": 195, "w_std": 14, "h_min":68, "h_max":78, "w_min":165, "w_max":230},
        "TE":  {"h_mean": 76, "h_std": 2.0, "w_mean": 250, "w_std": 18, "h_min":74, "h_max":80, "w_min":220, "w_max":290},
        "LT":  {"h_mean": 78, "h_std": 1.8, "w_mean": 315, "w_std": 20, "h_min":76, "h_max":81, "w_min":280, "w_max":360},
        "LG":  {"h_mean": 77, "h_std": 1.8, "w_mean": 310, "w_std": 20, "h_min":75, "h_max":80, "w_min":275, "w_max":350},
        "C":   {"h_mean": 76, "h_std": 1.6, "w_mean": 305, "w_std": 18, "h_min":74, "h_max":79, "w_min":270, "w_max":340},
        "RG":  {"h_mean": 77, "h_std": 1.8, "w_mean": 310, "w_std": 20, "h_min":75, "h_max":80, "w_min":275, "w_max":350},
        "RT":  {"h_mean": 78, "h_std": 1.8, "w_mean": 315, "w_std": 20, "h_min":76, "h_max":81, "w_min":280, "w_max":360},
        "LE":  {"h_mean": 76, "h_std": 2.0, "w_mean": 270, "w_std": 20, "h_min":73, "h_max":80, "w_min":230, "w_max":320},
        "RE":  {"h_mean": 76, "h_std": 2.0, "w_mean": 270, "w_std": 20, "h_min":73, "h_max":80, "w_min":230, "w_max":320},
        "DT":  {"h_mean": 75, "h_std": 2.0, "w_mean": 300, "w_std": 25, "h_min":72, "h_max":79, "w_min":260, "w_max":340},
        "MIKE":{"h_mean": 74, "h_std": 1.8, "w_mean": 240, "w_std": 18, "h_min":71, "h_max":77, "w_min":210, "w_max":270},
        "SAM": {"h_mean": 73, "h_std": 1.8, "w_mean": 235, "w_std": 18, "h_min":70, "h_max":76, "w_min":205, "w_max":265},
        "WILL":{"h_mean": 72, "h_std": 1.8, "w_mean": 225, "w_std": 16, "h_min":69, "h_max":75, "w_min":195, "w_max":255},
        "CB":  {"h_mean": 71, "h_std": 2.0, "w_mean": 190, "w_std": 12, "h_min":68, "h_max":75, "w_min":165, "w_max":215},
        "FS":  {"h_mean": 72, "h_std": 2.0, "w_mean": 200, "w_std": 14, "h_min":69, "h_max":76, "w_min":175, "w_max":230},
        "SS":  {"h_mean": 72, "h_std": 2.0, "w_mean": 205, "w_std": 14, "h_min":69, "h_max":76, "w_min":180, "w_max":235},
        "K":   {"h_mean": 72, "h_std": 2.0, "w_mean": 200, "w_std": 12, "h_min":68, "h_max":76, "w_min":170, "w_max":230},
        "P":   {"h_mean": 72, "h_std": 2.0, "w_mean": 200, "w_std": 12, "h_min":68, "h_max":76, "w_min":170, "w_max":230},
    }

    default_stats = {"h_mean": 72, "h_std": 2.5, "w_mean": 200, "w_std": 20, "h_min":66, "h_max":80, "w_min":150, "w_max":320}

    # truncated normal sampler using rejection sampling
    def truncated_normal(rng, mean, std, lo, hi, size=1):
        out = np.empty(size, dtype=float)
        for i in range(size):
            for _ in range(1000):
                v = rng.normal(mean, std)
                if lo <= v <= hi:
                    out[i] = v
                    break
            else:
                out[i] = float(np.clip(mean, lo, hi))
        return out if size > 1 else out[0]

    # ensure position column exists
    if position_col not in df.columns:
        df[position_col] = ""

    n = len(df)
    heights = np.empty(n, dtype=float)
    weights = np.empty(n, dtype=float)
    size_preset_col = np.empty(n, dtype=object)

    for idx, pos in enumerate(df[position_col].astype(str).fillna("").values):
        p = pos.strip().upper()
        stats = pos_stats.get(p, default_stats)

        # apply preset scaling to means and min/max ranges
        h_mean = stats["h_mean"] * height_scale
        h_std  = max(0.8, stats["h_std"] * (0.9 + 0.2 * (height_scale - 1)))  # modest std adjustment
        h_min  = max(60, math.floor(stats["h_min"] * height_scale))
        h_max  = min(90, math.ceil(stats["h_max"] * height_scale))

        w_mean = stats["w_mean"] * weight_scale
        w_std  = max(6, stats["w_std"] * (0.9 + 0.25 * (weight_scale - 1)))
        w_min  = max(120, math.floor(stats["w_min"] * weight_scale))
        w_max  = min(450, math.ceil(stats["w_max"] * weight_scale))

        # sample height
        h = truncated_normal(rng, h_mean, h_std, h_min, h_max, size=1)

        # derive expected weight from height using slope logic (keeps height-weight correlation)
        mean_h = stats["h_mean"] * height_scale
        mean_w = stats["w_mean"] * weight_scale

        if mean_w >= 300:
            slope = 9.0
        elif mean_w >= 250:
            slope = 7.5
        elif mean_w >= 230:
            slope = 6.0
        elif mean_w >= 210:
            slope = 5.0
        else:
            slope = 4.0

        expected_w = mean_w + slope * (h - mean_h)
        w_noise = truncated_normal(rng, 0, w_std, -3*w_std, 3*w_std, size=1)
        w = float(np.clip(expected_w + w_noise, w_min, w_max))

        heights[idx] = round(float(np.clip(h, h_min, h_max)), 1)
        weights[idx] = round(w)
        size_preset_col[idx] = chosen_norm

    df[height_in_col] = heights
    df[weight_col] = weights.astype(int)

    def fmt_height(inches):
        ft = int(inches) // 12
        inch = int(round(inches)) % 12
        return f"{ft}'{inch}\""

    df[height_str_col] = df[height_in_col].apply(fmt_height)

    # write preset metadata both to attrs and a column for easy downstream checks
    df.attrs["height_weight_preset"] = chosen_norm
    df.attrs["height_scale"] = height_scale
    df.attrs["weight_scale"] = weight_scale
    df["size_preset"] = pd.Series(size_preset_col, index=df.index).astype(str)

    return df
# ---------- Overall bounds and skew ----------
def rebalance_overalls_by_year(roster_df: pd.DataFrame, max_shift=2) -> pd.DataFrame:
    """
    Conservative rebalance of player overalls by academic year.
    Shifts FR/SO down and JR/SR up slightly to reduce unrealistic distributions.
    """
    target_means = {
        y: int(round(roster_df["Overall"].mean() + YEAR_OVERALL_BIAS.get(y, 0)))
        for y in ACADEMIC_YEARS
    }
    for y, target in target_means.items():
        mask = roster_df["Year"] == y
        if mask.sum() == 0:
            continue
        current_mean = roster_df.loc[mask, "Overall"].mean()
        diff = target - current_mean
        if abs(diff) < 0.5:
            continue
        shift = np.sign(diff) * min(max_shift, abs(diff))
        idxs = roster_df.loc[mask].index
        for i in idxs:
            new_val = int(np.clip(roster_df.at[i, "Overall"] + shift, 50, 99))
            roster_df.at[i, "Overall"] = new_val
    return roster_df


def parse_overall_bounds(s: str):
    s = (s or "").strip()
    m = re.fullmatch(r"(\d{1,3})\s*-\s*(\d{1,3})", s)
    if not m:
        raise ValueError("Enter a range like 60-75.")
    lo, hi = int(m.group(1)), int(m.group(2))
    lo, hi = max(50, min(lo, hi)), min(99, max(lo, hi))
    if lo >= hi:
        raise ValueError("Lower bound must be less than upper bound.")
    return lo, hi

def parse_skew_direction(s: str):
    s = (s or "").strip().lower()
    if s in ("higher","high","upper"): return "higher"
    if s in ("lower","low","bottom"): return "lower"
    if s in ("center","middle","mid"): return "center"
    raise ValueError("Skew must be 'higher', 'lower', or 'center'.")

def parse_skew_amount(s: str):
    s = (s or "").strip().replace("%","")
    try:
        val = float(s)
    except Exception:
        raise ValueError("Enter a percentage like 30 or 30%.")
    val = max(0.0, min(100.0, val))
    return val / 100.0

def compute_mode(lo: int, hi: int, direction: str, amount01: float) -> float:
    mid = (lo + hi) / 2.0
    if direction == "center": return mid
    if direction == "higher": return mid + amount01 * (hi - mid)
    if direction == "lower": return mid - amount01 * (mid - lo)
    return mid

def sample_target_ovr_from_skew(lo: int, hi: int, direction: str, amount01: float) -> int:
    mode = compute_mode(lo, hi, direction, amount01)
    val = np.random.triangular(lo, mode, hi)
    return int(np.clip(round(val), lo, hi))

def sample_target_ovr_by_year(year: str, lo: int, hi: int, direction: str, amount01: float):
    base = sample_target_ovr_from_skew(lo, hi, direction, amount01)
    bias = YEAR_OVERALL_BIAS.get(year, 0)
    dampen = 0.6
    adjusted = int(round(base + bias * dampen))
    return int(np.clip(adjusted, lo, hi))
def blank_inapplicable_aggregates(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Blanks aggregate attributes for positions where they are not applicable.
    Keeps aggregate columns blank for everyone, and blanks other aggregates
    by position if needed.
    """
    AGG_ATTR_ALLOWED_POSITIONS = {
        "Ballcarrier": {"RB","FB","QB"},
        "Blocking": {"LT","LG","C","RG","RT","TE","FB"},
        "Passing": {"QB"},
        "Defense": {"LE","RE","DT","SAM","MIKE","WILL","CB","FS","SS"},
        "Receiving": {"WR","TE","RB","FB"},
        "Kicking": {"K","P"},
        "General": {"WR","RB","TE","CB","FS","SS","QB"}
    }
    cols_lower = {c.lower(): c for c in roster_df.columns}
    for agg_attr, allowed_positions in AGG_ATTR_ALLOWED_POSITIONS.items():
        colname = cols_lower.get(agg_attr.lower())
        if colname is None:
            continue
        expanded_allowed = set()
        for p in allowed_positions:
            if p == "OL":
                expanded_allowed.update({"LT","LG","C","RG","RT"})
            else:
                expanded_allowed.add(p)
        mask_not_allowed = ~roster_df["Position"].isin(expanded_allowed)
        roster_df.loc[mask_not_allowed, colname] = np.nan
    return roster_df



# ---------- Analysis loading ----------
def load_analysis(path):
    try:
        ratios = pd.read_excel(path, sheet_name=RATIOS_SHEET)
        corrs = pd.read_excel(path, sheet_name=CORRS_SHEET)
        ridge = pd.read_excel(path, sheet_name=RIDGE_SHEET)
    except Exception:
        ratios, corrs, ridge = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for df in (ratios, corrs, ridge):
        if not df.empty:
            df.columns = [c.strip() for c in df.columns]
            if "Attribute" in df.columns:
                df["AttrKey"] = df["Attribute"].apply(normalize_attr_name)
    return ratios, corrs, ridge

def normalize_analysis_tables(ratios, corrs, ridge):
    for df in (ratios, corrs, ridge):
        if "Attribute" in df.columns and "AttrKey" not in df.columns:
            df["AttrKey"] = df["Attribute"].apply(normalize_attr_name)
    return ratios, corrs, ridge

def remap_analysis_tables_to_provided(ratios, corrs, ridge):
    for df in (ratios, corrs, ridge):
        if df is None or df.empty:
            continue
        # Normalize Position column from analysis names to provided names
        # when remapping analysis tables to your generator vocabulary
        if "Position" in df.columns:
            df["Position"] = df["Position"].apply(lambda p: analysis_to_provided_position(p))
        # Now map archetype names using the normalized position
        if "Archetype" in df.columns and "Position" in df.columns:
            df["Archetype"] = df.apply(lambda r: map_analysis_archetype_to_provided(r["Archetype"], r["Position"]), axis=1)
    return ratios, corrs, ridge
def build_weight_vector_from_ridge(ridge_df, attr_keys):
    """
    ridge_df: DataFrame with columns ['AttrKey','Weight'] (AttrKey normalized)
    attr_keys: list of normalized attribute keys in the same order used for sampling
    Returns: (w, b) numpy arrays/scalars aligned to attr_keys
    """
    import numpy as _np
    if ridge_df is None or ridge_df.empty:
        return None, 0.0
    # Normalize keys in ridge_df
    ridge_map = {str(k).strip().lower(): float(v) for k, v in zip(ridge_df['AttrKey'], ridge_df['Weight'])}
    w = _np.array([ridge_map.get(k, 0.0) for k in attr_keys], dtype=float)
    # If intercept column exists
    b = 0.0
    if 'Intercept' in ridge_df.columns:
        try:
            b = float(ridge_df['Intercept'].dropna().iloc[0])
        except Exception:
            b = 0.0
    return w, b
# ---------- Bounds ----------
def compute_attribute_bounds_from_raw(raw_file: str) -> dict:
    try:
        df = pd.read_excel(raw_file)
    except Exception:
        return {}
    bounds = {}
    for col in df.columns:
        if col not in ["PlayerName","Position","Archetype","Overall","OVR"]:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                lo, hi = int(vals.min()), int(vals.max())
                lo = max(lo, 20)
                hi = min(hi, 99)
                bounds[normalize_attr_name(col)] = (lo, hi)
    return bounds



# ---------- Position-aware bounds (new) ----------
def detect_pos_arch_columns(df: pd.DataFrame):
    cols = [c for c in df.columns]
    lower_map = {c.strip().lower(): c for c in cols}
    pos_candidates = ["position","pos","pos_short","pos_long","pos_shortname"]
    arch_candidates = ["archetype","arch","type","archetype_name"]
    pos_col = None; arch_col = None
    for cand in pos_candidates:
        if cand in lower_map:
            pos_col = lower_map[cand]; break
    for cand in arch_candidates:
        if cand in lower_map:
            arch_col = lower_map[cand]; break
    if pos_col is None:
        for c in cols:
            if "pos" in str(c).strip().lower():
                pos_col = c; break
    if arch_col is None:
        for c in cols:
            lc = str(c).strip().lower()
            if "arch" in lc or "type" in lc:
                arch_col = c; break
    return pos_col, arch_col

def find_attribute_columns(df: pd.DataFrame):
    exclude = {"firstname","lastname","position","archetype","overall","jersey","year","devtrait","potential","handedness"}
    cols = []
    for c in df.columns:
        key = normalize_attr_name(c)
        if key in exclude: continue
        try:
            sample = pd.to_numeric(df[c], errors="coerce")
            if sample.notna().any(): cols.append(c)
        except Exception:
            continue
    return cols
def compute_attribute_bounds_by_position(raw_file: str, min_count_for_exact=10, pct_clip=(0.01,0.99)):
    """
    Returns (bounds_by_pos, global_bounds)
    - bounds_by_pos: { pos: { attrkey: (lo,hi), ... }, ... }
    - global_bounds: { attrkey: (lo,hi), ... } fallback
    """
    try:
        df = pd.read_excel(raw_file)
    except Exception:
        return {}, {}

    df.columns = [c.strip() for c in df.columns]
    pos_col, _ = detect_pos_arch_columns(df)
    if pos_col is None or pos_col not in df.columns:
        pos_col = "Position" if "Position" in df.columns else None

    # Normalize position labels to provided names
    if pos_col:
        df[pos_col] = df[pos_col].astype(str).fillna("").apply(lambda x: map_analysis_position_to_provided(x).strip())

    # Identify numeric attribute columns (exclude metadata)
    exclude = {"PlayerName","Player","Position","Archetype","Overall","OVR","FirstName","LastName","Jersey","Year","DevTrait","Potential","Handedness"}
    attr_cols = [c for c in df.columns if normalize_attr_name(c) not in exclude]

    # Global bounds (percentile clip to remove extreme outliers)
    global_bounds = {}
    for c in attr_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        lo = int(max(20, np.percentile(vals, pct_clip[0]*100)))
        hi = int(min(99, np.percentile(vals, pct_clip[1]*100)))
        global_bounds[normalize_attr_name(c)] = (lo, hi)

    bounds_by_pos = {}
    if pos_col:
        for pos, group in df.groupby(pos_col):
            pos_bounds = {}
            for c in attr_cols:
                vals = pd.to_numeric(group[c], errors="coerce").dropna()
                if len(vals) >= min_count_for_exact:
                    lo = int(max(20, vals.min()))
                    hi = int(min(99, vals.max()))
                elif len(vals) > 0:
                    lo = int(max(20, np.percentile(vals, pct_clip[0]*100)))
                    hi = int(min(99, np.percentile(vals, pct_clip[1]*100)))
                else:
                    key = normalize_attr_name(c)
                    if key in global_bounds:
                        pos_bounds[key] = global_bounds[key]
                    continue
                pos_bounds[normalize_attr_name(c)] = (lo, hi)
            bounds_by_pos[pos] = pos_bounds

    return bounds_by_pos, global_bounds
# compute position-aware bounds once and pass as the 'bounds' argument
bounds_by_pos, global_bounds = compute_attribute_bounds_by_position(RAW_FILE)
bounds_tuple = (bounds_by_pos, global_bounds)

# optional debug
print("DEBUG: computed bounds_by_pos keys sample:", list(bounds_by_pos.keys())[:5])


def clamp_by_position(attr_key, val, position, bounds_by_pos, global_bounds):
    """
    Clamp a single attribute value using position-specific bounds if available,
    otherwise fall back to global bounds, then to (20,99).
    `position` should be the provided/normalized position (e.g., "RB", "MIKE").
    """
    key = normalize_attr_name(attr_key)
    pos_map = bounds_by_pos.get(position, {}) if bounds_by_pos else {}
    if key in pos_map:
        lo, hi = pos_map[key]
    else:
        lo, hi = global_bounds.get(key, (20, 99))
    return clamp(val, lo, hi)
def _unpack_bounds(bounds):
    """
    Accept either:
      - old-style global bounds dict, or
      - tuple (bounds_by_pos, global_bounds)
    Returns (bounds_by_pos, global_bounds)
    """
    if bounds is None:
        return {}, {}
    if isinstance(bounds, tuple) and len(bounds) == 2:
        return bounds
    # assume old global dict
    return {}, bounds
# ---------- Names ----------
def load_names(first_file=FIRST_NAMES_FILE, last_file=LAST_NAMES_FILE):
    if not os.path.exists(first_file) or not os.path.exists(last_file):
        raise FileNotFoundError("first_names.txt or last_names.txt not found.")
    with open(first_file, "r", encoding="utf-8") as f:
        first_names = [l.strip() for l in f if l.strip()]
    with open(last_file, "r", encoding="utf-8") as f:
        last_names = [l.strip() for l in f if l.strip()]
    return first_names, last_names

def generate_name(first_names, last_names):
    return random.choice(first_names), random.choice(last_names)

# ---------- Jersey numbers ----------
ALLOWED_NUMBERS = {
    "QB": list(range(1, 20)),
    "RB": list(range(1, 50)) + list(range(80, 90)),
    "FB": list(range(1, 50)) + list(range(80, 90)),
    "WR": list(range(1, 20)) + list(range(80, 90)),
    "TE": list(range(1, 50)) + list(range(80, 90)),
    "OL": list(range(50, 80)),
    "LT": list(range(50, 80)),
    "LG": list(range(50, 80)),
    "C": list(range(50, 80)),
    "RG": list(range(50, 80)),
    "RT": list(range(50, 80)),
    "LE": list(range(50, 80)) + list(range(90, 100)),
    "RE": list(range(50, 80)) + list(range(90, 100)),
    "DT": list(range(50, 80)) + list(range(90, 100)),
    "SAM": list(range(1, 60)) + list(range(90, 100)),
    "MIKE": list(range(1, 60)) + list(range(90, 100)),
    "WILL": list(range(1, 60)) + list(range(90, 100)),
    "CB": list(range(1, 50)),
    "FS": list(range(1, 50)),
    "SS": list(range(1, 50)),
    "K": list(range(1, 50)) + list(range(90, 100)),
    "P": list(range(1, 50)) + list(range(90, 100)),
}

def assign_jersey_number(position, used_numbers):
    allowed = ALLOWED_NUMBERS.get(position)
    if allowed is None:
        allowed = ALLOWED_NUMBERS.get("OL") if position in ["LT","LG","C","RG","RT"] else list(range(1,100))
    choices = [n for n in allowed if n not in used_numbers]
    if not choices:
        choices = allowed
    num = random.choice(choices)
    used_numbers.add(num)
    return num

# ---------- Ledoit-Wolf stats builder ----------
def detect_pos_arch_columns(df: pd.DataFrame):
    cols = [c for c in df.columns]
    lower_map = {c.strip().lower(): c for c in cols}
    pos_candidates = ["position","pos","pos_short","pos_long","pos_shortname"]
    arch_candidates = ["archetype","arch","type","archetype_name"]
    pos_col = None; arch_col = None
    for cand in pos_candidates:
        if cand in lower_map:
            pos_col = lower_map[cand]; break
    for cand in arch_candidates:
        if cand in lower_map:
            arch_col = lower_map[cand]; break
    if pos_col is None:
        for c in cols:
            if "pos" in str(c).strip().lower():
                pos_col = c; break
    if arch_col is None:
        for c in cols:
            lc = str(c).strip().lower()
            if "arch" in lc or "type" in lc:
                arch_col = c; break
    return pos_col, arch_col

def find_attribute_columns(df: pd.DataFrame):
    exclude = {"firstname","lastname","position","archetype","overall","jersey","year","devtrait","potential","handedness"}
    cols = []
    for c in df.columns:
        key = normalize_attr_name(c)
        if key in exclude: continue
        try:
            sample = pd.to_numeric(df[c], errors="coerce")
            if sample.notna().any(): cols.append(c)
        except Exception:
            continue
    return cols

def build_ledoit_stats(raw_path):
    """
    Build per-(Position,Archetype) Ledoit-Wolf stats from a raw Excel file.
    This version normalizes analysis-style position labels (e.g., HB/MLB/LOLB/ROLB)
    back to the generator's provided position names (RB/MIKE/SAM/WILL) before
    grouping and archetype mapping so keys align with ARCHETYPES.
    Returns: stats dict, attr_cols list, raw dataframe, pos_col name, arch_col name
    """
    try:
        raw = pd.read_excel(raw_path)
    except Exception:
        return {}, [], None, None, None

    # normalize column names
    raw.columns = [c.strip() for c in raw.columns]

    pos_col, arch_col = detect_pos_arch_columns(raw)
    attr_cols = find_attribute_columns(raw)
    stats = {}

    if not (pos_col and arch_col and pos_col in raw.columns and arch_col in raw.columns):
        return stats, attr_cols, raw, pos_col, arch_col

    # Work on a copy to avoid mutating caller data
    df = raw.copy()

    # Normalize position and archetype text to consistent strings
    df[pos_col] = df[pos_col].astype(str).fillna("").apply(lambda x: str(x).strip())
    df[arch_col] = df[arch_col].astype(str).fillna("").apply(lambda x: str(x).strip())

    # inside build_ledoit_stats when creating _NormPos
    df["_NormPos"] = df[pos_col].apply(lambda p: analysis_to_provided_position(p) if p is not None else p)

    # Map archetype names to the provided archetype list using the normalized position
    def _map_arch(row):
        raw_arch = row[arch_col]
        norm_pos = row["_NormPos"]
        try:
            return map_analysis_archetype_to_provided(raw_arch, norm_pos)
        except Exception:
            return raw_arch
    df["_NormArch"] = df.apply(_map_arch, axis=1)

    # Determine numeric attribute columns (coerce to numeric)
    numeric = df[attr_cols].apply(pd.to_numeric, errors="coerce")

    # Group by normalized position and normalized archetype
    grouped = df.groupby(["_NormPos", "_NormArch"])

    for (p, a), group in grouped:
        numeric_group = group[attr_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        if numeric_group.shape[0] < 1 or numeric_group.shape[1] == 0:
            print(f"WARNING: insufficient data for {p}/{a}, skipping")
            continue

        mean_vals = numeric_group.mean(axis=0)
        try:
            if numeric_group.shape[0] > 1:
                lw = LedoitWolf().fit(numeric_group.fillna(0).values)
                cov = pd.DataFrame(lw.covariance_, index=numeric_group.columns, columns=numeric_group.columns)
            else:
                cov = pd.DataFrame(np.eye(len(numeric_group.columns)),
                                   index=numeric_group.columns, columns=numeric_group.columns)
        except Exception as e:
            print(f"WARNING: covariance failed for {p}/{a}: {e}")
            cov = numeric_group.cov().fillna(0)

        overall_mean = None
        if "Overall" in group.columns:
            overall_mean = pd.to_numeric(group["Overall"], errors="coerce").mean()
        elif "OVR" in group.columns:
            overall_mean = pd.to_numeric(group["OVR"], errors="coerce").mean()
        if overall_mean is None or np.isnan(overall_mean):
            overall_mean = mean_vals.mean()

        stats[(p or "", a or "")] = {
            "mean": mean_vals,
            "cov": cov,
            "n": numeric_group.shape[0],
            "overall_mean": float(overall_mean)
        }


    return stats, attr_cols, raw, pos_col, arch_col

# ---------- Multivariate sampler ----------
def synthesize_attributes_mv(attr_df, target_ovr, bounds, stats_obj, shrink=0.5, noise_scale=0.6, player_pos=None):
    """
    Multivariate synthesis using position-aware bounds.
    `bounds` may be (bounds_by_pos, global_bounds) or old global dict.
    `player_pos` should be provided (e.g., "RB", "MIKE"); if None, fallback to "".
    """
    bounds_by_pos, global_bounds = _unpack_bounds(bounds)
    if player_pos is None:
        player_pos = ""

    if attr_df is None or len(attr_df) == 0:
        if stats_obj is not None and "mean" in stats_obj and not stats_obj["mean"].empty:
            attrs = list(stats_obj["mean"].index)
            attr_df = pd.DataFrame({"Attribute": attrs, "AttrKey": [normalize_attr_name(a) for a in attrs]})
        else:
            return synthesize_attributes_simple(pd.DataFrame(), target_ovr, bounds, player_pos=player_pos)

    if "AttrKey" not in attr_df.columns:
        attr_df["AttrKey"] = attr_df["Attribute"].apply(normalize_attr_name)
    attr_df = attr_df[~attr_df["AttrKey"].isin(AGGREGATE_ATTR_KEYS)].reset_index(drop=True)

    keys = attr_df["AttrKey"].tolist()
    attrs_human = attr_df["Attribute"].tolist()

    if stats_obj is None or "mean" not in stats_obj or stats_obj["mean"].empty:
        return synthesize_attributes_simple(attr_df, target_ovr, bounds, player_pos=player_pos)

    # Build mean vector aligned to keys
    mean_series = stats_obj["mean"]
    mean_index_norm = {normalize_attr_name(k): k for k in mean_series.index}
    mean_vec = []
    for k in keys:
        if k in mean_index_norm:
            mean_vec.append(mean_series[mean_index_norm[k]])
        elif k in mean_series.index:
            mean_vec.append(mean_series[k])
        else:
            mean_vec.append(np.nan)
    mean_vec = np.array(mean_vec, dtype=float)
    if np.isnan(mean_vec).any():
        return synthesize_attributes_simple(attr_df, target_ovr, bounds, player_pos=player_pos)

    cov_df = stats_obj.get("cov", pd.DataFrame()).reindex(index=keys, columns=keys).fillna(0)
    cov = cov_df.values if not cov_df.empty else np.eye(len(keys))

    analysis_overall_mean = stats_obj.get("overall_mean", target_ovr)
    scale = target_ovr / max(analysis_overall_mean, 1.0)
    mu = mean_vec * scale

    cov_shrunk = cov * shrink + np.eye(len(keys)) * 1.0
    try:
        L = cholesky(cov_shrunk)
        z = np.random.normal(size=len(keys))
        sample = mu + L.dot(z) * noise_scale
    except LinAlgError:
        sample = mu + np.random.normal(0, 1.0, size=len(keys)) * noise_scale
    # --- Insert after sample is generated and before clamping ---
        # prepare attr_keys in the same order as 'keys' variable in synthesize_attributes_mv
        attr_keys = keys  # keys is attr_df["AttrKey"].tolist() earlier in the function

        # Build weight vector once (you can cache this outside the function for speed)
        # ridge_df should be loaded from your analysis file earlier; if not available, set ridge_df = None
        try:
            # If you have a global 'ridge' DataFrame from load_analysis, use it
            ridge_df_local = globals().get('ridge', None)
        except Exception:
            ridge_df_local = None

        w, b = build_weight_vector_from_ridge(ridge_df_local, attr_keys)

        # Prepare covariance matrix aligned to attr order
        Sigma = None
        if w is not None and 'cov' in stats_obj and stats_obj.get('cov') is not None:
            try:
                cov_df = stats_obj.get('cov', pd.DataFrame()).reindex(index=attr_df['Attribute'], columns=attr_df['Attribute']).fillna(0)
                # cov_df index/columns are human names; ensure alignment to keys if needed
                # If cov_df uses AttrKey names, reindex by attr_keys instead
                # Convert to numpy
                Sigma = cov_df.values if not cov_df.empty else None
            except Exception:
                Sigma = None

        # Project sample to match target overall using covariance-aware projection
        try:
            if w is not None and w.sum() != 0:
                sample = project_to_target_overall(sample, w, b, target_ovr, Sigma=Sigma)
            else:
                # fallback: scale mean vector approach (weaker but still improves correlation)
                # analysis_overall_mean is available earlier as stats_obj.get("overall_mean", target_ovr)
                analysis_mean = stats_obj.get("overall_mean", target_ovr)
                if analysis_mean and analysis_mean > 0:
                    scale = float(target_ovr) / float(max(analysis_mean, 1.0))
                    sample = sample * scale
        except Exception:
            # if projection fails, leave sample unchanged
            pass

        # Continue with clamping and conversion to ints as before

    out = {}
    for k, val, attr in zip(keys, sample, attrs_human):
        out[attr] = clamp_by_position(k, val, player_pos, bounds_by_pos, global_bounds)

    if not out:
        return synthesize_attributes_simple(attr_df, target_ovr, bounds, player_pos=player_pos)
    return out


def synthesize_attributes_simple(attr_df, target_ovr, bounds, noise_scale=0.6, player_pos=None):
    """
    Ratio-based fallback with position-aware clamping.
    `bounds` may be (bounds_by_pos, global_bounds) or old global dict.
    """
    bounds_by_pos, global_bounds = _unpack_bounds(bounds)
    if player_pos is None:
        player_pos = ""

    if "AttrKey" not in attr_df.columns:
        attr_df["AttrKey"] = attr_df["Attribute"].apply(normalize_attr_name)
    attr_df = attr_df[~attr_df["AttrKey"].isin(AGGREGATE_ATTR_KEYS)].reset_index(drop=True)

    if "MeanRatioAttrOverOverall" not in attr_df.columns:
        attr_df["MeanRatioAttrOverOverall"] = 0.8
    base = attr_df["MeanRatioAttrOverOverall"].fillna(attr_df["MeanRatioAttrOverOverall"].median()).clip(lower=0.5)
    dampen = 0.88
    base_vals = base.values * target_ovr * dampen
    corr = attr_df.get("PearsonCorrWithOverall", pd.Series([0]*len(base_vals))).fillna(0.0).values
    corr_clip = np.clip(corr, 0.0, 1.0)
    noise_std = (1.0 - corr_clip) * (noise_scale * 0.7)
    noisy = base_vals + np.random.normal(0, noise_std)
    clamped = []
    for attr, val, key in zip(attr_df["Attribute"].tolist(), noisy, attr_df["AttrKey"].tolist()):
        clamped.append(clamp_by_position(key, val, player_pos, bounds_by_pos, global_bounds))
    return dict(zip(attr_df["Attribute"].tolist(), clamped))

# ---------- Grid search ----------
def mean_abs_corr_diff(corr_a, corr_b):
    common = corr_a.index.intersection(corr_b.index)
    if len(common) < 2: return float('inf')
    a = corr_a.reindex(index=common, columns=common)
    b = corr_b.reindex(index=common, columns=common)
    diff = (a - b).abs().values
    np.fill_diagonal(diff, 0)
    return float(np.nanmean(diff))

def grid_search_params(stats, attr_list, bounds, analysis_corr, shrink_grid, noise_grid, sample_n=SAMPLE_N):
    best = None
    best_score = float('inf')
    results = []
    stat_keys = list(stats.keys())
    #print(stat_keys)
    if not stat_keys:
        return None, pd.DataFrame(results)
    for shrink in shrink_grid:
        for noise in noise_grid:
            regen = []
            for i in range(sample_n):
                key = random.choice(stat_keys)
                stats_obj = stats[key]
                overall_mean = stats_obj.get("overall_mean", 75)
                target_ovr = int(np.clip(int(np.random.normal(overall_mean, 6)), 50, 99))
                attr_df = pd.DataFrame({"Attribute": attr_list, "AttrKey": [normalize_attr_name(a) for a in attr_list]})
                attr_df = attr_df[~attr_df["AttrKey"].isin(AGGREGATE_ATTR_KEYS)].reset_index(drop=True)
                pos_for_sample = key[0] if isinstance(key, tuple) else ""
                attrs = synthesize_attributes_mv(attr_df, target_ovr, bounds, stats_obj, shrink, noise, player_pos=pos_for_sample)

                regen.append(attrs)
            regen_df = pd.DataFrame(regen)
            regen_corr = regen_df.corr()
            score = mean_abs_corr_diff(analysis_corr, regen_corr)
            results.append({"shrink": shrink, "noise": noise, "score": score})
            print(f"Grid try shrink={shrink}, noise={noise}, score={score:.4f}")
            if score < best_score:
                best_score = score
                best = {"shrink": shrink, "noise": noise, "score": score}
    return best, pd.DataFrame(results)

# ---------- Regression helper ----------
def compute_regression_weights(df_attrs: pd.DataFrame, target: pd.Series):
    X = df_attrs.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    var_mask = X.std(axis=0) > 1e-6
    X = X.loc[:, var_mask]
    if X.shape[1] == 0:
        return pd.Series(dtype=float)
    X_mat = X.values
    y = pd.to_numeric(target, errors="coerce").fillna(target.mean()).values
    try:
        coef, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
    except Exception:
        coef = np.zeros(X_mat.shape[1])
    coef = np.nan_to_num(coef)
    denom = np.sum(np.abs(coef))
    norm = coef if denom == 0 else coef / denom
    return pd.Series(norm, index=X.columns)

# ---------- Safe correlation ----------
def safe_corr(a: pd.Series, b: pd.Series, min_n: int = 3) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < min_n: return np.nan
    if df["a"].nunique() <= 1 or df["b"].nunique() <= 1: return np.nan
    return float(df["a"].corr(df["b"]))

# ---------- Graphs and diagnostics ----------
def plot_heatmap(mat: pd.DataFrame, path: str, title: str, vmin=-0.6, vmax=1.0):
    if mat is None or mat.empty: return
    plt.figure(figsize=(12,10))
    sns.heatmap(mat, cmap="coolwarm", center=0, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def generate_position_graphs(roster: pd.DataFrame, out_dir=GEN_DATA_DIR):
    os.makedirs(out_dir, exist_ok=True)
    roster["NormPosition"] = roster["Position"].apply(map_to_analysis_position)

    id_cols = {"FirstName","LastName","Position","NormPosition","Archetype","Overall","Jersey","Year","DevTrait","Potential","Handedness"}
    for pos in roster["NormPosition"].unique():
        sub = roster[roster["NormPosition"] == pos].copy()
        numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
        attr_cols = [c for c in numeric_cols if c not in id_cols]
        if not attr_cols:
            candidate_cols = [c for c in sub.columns if c not in id_cols]
            if candidate_cols:
                coerced = sub[candidate_cols].apply(pd.to_numeric, errors="coerce")
                attr_cols = [c for c in coerced.columns if coerced[c].notna().any()]
                for c in attr_cols:
                    sub[c] = coerced[c].fillna(0)
        if len(attr_cols) == 0:
            continue
        avg = sub.groupby("Archetype")[attr_cols].mean().T
        plt.figure(figsize=(12,6))
        sns.heatmap(avg, annot=False, cmap="coolwarm", cbar=True)
        plt.title(f"{pos} – Average Attributes by Archetype")
        plt.ylabel("Attribute")
        plt.xlabel("Archetype")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{pos}_heatmap.png")
        plt.close()

        plt.figure(figsize=(8,5))
        sns.histplot(sub["Overall"], bins=10, kde=True)
        plt.title(f"{pos} – Overall Distribution")
        plt.xlabel("Overall")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{pos}_overall.png")
        plt.close()



def generate_diagnostics_plots(roster: pd.DataFrame, out_dir=GEN_DATA_DIR):
    os.makedirs(out_dir, exist_ok=True)
    roster["NormPosition"] = roster["Position"].apply(map_to_analysis_position)

    id_cols = {"FirstName","LastName","Position","NormPosition","Archetype","Overall","Jersey","Year","DevTrait","Potential","Handedness"}
    for a in ["Speed","Strength","Awareness"]:
        if a in roster.columns:
            plt.figure(figsize=(6,4))
            sns.scatterplot(data=roster, x="Overall", y=a, hue="NormPosition", legend=False, alpha=0.7)
            plt.title(f"Overall vs {a}")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/overall_vs_{a}.png")
            plt.close()

    attr_cols = [c for c in roster.columns if c not in id_cols and roster[c].dtype in [int,float]]
    if len(attr_cols) >= 3:
        corr = roster[attr_cols].corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation matrix (generated roster)")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/generated_correlation_heatmap.png")
        plt.close()

# ---------- Export ----------
    '''print("EXPORT CHECK: roster shape:", roster.shape)
    print("EXPORT CHECK: roster columns:", roster.columns.tolist())
    print("EXPORT CHECK: sample rows:", roster.head(3).to_dict(orient="records"))
    print("EXPORT CHECK: numeric attr cols:", roster.select_dtypes(include=[np.number]).columns.tolist()[:40])
    print("EXPORT CHECK: contains NormPosition:", "NormPosition" in roster.columns)
'''
def export_roster(df: pd.DataFrame, output_path: str,
                  ridge: pd.DataFrame = pd.DataFrame(),
                  debug_info: dict = None):
    """
    Export the generated roster DataFrame to Excel with multiple sheets:
    - GeneratedRoster: full roster with id + attribute columns
    - Meta: export metadata
    - Summary: per-position averages and top attributes
    - Debug sheets: optional diagnostics if debug_info is provided
    """

    # Normalize column names for matching but keep original names
    col_norm = {c: normalize_attr_name(c) for c in df.columns}
    id_norm_set = {normalize_attr_name(x) for x in
                   ["FirstName","LastName","Position","Archetype","Overall",
                    "Jersey","Year","DevTrait","Potential","Handedness"]}

    # find actual id columns (preserve original names)
    id_cols = [c for c, n in col_norm.items() if n in id_norm_set]

    # Ensure common id columns exist (add placeholders if missing)
    for expected in ["FirstName","LastName","Position","Archetype","Overall"]:
        if expected not in df.columns and normalize_attr_name(expected) not in col_norm.values():
            df[expected] = ""  # placeholder so export keeps schema
            id_cols.append(expected)

    # Add aggregate attrs if missing (they must be present and blank)
    for agg in AGGREGATE_ATTRS:
        if agg not in df.columns:
            df[agg] = np.nan

    # Attributes are everything except id columns
    attr_cols = [c for c in df.columns if c not in id_cols]

    # Order: id columns (stable order) then attributes
    ordered = id_cols + attr_cols
    df = df.reindex(columns=ordered)

    # --- Export to Excel ---
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Full roster
        df.to_excel(writer, index=False, sheet_name="GeneratedRoster")

        # Meta sheet
        meta = pd.DataFrame({
            "Field": ["Exported On","Source Measures","Positions Covered","Total Players"],
            "Value": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                os.path.abspath(ANALYSIS_FILE) if os.path.exists(ANALYSIS_FILE) else "",
                ", ".join(sorted(df["Position"].unique().tolist())) if "Position" in df.columns else "",
                len(df)
            ]
        })
        meta.to_excel(writer, index=False, sheet_name="Meta")

        # Summary sheet
        summary_rows = []
        for pos in df["Position"].unique():
            sub = df[df["Position"] == pos]
            avg_ovr = sub["Overall"].mean()
            idc = set(id_cols)
            avg_attrs = sub[[c for c in df.columns if c not in idc]].mean(numeric_only=True).sort_values(ascending=False).head(5)
            top_attrs = ", ".join([f"{a} ({int(v)})" for a, v in avg_attrs.items()]) if len(avg_attrs) > 0 else ""
            ridge_sub = ridge[ridge["Position"] == pos] if "Position" in ridge.columns else pd.DataFrame()
            ridge_top = ridge_sub.sort_values("RidgeWeight", ascending=False).head(5) if not ridge_sub.empty else pd.DataFrame()
            ridge_attrs = ", ".join([f"{a} ({round(w,2)})" for a, w in zip(ridge_top.get("Attribute", []), ridge_top.get("RidgeWeight", []))])
            summary_rows.append({
                "Position": pos,
                "Average Overall": round(avg_ovr, 1),
                "Top 5 Attributes (avg)": top_attrs,
                "Top 5 Ridge Drivers": ridge_attrs
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

        # Optional debug sheets
        if debug_info:
            if "missing_expectations" in debug_info:
                pd.DataFrame(
                    [(k, ",".join(v[:200])) for k, v in debug_info["missing_expectations"].items()],
                    columns=["Position","MissingExpectations"]
                ).to_excel(writer, index=False, sheet_name="Debug_MissingExpectations")
            if "coverage" in debug_info:
                debug_info["coverage"].to_excel(writer, index=False, sheet_name="Debug_Coverage")

    print(f"Generated roster exported to: {output_path}")

# ---------- Validation ----------
def validate_and_report(roster: pd.DataFrame,
                        ratios: pd.DataFrame,
                        corrs: pd.DataFrame,
                        ridge: pd.DataFrame,
                        output_xlsx_path: str,
                        ratio_tol: float = 0.15,
                        corr_tol: float = 0.25,
                        ridge_tol: float = 0.30,
                        auto_fix: bool = False,
                        debug: bool = False):
    if "NormPosition" not in roster.columns:
        if "Position" not in roster.columns:
            raise KeyError("Roster must contain a Position column for validation.")
        # Try known mapping helpers in order of likelihood
        mapper = None
        for fn in ("analysis_to_provided_position", "map_analysis_position_to_provided",
                   "provided_to_analysis_position", "map_to_analysis_position"):
            if fn in globals() and callable(globals()[fn]):
                # choose the function that maps analysis->provided if available
                mapper = globals()[fn]
                break
        # Fallback: identity mapping
        if mapper is None:
            mapper = lambda x: str(x).strip() if pd.notna(x) else x
        roster = roster.copy()
        roster["NormPosition"] = roster["Position"].apply(lambda p: mapper(p) if pd.notna(p) else p)

    id_cols = {"FirstName","LastName","Position","NormPosition","Archetype","Overall",
               "Jersey","Year","DevTrait","Potential","Handedness"}
    candidate_cols = [c for c in roster.columns
                      if c not in id_cols and normalize_attr_name(c) not in AGGREGATE_ATTR_KEYS]
    roster[candidate_cols] = roster[candidate_cols].apply(pd.to_numeric, errors="coerce")

    ratios_lookup = ratios.set_index(["Position","Archetype","AttrKey"])["MeanRatioAttrOverOverall"].to_dict() if not ratios.empty else {}
    corr_lookup   = corrs.set_index(["Position","Archetype","AttrKey"])["PearsonCorrWithOverall"].to_dict() if not corrs.empty else {}
    ridge_lookup  = ridge.set_index(["Position","Archetype","AttrKey"])["RidgeWeight"].to_dict() if not ridge.empty else {}

    attr_key_map = {c: normalize_attr_name(c) for c in candidate_cols}
    rows, flags = [], []

    for (pos, arch), group in roster.groupby(["NormPosition","Archetype"]):
        if group.empty or len(group) < 1:
            continue
        mean_overall = group["Overall"].mean()
        attrs = [c for c in candidate_cols if group[c].notna().any()]
        if len(attrs) == 0:
            continue

        for attr in attrs:
            key = (pos, arch, attr_key_map[attr])
            ratio_expect = ratios_lookup.get(key)
            corr_expect  = corr_lookup.get(key)
            ridge_expect = ridge_lookup.get(key)

            ratio_actual = group[attr].mean() / mean_overall if mean_overall > 0 else np.nan
            corr_actual  = safe_corr(group[attr], group["Overall"])
            reg_weights  = compute_regression_weights(group[attrs], group["Overall"])
            ridge_actual = reg_weights.get(attr, np.nan)

            rows.append({
                "Position": pos,
                "Archetype": arch,
                "Attribute": attr,
                "RatioActual": ratio_actual,
                "RatioExpect": ratio_expect,
                "CorrActual": corr_actual,
                "CorrExpect": corr_expect,
                "RidgeActual": ridge_actual,
                "RidgeExpect": ridge_expect
            })

    result_df = pd.DataFrame(rows)
    with pd.ExcelWriter("output_diagnostics", engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Validation")
    return result_df



# ---------- Diagnostics helpers ----------
def report_missing_expectations(ratios, roster):
    id_cols = {"FirstName","LastName","Position","Archetype","Overall","Jersey","Year","DevTrait","Potential","Handedness"}
    generated_keys = set([normalize_attr_name(c) for c in roster.columns if c not in id_cols])
    missing = {}
    if "AttrKey" in ratios.columns:
        for pos in roster["Position"].unique():
            analysis_pos = map_to_analysis_position(pos)
            expected_keys = set(ratios[ratios["Position"]==analysis_pos]["AttrKey"].dropna().unique())
            missing[pos] = sorted(list(expected_keys - generated_keys))
    else:
        for pos in roster["Position"].unique():
            missing[pos] = []
    return missing

def coverage_table(roster):
    id_cols = {"FirstName","LastName","Position","Archetype","Overall","Jersey","Year","DevTrait","Potential","Handedness"}
    cols = [c for c in roster.columns if c not in id_cols]
    report = []
    for pos, g in roster.groupby("Position"):
        row = {"Position": pos, "Players": len(g)}
        for c in cols:
            row[c] = int(g[c].notna().sum())
        report.append(row)
    return pd.DataFrame(report)

# ---------- Deterministic archetype distribution helper ----------
def compute_arch_counts_for_position(pos: str, total_count: int, ratios_df: pd.DataFrame):
    """
    Determine deterministic counts per archetype for a given position.
    If ratios_df contains archetype frequency info for this position, use it to proportionally allocate counts.
    Otherwise, distribute evenly across provided ARCHETYPES[pos].
    Returns a dict {arch_name: count}.
    """
    arch_list = get_arch_list_for_pos(pos)
    if ratios_df is None or ratios_df.empty:
        base = total_count // len(arch_list)
        rem = total_count % len(arch_list)
        return {arch: base + (1 if i < rem else 0) for i, arch in enumerate(arch_list)}
    # attempt to compute frequencies from ratios_df: count rows per archetype for this position
    analysis_pos = map_to_analysis_position(pos)
    subset = ratios_df[ratios_df["Position"] == analysis_pos] if "Position" in ratios_df.columns else pd.DataFrame()
    if subset.empty:
        base = total_count // len(arch_list)
        rem = total_count % len(arch_list)
        return {arch: base + (1 if i < rem else 0) for i, arch in enumerate(arch_list)}
    # count occurrences per archetype (after remapping)
    counts = subset["Archetype"].value_counts().to_dict()
    # map counts to provided archetypes; if an analysis archetype doesn't match, it was remapped earlier
    # ensure every provided archetype has at least 0
    freq = {arch: counts.get(arch, 0) for arch in arch_list}
    total_freq = sum(freq.values())
    if total_freq == 0:
        base = total_count // len(arch_list)
        rem = total_count % len(arch_list)
        return {arch: base + (1 if i < rem else 0) for i, arch in enumerate(arch_list)}
    # allocate proportionally, then fix rounding by distributing remainder to largest fractional parts
    raw_alloc = {arch: (freq[arch] / total_freq) * total_count for arch in arch_list}
    floored = {arch: int(math.floor(v)) for arch, v in raw_alloc.items()}
    allocated = sum(floored.values())
    remainder = total_count - allocated
    # compute fractional parts
    fracs = sorted([(arch, raw_alloc[arch] - floored[arch]) for arch in arch_list], key=lambda x: x[1], reverse=True)
    for i in range(remainder):
        arch = fracs[i % len(fracs)][0]
        floored[arch] += 1
    return floored

# ---------- Main ----------
def main(custom_name=None,
         overall_range_input=None,
         skew_dir_input=None,
         skew_amt_input=None,
         allow_outliers=None,
         auto_fix=None,
         debug=None):
    """
    Roster generator entrypoint.
    - If arguments are provided, they are used directly.
    - If arguments are None, the function will prompt interactively.
    Returns: roster_df (pandas.DataFrame) of generated players.
    """

    print("Roster generator starting.")

    # --- filename ---
    if custom_name is None:
        custom_name = input("Enter a name for the generated roster spreadsheet (without extension): ").strip()
        if not custom_name:
            custom_name = "generated_roster"
    output_file = f"{custom_name}.xlsx"

    # --- overall range ---
    if overall_range_input is None:
        overall_range_input = input("Enter player overall range (e.g., 60-75): ").strip()
    try:
        ovr_lo, ovr_hi = parse_overall_bounds(overall_range_input)
    except Exception:
        print("Invalid overall range; using default 60-75.")
        ovr_lo, ovr_hi = 60, 75

    midpoint = (ovr_lo + ovr_hi) / 2.0
    print(f"Range selected: {ovr_lo}-{ovr_hi}. Midpoint: {midpoint:.1f}")
    print(f"Skew examples: higher 30% -> {compute_mode(ovr_lo, ovr_hi, 'higher', 0.30):.1f}, "
          f"lower 50% -> {compute_mode(ovr_lo, ovr_hi, 'lower', 0.50):.1f}, center -> {midpoint:.1f}")
    try:
        direction, amount01 = read_skew_prompt_cli("Choose skew direction and percent (e.g. 'higher 30%'): ")
    except ValueError as e:
        print("Input error:", e)
        direction, amount01 = "center", 0.0
    skew_dir=direction
    skew_amt01=amount01


    # --- flags ---
    if allow_outliers is None:
        allow_outliers = input("Allow athletic outliers (e.g., 99 Speed)? (y/N): ").strip().lower() == "y"
    if auto_fix is None:
        auto_fix = input("Run validation and attempt conservative auto-fix if flagged? (y/N): ").strip().lower() == "y"
    if debug is None:
        debug = input("Enable debug diagnostics (prints and writes missing/coverage)? (y/N): ").strip().lower() == "y"

    # Load analysis and bounds
    ratios, corrs, ridge = load_analysis(ANALYSIS_FILE)
    ratios, corrs, ridge = normalize_analysis_tables(ratios, corrs, ridge)
    ratios, corrs, ridge = remap_analysis_tables_to_provided(ratios, corrs, ridge)
    # compute position-aware bounds once
    bounds_by_pos, global_bounds = compute_attribute_bounds_by_position(RAW_FILE)
    bounds = (bounds_by_pos, global_bounds)   # use this tuple as the new 'bounds' argument
    first_names, last_names = load_names()

    # Build LedoitWolf stats and run grid search diagnostics
    stats, attr_cols, raw_df, pos_col, arch_col = build_ledoit_stats(RAW_FILE)
    best = None; grid_results = pd.DataFrame()
    if raw_df is None or raw_df.empty or not attr_cols:
        print("Warning: raw data missing or no numeric attributes; skipping grid search.")
    else:
        analysis_corr = raw_df[attr_cols].corr()
        best, grid_results = grid_search_params(stats,attr_cols,bounds_tuple,analysis_corr,SHRINK_GRID,NOISE_GRID,sample_n=1000)
        print("Grid search complete. Best params:", best)
        with pd.ExcelWriter(DIAGNOSTIC_XLSX, engine="openpyxl") as writer:
            analysis_corr.to_excel(writer, sheet_name="Analysis_Corr")
            grid_results.to_excel(writer, sheet_name="Grid_Results", index=False)

    # Roster generation with deterministic archetype distribution
    roster_rows = []
    used_numbers = set()
    for pos, count in POSITIONS.items():
        analysis_pos = map_to_analysis_position(pos)
        arch_list = get_arch_list_for_pos(analysis_pos)

        arch_counts = compute_arch_counts_for_position(analysis_pos, count, ratios if not ratios.empty else pd.DataFrame())
        total_alloc = sum(arch_counts.values())
        if total_alloc != count:
            diff = count - total_alloc
            keys = list(arch_counts.keys())
            for i in range(abs(diff)):
                k = keys[i % len(keys)]
                arch_counts[k] = arch_counts.get(k, 0) + (1 if diff > 0 else -1)

        for arch in arch_list:
            n = arch_counts.get(arch, 0)
            for i in range(n):
                year = random.choice(ACADEMIC_YEARS)
                target_ovr = sample_target_ovr_by_year(year, ovr_lo, ovr_hi, skew_dir, skew_amt01)
                dev = pick_dev_trait_by_year(year)
                pot = pick_potential()

                if USE_ANALYSIS_ARCH and not ratios.empty:
                    r = ratios[(ratios["Position"] == analysis_pos) & (ratios["Archetype"] == arch)].copy()
                    c = corrs[(corrs["Position"] == analysis_pos) & (corrs["Archetype"] == arch)].copy()
                    w = ridge[(ridge["Position"] == analysis_pos) & (ridge["Archetype"] == arch)].copy()
                    for df in (r, c, w):
                        if "AttrKey" not in df.columns and "Attribute" in df.columns:
                            df["AttrKey"] = df["Attribute"].apply(normalize_attr_name)
                    if not r.empty:
                        attr_df = r[["Attribute","AttrKey","MeanRatioAttrOverOverall"]].merge(
                            c[["AttrKey","PearsonCorrWithOverall"]], on="AttrKey", how="outer"
                        ).merge(
                            w[["AttrKey","RidgeWeight"]], on="AttrKey", how="outer"
                        )
                        attr_df["Attribute"] = attr_df["Attribute"].fillna(attr_df["AttrKey"])
                        attr_df = attr_df[["Attribute","AttrKey","MeanRatioAttrOverOverall","PearsonCorrWithOverall","RidgeWeight"]]
                        attr_df = attr_df.dropna(subset=["AttrKey"]).reset_index(drop=True)
                    else:
                        attr_df = pd.DataFrame({
                            "Attribute": ["Speed","Acceleration","Agility","Strength","Awareness","Toughness"],
                            "AttrKey": [normalize_attr_name(x) for x in ["Speed","Acceleration","Agility","Strength","Awareness","Toughness"]],
                            "MeanRatioAttrOverOverall": [0.9,0.9,0.85,0.95,0.8,0.85],
                            "PearsonCorrWithOverall": [0.6,0.6,0.5,0.7,0.5,0.5],
                            "RidgeWeight": [0.2,0.15,0.1,0.25,0.1,0.1]
                        })
                else:
                    attr_df = pd.DataFrame({
                        "Attribute": ["Speed","Acceleration","Agility","Strength","Awareness","Toughness"],
                        "AttrKey": [normalize_attr_name(x) for x in ["Speed","Acceleration","Agility","Strength","Awareness","Toughness"]],
                        "MeanRatioAttrOverOverall": [0.9,0.9,0.85,0.95,0.8,0.85],
                        "PearsonCorrWithOverall": [0.6,0.6,0.5,0.7,0.5,0.5],
                        "RidgeWeight": [0.2,0.15,0.1,0.25,0.1,0.1]
                    })

                if not attr_df.empty and "AttrKey" in attr_df.columns:
                    attr_df = attr_df[~attr_df["AttrKey"].isin(AGGREGATE_ATTR_KEYS)].reset_index(drop=True)

                stats_key = (analysis_pos, arch)
                stats_obj = stats.get(stats_key, None) if stats else None
                shrink_used = best["shrink"] if isinstance(best, dict) else 0.5
                noise_used = best["noise"] if isinstance(best, dict) else 0.6
                player_pos = map_analysis_position_to_provided(analysis_pos) if analysis_pos is not None else position
                attrs = synthesize_attributes_mv(attr_df, target_ovr, bounds, stats_obj, shrink_used, noise_used, player_pos)
                # fallback call (if used directly)


                if not allow_outliers:
                    for attr in list(attrs.keys()):
                        key = normalize_attr_name(attr)
                        if key in bounds and attrs[attr] >= 98:
                            lo, hi = bounds[key]
                            attrs[attr] = clamp((lo + hi) / 2.0, lo, hi)

                fname, lname = generate_name(first_names, last_names)
                jersey = assign_jersey_number(analysis_pos, used_numbers)
                handed = pick_handedness(analysis_pos)
                row = {"FirstName": fname, "LastName": lname,
                       "Position": analysis_pos, "Archetype": arch,
                       "Overall": target_ovr, "Jersey": jersey, "Year": year,
                       "DevTrait": dev, "Potential": pot, "Handedness": handed}
                row.update(attrs)
                roster_rows.append(row)

    # Build DataFrame with normalized positions
    roster = pd.DataFrame(roster_rows)
    roster["PosIndex"] = roster["Position"].map(lambda p: POSITION_INDEX.get(p, 999))
    roster = roster.sort_values(["PosIndex","Overall"], ascending=[True,False]).drop(columns=["PosIndex"])
    try:
        canonical_prompt = read_majority_prompt_cli()
    except ValueError as e:
        print("Input error:", e)
        canonical_prompt = "majority 3"   # safe default

    roster = roster = hs_star_rating(roster, prompt=canonical_prompt, col_name="hs star rating", overwrite=False)
    roster = add_height_weight_with_presets(roster, seed=42, overwrite=False)



    # Ensure aggregate columns exist and are blank for everyone
    for agg in AGGREGATE_ATTRS:
        roster[agg] = np.nan

    # Blank inapplicable aggregates
    roster = blank_inapplicable_aggregates(roster)

    # Optional conservative rebalance by year
    roster = rebalance_overalls_by_year(roster)
    # Debug diagnostics
    debug_info = None
    if debug:
        missing = report_missing_expectations(ratios, roster)
        coverage = coverage_table(roster)
        debug_info = {"missing_expectations": missing, "coverage": coverage}
        print("Debug: missing expectations (sample):")
        for k, v in list(missing.items())[:8]:
            print(f" - {k}: {len(v)} missing (showing up to 10): {v[:10]}")
        print("Debug: coverage head:")
        print(coverage.head())

    # Define ID_COLS once globally
    ID_COLS = ["FirstName","LastName","Position","NormPosition","Archetype",
               "Overall","Jersey","Year","DevTrait","Potential","Handedness"]

    # Export roster
    
    #Generate previous redshirt collumn
    roster = add_previous_redshirt_column(roster, col_name="previous redshirt", yes_prob=0.15, overwrite=False, random_seed=42)

    roster = add_dealbreaker_column(roster, col_name="dealbreaker", equal=True, overwrite=False, seed=123)

    export_roster(roster, output_file, ridge, debug_info=debug_info)

    # Validation + optional auto-fix
    validation_result = validate_and_report(roster, ratios, corrs, ridge, output_file,
                                            auto_fix=auto_fix, debug=debug)
    print("Validation result:", validation_result)

    # Graphs
    try:
        generate_position_graphs(roster)
        generate_diagnostics_plots(roster)
    except Exception as e:
        print("Warning: diagnostics generation encountered an error:", e)

    print("Done. Check the generated Excel file and GEN_Data folder for graphs, diagnostics, and validation sheets.")
    return roster

if __name__ == "__main__":
    main()

