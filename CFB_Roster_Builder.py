
import random
import os
from datetime import datetime


import pandas as pd

# Optional: reproducibility
# random.seed(42)


# Define possible positions with typical counts for a college football roster
POSITIONS = {
    "QB": 5,
    "RB": 6,
    "FB":1,
    "WR": 9,
    "TE": 5,
    "LT": 3,
    "LG": 3,
    "C": 3,
    "RG": 3,
    "RT": 3,
    "LE": 4,
    "RE": 4,
    "DT": 6,
    "SAM": 4,
    "MIKE": 4,
    "WILL": 4,
    "CB": 8,
    "FS":3,
    "SS":3,
    "K": 2,
    "P": 1,
}

# Desired sort order (QB first, P last)
POSITION_ORDER = ["QB", "RB", "WR", "TE", "LT", "LG","C","RG","RT","LE","RE","DT",
                  "SAM", "MIKE", "WILL", "CB", "FS", "SS", "K", "P"]
POSITION_INDEX = {pos: i for i, pos in enumerate(POSITION_ORDER)}

ACADEMIC_YEARS = ["FR", "SO", "JR", "SR"]

# Position-based allowed jersey number ranges (adjustable)
ALLOWED_NUMBERS = {
    "QB": list(range(0, 20)),                       # 1–19
    "RB": list(range(0, 50)) + list(range(80, 90)),
    "FB": list(range(0, 50)) + list(range(80, 90)),# 1–49, 80–89
    "WR": list(range(0, 20)) + list(range(80, 90)),
    "TE": list(range(0, 50)) + list(range(80, 90)),
    "OL": list(range(50, 80)),
    "LT": list(range(50, 80)),
    "LG": list(range(50, 80)),
    "C": list(range(50, 80)),
    "RG": list(range(50, 80)),
    "RT": list(range(50, 80)),# 50–79
    "LE": list(range(50, 80)) + list(range(90, 100)),
    "RE": list(range(50, 80)) + list(range(90, 100)),
    "DT": list(range(50, 80)) + list(range(90, 100)),
    "SAM": list(range(0, 60)) + list(range(90, 100)),
    "MIKE": list(range(0, 60)) + list(range(90, 100)),
    "WILL": list(range(0, 60)) + list(range(90, 100)),
    "CB": list(range(0, 50)),
    "FS": list(range(0, 50)),
    "SS": list(range(0, 50)),
    "K": list(range(0, 50)) + list(range(90, 100)),
    "P": list(range(0, 50)) + list(range(90, 100)),
}

# Dev trait distribution (increasingly rare)
DEV_TRAITS = ["Normal", "Impact", "Star", "Superstar"]
DEV_WEIGHTS = [0.70, 0.20, 0.08, 0.02]  # sum to 1

# Potential distribution (median most common)
POTENTIALS = ["Low", "Medium", "High"]
POT_WEIGHTS = [0.20, 0.60, 0.20]

# Handedness realistic distribution by position (probability of Right-handed)
HANDEDNESS_BY_POSITION = {
    # Most positions heavily right-handed; kickers/punters slightly more varied
   "QB": 0.90,
    "RB": 0.85,
    "WR": 0.85,
    "TE": 0.88,
    "LT": 0.95,
    "LG": 0.95,
    "C": 0.95,
    "RG": 0.95,
    "RT": 0.95,
    "LE": 0.95,
    "RE": 0.95,
    "DT": 0.95,
    "MIKE": 0.90,
    "SAM": 0.90,
    "WILL": 0.90,
    "CB": 0.90,
    "FS": 0.90,
    "SS": 0.90,
    "K": 0.80,
    "P": 0.80,
}

# Positional archetypes (examples; expand as desired)
ARCHETYPES = {
    "QB": ["Pocket Passer", "Backfield Creator", "Dual Threat", "Pure Runner"],
    "RB": ["Contact Seeker", "East/West Playmaker", "Backfield Threat", "North/South Blocker", "North/South Receiver", "Elusive Bruiser"],
    "FB": ["Blocking", "Utility"],
    "WR": ["Route Artist", "Speedster", "Physical Route Runner", "Elusive Route Runner", "Gritty Possession", "Gadget", "Contested Specialist"],
    "TE": ["Physical Route Runner", "Vertical Threat", "Pure Blocker", "Gritty Possession", "Pure Possession"],
    "LT": ["Pass Protector", "Raw Strength", "Well Rounded", "Agile"],
    "LG": ["Pass Protector", "Raw Strength", "Well Rounded", "Agile"],
    "C": ["Pass Protector", "Raw Strength", "Well Rounded", "Agile"],
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
    "P": ["Power", "Accurate"],
}

# Name loading (optional external files)
def load_names(filename, fallback):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
            if names:
                return names
    return fallback

FIRST_NAMES = load_names(
    "first_names.txt",
    ["James","Michael","Robert","John","David","Chris","Daniel","Joseph","Kevin","Brian",
     "Ethan","Noah","Liam","Mason","Logan","Aiden","Caleb","Owen","Wyatt","Lucas"]
)
LAST_NAMES = load_names(
    "last_names.txt",
    ["Smith","Johnson","Williams","Brown","Jones","Miller","Davis","Garcia","Rodriguez","Wilson",
     "Martinez","Anderson","Taylor","Thomas","Hernandez","Moore","Martin","Jackson","Thompson","White"]
)

def generate_name_parts():
    """Return (first_name, last_name)."""
    return random.choice(FIRST_NAMES), random.choice(LAST_NAMES)

def distribute_years(count):
    weights = {"FR": 0.30, "SO": 0.30, "JR": 0.20, "SR": 0.20}
    return random.choices(ACADEMIC_YEARS, weights=[weights[y] for y in ACADEMIC_YEARS], k=count)

def position_overall_base(position):
    bases = {"QB": 68,"RB": 67,"WR": 66,"TE": 65,"OL": 67,"DL": 66,"LB": 66,"DB": 66,"K": 64,"P": 64,"LS": 63}
    return bases.get(position, 65)

def generate_overall(position, year):
    base = position_overall_base(position)
    year_mod = {"FR": -2, "SO": -1, "JR": 0, "SR": +1}[year]
    variance = random.gauss(0, 4)
    overall = base + year_mod + variance
    return max(50, min(85, round(overall)))

def rescale_to_range(overalls, min_avg=65, max_avg=73):
    baseline = 50
    avg = sum(overalls) / len(overalls)
    target = random.uniform(min_avg, max_avg)

    if abs(avg - target) < 0.05 and min_avg <= avg <= max_avg:
        return overalls, avg

    denom = (avg - baseline)
    s = (target - baseline) / denom if abs(denom) > 1e-6 else 1.0
    scaled = [baseline + (o - baseline) * s for o in overalls]
    scaled = [max(50, min(85, round(x))) for x in scaled]
    new_avg = sum(scaled) / len(scaled)

    for _ in range(10):
        if min_avg <= new_avg <= max_avg:
            break
        s *= 1.02 if new_avg < min_avg else 0.98
        scaled = [baseline + (o - baseline) * s for o in overalls]
        scaled = [max(50, min(85, round(x))) for x in scaled]
        new_avg = sum(scaled) / len(scaled)

    if new_avg < min_avg or new_avg > max_avg:
        target = min(max(new_avg, min_avg), max_avg)
        denom2 = (sum(overalls) / len(overalls) - baseline)
        s = (target - baseline) / denom2 if abs(denom2) > 1e-6 else 1.0
        scaled = [baseline + (o - baseline) * s for o in overalls]
        scaled = [max(50, min(85, round(x))) for x in scaled]
        new_avg = sum(scaled) / len(scaled)

    return scaled, new_avg

def assign_numbers_by_position(positions_map, allowed_map, max_attempts=2000):
    # Pre-check
    for pos, count in positions_map.items():
        allowed = set(allowed_map.get(pos, range(1,100)))
        if len(allowed) < count:
            raise ValueError(f"Not enough allowed numbers for {pos}: need {count}, have {len(allowed)}")

    pos_order = sorted(positions_map.keys(), key=lambda p: -positions_map[p])

    for attempt in range(max_attempts):
        used = set()
        assignment = {p: [] for p in positions_map}
        success = True

        for pos in pos_order:
            allowed = list(set(allowed_map.get(pos, range(1,100))))
            random.shuffle(allowed)
            available = [n for n in allowed if n not in used]
            if len(available) < positions_map[pos]:
                success = False
                break
            chosen = available[:positions_map[pos]]
            assignment[pos] = chosen
            used.update(chosen)

        if success:
            flat = []
            for pos in positions_map:
                flat.extend(assignment[pos])
            return flat

    raise ValueError(
        "Unable to assign unique numbers with the given ALLOWED_NUMBERS configuration. "
        "Try expanding ranges or reducing overlap between positions."
    )

def choose_dev_trait():
    return random.choices(DEV_TRAITS, weights=DEV_WEIGHTS, k=1)[0]

def choose_potential():
    return random.choices(POTENTIALS, weights=POT_WEIGHTS, k=1)[0]

def choose_handedness(position):
    right_prob = HANDEDNESS_BY_POSITION.get(position, 0.90)
    return "Right" if random.random() < right_prob else "Left"

def choose_archetype(position):
    choices = ARCHETYPES.get(position)
    if not choices:
        return "General"
    return random.choice(choices)

def generate_roster():
    total_players = sum(POSITIONS.values())

    numbers_flat = assign_numbers_by_position(POSITIONS, ALLOWED_NUMBERS)
    years = distribute_years(total_players)

    roster = []
    idx_num = 0
    idx_year = 0
    for position, count in POSITIONS.items():
        for _ in range(count):
            first, last = generate_name_parts()
            roster.append({
                "FirstName": first,
                "LastName": last,
                "Number": numbers_flat[idx_num],
                "Position": position,
                "Year": years[idx_year],
                # placeholders for attributes to be filled later
            })
            idx_num += 1
            idx_year += 1

    # Generate overalls and other attributes
    overalls = [generate_overall(p["Position"], p["Year"]) for p in roster]
    overalls, avg = rescale_to_range(overalls, min_avg=65, max_avg=73)

    for p, o in zip(roster, overalls):
        p["Overall"] = o
        p["DevTrait"] = choose_dev_trait()
        p["Potential"] = choose_potential()
        p["Handedness"] = choose_handedness(p["Position"])
        p["Archetype"] = choose_archetype(p["Position"])

    return roster, avg

def export_to_excel(roster, avg_overall, filename):
    df = pd.DataFrame(roster, columns=[
        "FirstName", "LastName", "Number", "Position", "Year", "Overall",
        "DevTrait", "Potential", "Handedness", "Archetype"
    ])

    # Map position order and ensure Overall is numeric
    df["PosOrder"] = df["Position"].map(lambda p: POSITION_INDEX.get(p, 999))
    df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce").fillna(0).astype(int)

    # Sort by position order, then by Overall descending within each position, then by Number
    df = df.sort_values(by=["PosOrder", "Overall", "Number"], ascending=[True, False, True]).drop(columns=["PosOrder"]).reset_index(drop=True)

    meta = pd.DataFrame({
        "Field": ["Generated On", "Total Players", "Average Overall", "Name Sources"],
        "Value": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(df),
            f"{avg_overall:.2f}",
            f"first_names.txt ({'found' if os.path.exists('first_names.txt') else 'fallback'}), "
            f"last_names.txt ({'found' if os.path.exists('last_names.txt') else 'fallback'})"
        ]
    })

    # Ensure .xlsx extension
    if not filename.lower().endswith(".xlsx"):
        filename = filename + ".xlsx"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Roster")
        meta.to_excel(writer, index=False, sheet_name="Meta")

    return filename

def prompt_filename():
    while True:
        name = input("Enter a name for the new spreadsheet (no extension): ").strip()
        if not name:
            print("Filename cannot be empty. Try again.")
            continue
        forbidden = '<>:"/\\|?*'
        cleaned = "".join(ch for ch in name if ch not in forbidden)
        if not cleaned:
            print("Filename invalid after sanitization. Try a different name.")
            continue
        return cleaned + ".xlsx"
def prompt_yes_no(prompt):
    """Ask a yes/no question and return True for yes, False for no."""
    while True:
        ans = input(f"{prompt} (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def main():
    try:
        while True:
            roster, avg = generate_roster()
            filename = prompt_filename()
            try:
                out_file = export_to_excel(roster, avg, filename)
                print(f"Excel roster written to: {out_file}")
                print(f"Total players: {len(roster)}  Average Overall: {avg:.2f}")
            except Exception as e:
                print(f"Error exporting to Excel: {e}")

            # Ask user if they'd like to run again
            again = prompt_yes_no("Would you like to generate another roster")
            if not again:
                print("Done. Goodbye.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


   




if __name__ == "__main__":
    main()


