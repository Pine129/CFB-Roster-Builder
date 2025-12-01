import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from attributes_integration import generate_attributes_for_roster, load_schema
from CFB_Roster_Builder import generate_roster

runs = 100
all_rows = []
for i in range(runs):
    roster, _avg = generate_roster()
    roster_with_attrs = generate_attributes_for_roster(roster, schema_path="schema.json", rng_seed=None, match_target_overall=True)
    all_rows.extend(roster_with_attrs)

df = pd.DataFrame(all_rows)

# Load schema to get attribute list
schema = load_schema("schema.json")
attrs = schema["attr_order"]

# Compute min and max per Position+Archetype
grouped_min = df.groupby(["Position","Archetype"])[attrs].min().reset_index()
grouped_max = df.groupby(["Position","Archetype"])[attrs].max().reset_index()

# Melt to long form
min_melt = grouped_min.melt(id_vars=["Position","Archetype"], value_vars=attrs,
                            var_name="Attribute", value_name="MinValue")
max_melt = grouped_max.melt(id_vars=["Position","Archetype"], value_vars=attrs,
                            var_name="Attribute", value_name="MaxValue")

# Merge min and max
range_df = pd.merge(min_melt, max_melt, on=["Position","Archetype","Attribute"])
range_df["AttrIndex"] = range_df["Attribute"].apply(lambda a: attrs.index(a))

# Ensure run_data folder exists
os.makedirs("run_data", exist_ok=True)

# Plot ranges per position, color by archetype
for pos, subdf in range_df.groupby("Position"):
    plt.figure(figsize=(18, 8))
    # Draw vertical lines showing min–max range, colored by archetype
    for archetype, arch_df in subdf.groupby("Archetype"):
        plt.scatter(arch_df["AttrIndex"], arch_df["MinValue"], label=f"{archetype} (min)", marker="_")
        plt.scatter(arch_df["AttrIndex"], arch_df["MaxValue"], label=f"{archetype} (max)", marker="_")
        for _, row in arch_df.iterrows():
            plt.plot([row["AttrIndex"], row["AttrIndex"]],
                     [row["MinValue"], row["MaxValue"]],
                     alpha=0.6, label=None)
    plt.xticks(ticks=range(len(attrs)), labels=attrs, rotation=90)
    plt.xlabel("Attribute")
    plt.ylabel("Range of values (min–max)")
    plt.title(f"Attribute ranges — {pos} (split by Archetype)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    filename = f"run_data/{pos}_range_by_archetype.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")
