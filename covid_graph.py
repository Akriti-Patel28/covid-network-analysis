import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import os


# =====================================================
# 1. LOAD REAL COVID-19 DATASET
# =====================================================

print("Loading real COVID-19 India dataset...")

df = pd.read_csv("complete.csv")

print("\nDataset loaded successfully!")


# =====================================================
# 2. DISPLAY ORIGINAL DATASET INFORMATION
# =====================================================

print("\nOriginal Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# =====================================================
# 3. RENAME COLUMNS
# =====================================================

df = df.rename(columns={

    "Name of State / UT": "State",

    "Total Confirmed cases": "Confirmed",

    "Cured/Discharged/Migrated": "Recovered",

    "Death": "Deceased",

    "New cases": "New_Cases",

    "New deaths": "New_Deaths",

    "New recovered": "New_Recovered"

})


# =====================================================
# 4. CHECK REQUIRED COLUMNS
# =====================================================

required_columns = [
    "Date",
    "State",
    "Confirmed",
    "Recovered",
    "Deceased"
]

for column in required_columns:

    if column not in df.columns:

        print(
            f"\nERROR: Required column '{column}' not found."
        )

        print("\nAvailable columns are:")

        print(df.columns.tolist())

        exit()


# =====================================================
# 5. DATA CLEANING
# =====================================================

print("\nCleaning dataset...")

# Convert date into proper datetime format

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)


# Remove rows where date could not be converted

df = df.dropna(
    subset=["Date"]
)


# Sort data by date

df = df.sort_values(
    "Date"
)


# Convert numerical columns to numbers

numeric_columns = [
    "Confirmed",
    "Recovered",
    "Deceased"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].fillna(0)


# =====================================================
# 6. CALCULATE ACTIVE CASES
# =====================================================

df["Active"] = (
    df["Confirmed"]
    - df["Recovered"]
    - df["Deceased"]
)


# Active cases cannot be negative

df["Active"] = df["Active"].clip(
    lower=0
)


# =====================================================
# 7. CALCULATE NEW CASES
# =====================================================

if "New_Cases" not in df.columns:

    df["New_Cases"] = (
        df.groupby("State")["Confirmed"]
        .diff()
    )

else:

    df["New_Cases"] = pd.to_numeric(
        df["New_Cases"],
        errors="coerce"
    )


df["New_Cases"] = df["New_Cases"].fillna(0)

df["New_Cases"] = df["New_Cases"].clip(
    lower=0
)


# =====================================================
# 8. DATASET INFORMATION
# =====================================================

print("\n======================================")
print("DATASET INFORMATION")
print("======================================")

print(
    "\nDate Range:"
)

print(
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)


print(
    "\nNumber of States/UTs:",
    df["State"].nunique()
)


print(
    "\nTotal Records:",
    len(df)
)


print(
    "\nFinal Columns:"
)

print(
    df.columns.tolist()
)


# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# =====================================================
# GRAPH 1
# COVID CASES OVER TIME
# =====================================================

print("\nGenerating Graph 1...")

selected_states = [
    "Maharashtra",
    "Kerala",
    "Karnataka",
    "Tamil Nadu",
    "Delhi"
]


plt.figure(
    figsize=(14, 7)
)


for state in selected_states:

    state_data = df[
        df["State"] == state
    ]

    if len(state_data) == 0:

        continue

    plt.plot(
        state_data["Date"],
        state_data["Confirmed"],
        label=state,
        linewidth=2
    )


plt.xlabel(
    "Date"
)

plt.ylabel(
    "Confirmed Cases"
)

plt.title(
    "COVID-19 Confirmed Cases Over Time",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "outputs/01_cases_over_time.png",
    dpi=300
)

plt.show()


# =====================================================
# GRAPH 2
# ACTIVE CASES OVER TIME
# =====================================================

print("\nGenerating Graph 2...")


plt.figure(
    figsize=(14, 7)
)


for state in selected_states:

    state_data = df[
        df["State"] == state
    ]

    if len(state_data) == 0:

        continue

    plt.plot(
        state_data["Date"],
        state_data["Active"],
        label=state,
        linewidth=2
    )


plt.xlabel(
    "Date"
)

plt.ylabel(
    "Active Cases"
)

plt.title(
    "Active COVID-19 Cases Over Time",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "outputs/02_active_cases.png",
    dpi=300
)

plt.show()


# =====================================================
# GET LATEST DATE
# =====================================================

latest_date = df["Date"].max()

latest = df[
    df["Date"] == latest_date
].copy()


# =====================================================
# GRAPH 3
# STATE-WISE COVID CASES
# =====================================================

print("\nGenerating Graph 3...")


latest = latest.sort_values(
    "Confirmed",
    ascending=False
)


top_states = latest.head(15)


plt.figure(
    figsize=(12, 7)
)


plt.barh(
    top_states["State"][::-1],
    top_states["Confirmed"][::-1]
)


plt.xlabel(
    "Confirmed Cases"
)

plt.ylabel(
    "State"
)

plt.title(
    f"State-wise COVID-19 Cases ({latest_date.date()})",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "outputs/03_state_cases.png",
    dpi=300
)

plt.show()


# =====================================================
# GRAPH 4
# COVID ACTIVE CASE HEATMAP
# =====================================================

print("\nGenerating Graph 4...")


heatmap_data = df[
    df["State"].isin(selected_states)
].pivot_table(
    index="State",
    columns="Date",
    values="Active",
    aggfunc="sum"
)


heatmap_data = heatmap_data.fillna(0)


plt.figure(
    figsize=(16, 7)
)


sns.heatmap(
    heatmap_data,
    cmap="Reds"
)


plt.title(
    "COVID-19 Active Cases Heatmap",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "State"
)

plt.tight_layout()

plt.savefig(
    "outputs/04_heatmap.png",
    dpi=300
)

plt.show()


# =====================================================
# 9. CREATE STATE TIME-SERIES DATA
# =====================================================

print("\nPreparing data for graph network...")


network_states = (
    latest
    .sort_values(
        "Confirmed",
        ascending=False
    )
    .head(15)["State"]
    .tolist()
)


network_data = df[
    df["State"].isin(network_states)
].pivot_table(
    index="State",
    columns="Date",
    values="New_Cases",
    aggfunc="sum"
)


network_data = network_data.fillna(0)


# =====================================================
# 10. SCALE THE DATA
# =====================================================

scaler = StandardScaler()


X = scaler.fit_transform(
    network_data
)


# =====================================================
# 11. CALCULATE COSINE SIMILARITY
# =====================================================

similarity = cosine_similarity(
    X
)


similarity_df = pd.DataFrame(
    similarity,
    index=network_data.index,
    columns=network_data.index
)


# =====================================================
# 12. CREATE COVID SIMILARITY GRAPH
# =====================================================

print("\nCreating COVID similarity network...")


G = nx.Graph()


# Add states as nodes

for state in network_states:

    G.add_node(
        state
    )


# Similarity threshold

threshold = 0.85


# Add edges based on similarity

for i in range(
    len(network_states)
):

    for j in range(
        i + 1,
        len(network_states)
    ):

        state1 = network_states[i]

        state2 = network_states[j]


        score = similarity_df.loc[
            state1,
            state2
        ]


        if score >= threshold:

            G.add_edge(
                state1,
                state2,
                weight=score
            )


print(
    "\nNumber of Nodes:",
    G.number_of_nodes()
)


print(
    "Number of Edges:",
    G.number_of_edges()
)


# =====================================================
# GRAPH 5
# COVID SIMILARITY NETWORK
# =====================================================

print("\nGenerating Graph 5...")


plt.figure(
    figsize=(14, 10)
)


pos = nx.spring_layout(
    G,
    seed=42,
    weight="weight"
)


degrees = dict(
    G.degree()
)


node_sizes = [
    700 + degrees[node] * 500
    for node in G.nodes
]


node_colors = [
    degrees[node]
    for node in G.nodes
]


# Draw edges

nx.draw_networkx_edges(
    G,
    pos,
    width=1.5,
    alpha=0.5
)


# Draw nodes

nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap="Reds"
)


# Draw labels

nx.draw_networkx_labels(
    G,
    pos,
    font_size=9,
    font_weight="bold"
)


plt.colorbar(
    nodes,
    label="Number of Connections"
)


plt.title(
    "COVID-19 State Similarity Network",
    fontsize=18,
    fontweight="bold"
)


plt.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    "outputs/05_covid_network.png",
    dpi=300
)


plt.show()


# =====================================================
# 13. CENTRALITY ANALYSIS
# =====================================================

print("\nCalculating network centrality...")


# Degree centrality

degree_centrality = nx.degree_centrality(
    G
)


# Betweenness centrality

betweenness_centrality = nx.betweenness_centrality(
    G
)


# Closeness centrality

closeness_centrality = nx.closeness_centrality(
    G
)


# Create centrality dataframe

centrality_df = pd.DataFrame({

    "State":
        list(G.nodes),

    "Degree_Centrality":
        [
            degree_centrality[state]
            for state in G.nodes
        ],

    "Betweenness_Centrality":
        [
            betweenness_centrality[state]
            for state in G.nodes
        ],

    "Closeness_Centrality":
        [
            closeness_centrality[state]
            for state in G.nodes
        ]

})


# =====================================================
# GRAPH 6
# DEGREE CENTRALITY
# =====================================================

print("\nGenerating Graph 6...")


centrality_df = centrality_df.sort_values(
    "Degree_Centrality",
    ascending=False
)


plt.figure(
    figsize=(12, 7)
)


plt.bar(
    centrality_df["State"],
    centrality_df["Degree_Centrality"]
)


plt.xticks(
    rotation=45,
    ha="right"
)


plt.xlabel(
    "State"
)

plt.ylabel(
    "Degree Centrality"
)


plt.title(
    "COVID Network Degree Centrality",
    fontsize=16,
    fontweight="bold"
)


plt.tight_layout()


plt.savefig(
    "outputs/06_centrality.png",
    dpi=300
)


plt.show()


# =====================================================
# GRAPH 7
# CENTRALITY VS COVID BURDEN
# =====================================================

print("\nGenerating Graph 7...")


comparison = centrality_df.copy()


comparison["Confirmed"] = (
    comparison["State"]
    .map(
        latest.set_index(
            "State"
        )["Confirmed"]
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.scatter(
    comparison["Degree_Centrality"],
    comparison["Confirmed"],
    s=120
)


# Add state names

for _, row in comparison.iterrows():

    plt.annotate(
        row["State"],
        (
            row["Degree_Centrality"],
            row["Confirmed"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )


plt.xlabel(
    "Degree Centrality"
)

plt.ylabel(
    "Confirmed COVID-19 Cases"
)


plt.title(
    "Network Centrality vs COVID-19 Burden",
    fontsize=16,
    fontweight="bold"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    "outputs/07_centrality_vs_cases.png",
    dpi=300
)


plt.show()


# =====================================================
# 14. DISPLAY CENTRALITY TABLE
# =====================================================

print("\n======================================")
print("NETWORK CENTRALITY")
print("======================================")


print(
    centrality_df.to_string(
        index=False
    )
)


# =====================================================
# 15. FINAL PROJECT SUMMARY
# =====================================================

print("\n======================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("======================================")


print(
    "\nDataset Date Range:"
)

print(
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)


print(
    "\nLatest Date:",
    latest_date.date()
)


print(
    "\nTotal States/UTs:",
    df["State"].nunique()
)


print(
    "\nTotal Records:",
    len(df)
)


print(
    "\nNetwork Nodes:",
    G.number_of_nodes()
)


print(
    "\nNetwork Edges:",
    G.number_of_edges()
)


print(
    "\nGraphs Generated: 7"
)


print(
    "\nAll graphs are saved inside the 'outputs' folder."
)


print("\n")
print("DONE!")
