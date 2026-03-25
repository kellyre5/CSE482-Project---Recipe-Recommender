"""
feature_selection.py
--------------------
Unsupervised feature selection for the Recipe Recommendation System.

Context
-------
The recommender uses TF-IDF + cosine similarity (content-based filtering).
There is no prediction target. The goal of feature selection here is:

    "Which features best DIFFERENTIATE recipes from one another?"

High-discrimination features improve cosine similarity precision.
Low-discrimination features add noise and dilute meaningful distances.

Methods by feature group
------------------------
Numerical  (minutes, n_steps, n_ingredients, 7 nutrition cols)
  - Coefficient of Variation (CV = std/mean): measures relative spread.
    A low CV means nearly every recipe has the same value -> useless for separation.
  - Pairwise Pearson correlation: identifies redundant feature pairs so we
    don't double-weight the same signal (e.g. total_fat ~= saturated_fat).

Text / Binary  (tags, ingredients via TF-IDF)
  - Mean IDF score per feature group: IDF is already a discrimination measure.
    Terms appearing in every recipe have IDF ~= 0 (useless); rare, specific terms
    have high IDF (very discriminative). We surface the top terms from each group.

Combined
  - TruncatedSVD explained variance ratio: shows how much signal each latent
    dimension carries after combining all features

Outputs
-------
  feature_scores.csv     - Numerical feature scores (CV, mean, std, keep flag)
  correlation_matrix.png - Heatmap of numeric feature correlations
  top_tfidf_tags.csv     - Top discriminative tags by IDF
  top_tfidf_ingr.csv     - Top discriminative ingredients by IDF
  idf_tags.png           - Tag IDF bar chart
  idf_ingredients.png    - Ingredient IDF bar chart
  svd_variance.png       - Cumulative explained variance curve
  top_features.json      - Final structured output for the prompt builder
"""

import ast
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# ── 0. Configuration ──────────────────────────────────────────────────────────

RECIPES_PATH      = "../data/RAW_recipes_clean.csv"
INTERACTIONS_PATH = "../data/RAW_interactions.csv"

# Minimum ratings a recipe needs to be included
MIN_RATING_COUNT = 2

# CV threshold: numerical features below this are flagged as low-discrimination
CV_THRESHOLD = 0.10

# Correlation threshold: one of any pair above this is flagged as redundant
CORR_THRESHOLD = 0.80

# How many top tags/ingredients to surface in the JSON output
TOP_N_TAGS = 20
TOP_N_INGR = 20

# Number of SVD components to analyze for explained variance
SVD_COMPONENTS = 60

NUTRITION_COLS = [
    "calories", "total_fat", "sugar",
    "sodium", "protein", "saturated_fat", "carbohydrates",
]

NUMERIC_COLS = ["minutes", "n_steps", "n_ingredients"] + NUTRITION_COLS

# Noise tags to strip before TF-IDF
NOISE_TAGS = {
    "time-to-make", "course", "main-ingredient",
    "preparation", "ingredients", "number-of-servings",
}

# ── 1. Load & filter data ───────────────────────

print("Loading data ...")
recipes      = pd.read_csv(RECIPES_PATH)
interactions = pd.read_csv(INTERACTIONS_PATH).dropna()

rating_counts = (
    interactions.groupby("recipe_id")
    .size()
    .reset_index(name="rating_count")
)

df = recipes.merge(rating_counts, left_on="id", right_on="recipe_id", how="left")
df["rating_count"] = df["rating_count"].fillna(0)
df = df[df["rating_count"] >= MIN_RATING_COUNT].reset_index(drop=True)
df = df.dropna(subset=["name", "ingredients", "tags"]).reset_index(drop=True)

print(f"  Working dataset: {len(df):,} recipes")

# ── 2. Parse string-encoded list columns ─────────────────────────────────────

for col in ["ingredients", "tags", "nutrition", "steps"]:
    if col in df.columns:
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

# ── 3. Unpack nutrition list -> named columns ─────────────────────────────────

nutrition_df = pd.DataFrame(df["nutrition"].tolist(), columns=NUTRITION_COLS, index=df.index)
df = pd.concat([df, nutrition_df], axis=1)

# ── 4. Cap outliers at 99th percentile ───

for col in NUMERIC_COLS:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)


# =============================================================================
# SECTION A: NUMERICAL FEATURE SELECTION
# =============================================================================

print("\n-- Section A: Numerical features --")

numeric_data = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())

# ── A1. Coefficient of Variation ─────────────────────────────────────────────
# CV = std / mean  ->  measures spread *relative* to the feature's own scale.
# This makes minutes and calories comparable without normalizing first.
# A feature with CV < 0.10 barely varies across recipes and hurts cosine
# similarity by adding a near-constant dimension that inflates dot products
# without encoding meaningful differences.

cv_stats = pd.DataFrame({
    "feature": NUMERIC_COLS,
    "mean":    numeric_data.mean().values,
    "std":     numeric_data.std().values,
    "cv":      (numeric_data.std() / numeric_data.mean().replace(0, np.nan)).values,
}).set_index("feature")

cv_stats["low_discrimination"] = cv_stats["cv"] < CV_THRESHOLD
cv_stats = cv_stats.sort_values("cv", ascending=False)

print(f"\n  Coefficient of Variation (threshold = {CV_THRESHOLD}):")
print(cv_stats[["cv", "low_discrimination"]].round(4).to_string())

# ── A2. Pairwise Pearson correlation ─────────────────────────────────────────
# Highly correlated pairs carry redundant signal. Keeping both double-weights
# that axis in cosine similarity without adding new discrimination.

scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(numeric_data), columns=NUMERIC_COLS)
corr   = scaled.corr(method="pearson")

redundant_pairs = []
for i, feat_a in enumerate(NUMERIC_COLS):
    for j, feat_b in enumerate(NUMERIC_COLS):
        if i >= j:
            continue
        r = abs(corr.loc[feat_a, feat_b])
        if r >= CORR_THRESHOLD:
            # Drop the one with lower CV (less discriminative of the pair)
            drop_candidate = (
                feat_a
                if cv_stats.loc[feat_a, "cv"] < cv_stats.loc[feat_b, "cv"]
                else feat_b
            )
            redundant_pairs.append({
                "feature_a":      feat_a,
                "feature_b":      feat_b,
                "pearson_r":      round(r, 4),
                "drop_candidate": drop_candidate,
            })

redundant_df = pd.DataFrame(redundant_pairs)
print(f"\n  Correlated pairs (|r| >= {CORR_THRESHOLD}):")
if len(redundant_df):
    print(redundant_df.to_string(index=False))
else:
    print("  None found.")

# ── A3. Correlation heatmap ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
    vmin=-1, vmax=1
)
ax.set_title(
    f"Numeric Feature Correlation Matrix\n(|r| >= {CORR_THRESHOLD} = redundant pair)",
    fontsize=12
)
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.close()
print("\n  Saved -> correlation_matrix.png")

# ── A4. Summarise numerical selection ────────────────────────────────────────

drop_low_cv    = set(cv_stats[cv_stats["low_discrimination"]].index)
drop_redundant = set(redundant_df["drop_candidate"].tolist()) if len(redundant_df) else set()
drop_numeric   = drop_low_cv | drop_redundant
keep_numeric   = [f for f in NUMERIC_COLS if f not in drop_numeric]

cv_stats["redundant_flag"]   = cv_stats.index.isin(drop_redundant)
cv_stats["recommended_drop"] = cv_stats.index.isin(drop_numeric)
cv_stats.to_csv("feature_scores.csv")

print(f"\n  Recommended numeric features to KEEP ({len(keep_numeric)}): {keep_numeric}")
print(f"  Recommended to DROP  ({len(drop_numeric)}): {sorted(drop_numeric)}")
print("  Full scores saved -> feature_scores.csv")


# =============================================================================
# SECTION B: TEXT FEATURE SELECTION (TF-IDF / IDF Discrimination)
# =============================================================================

print("\n-- Section B: Text features (IDF discrimination) --")


def list_to_string(lst, noise_filter=None):
    """Convert list to token string; optionally filter noise terms."""
    if isinstance(lst, list):
        tokens = [str(t).replace(" ", "_").lower() for t in lst]
        if noise_filter:
            tokens = [t for t in tokens if t.replace("_", "-") not in noise_filter]
        return " ".join(tokens)
    return str(lst).lower()


def fit_tfidf_and_score(corpus, label, max_features=1000):
    """
    Fit TF-IDF, then rank terms by IDF score.

    IDF intuition for feature selection:
      - Low  IDF (near 0): term appears in almost every recipe -> no discrimination
      - High IDF:          term is rare and specific -> highly discriminative

    We want a mix: moderately common (present enough to match users) and
    high-IDF terms (specific enough to separate recipes).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(corpus)
    terms  = vectorizer.get_feature_names_out()
    idfs   = vectorizer.idf_

    scores_df = (
        pd.DataFrame({"term": terms, "idf": idfs})
        .sort_values("idf", ascending=False)
        .reset_index(drop=True)
    )
    scores_df.index += 1
    scores_df.index.name = "idf_rank"

    doc_freq = np.diff(matrix.tocsc().indptr)   # nnz per column = doc freq
    scores_df["doc_freq"]  = doc_freq
    scores_df["doc_freq%"] = (doc_freq / len(corpus) * 100).round(2)

    print(f"\n  [{label}] TF-IDF matrix shape: {matrix.shape}")
    print(f"  Top 10 most discriminative (high IDF):")
    print(scores_df.head(10)[["term", "idf", "doc_freq%"]].to_string())
    print(f"\n  Bottom 10 least discriminative (low IDF -- near-universal terms):")
    print(scores_df.tail(10)[["term", "idf", "doc_freq%"]].to_string())

    return matrix, vectorizer, scores_df


# Tags
tags_corpus = df["tags"].apply(lambda x: list_to_string(x, noise_filter=NOISE_TAGS))
mat_tags, vec_tags, tags_scores = fit_tfidf_and_score(
    tags_corpus, "tags", max_features=300
)
tags_scores.to_csv("top_tfidf_tags.csv")
print("  Full tag scores saved -> top_tfidf_tags.csv")

# Ingredients
ingr_corpus = df["ingredients"].apply(list_to_string)
mat_ingr, vec_ingr, ingr_scores = fit_tfidf_and_score(
    ingr_corpus, "ingredients", max_features=800
)
ingr_scores.to_csv("top_tfidf_ingr.csv")
print("  Full ingredient scores saved -> top_tfidf_ingr.csv")


# ── IDF bar charts ────────────────────────────────────────────────────────────

def plot_idf_bars(scores_df, title, filename, top_n=25):
    top      = scores_df.head(top_n)
    bottom   = scores_df.tail(top_n)
    combined = pd.concat([top, bottom]).drop_duplicates("term")
    n_top    = len(top)
    colors   = ["steelblue"] * n_top + ["salmon"] * (len(combined) - n_top)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(combined["term"][::-1], combined["idf"][::-1], color=colors[::-1])
    ax.axvline(
        scores_df["idf"].median(), color="black", linestyle="--",
        label=f"Median IDF = {scores_df['idf'].median():.2f}"
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("IDF Score  (higher = more discriminative)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved -> {filename}")


plot_idf_bars(
    tags_scores,
    "Tag Term Discrimination (IDF) -- blue=high, red=low",
    "idf_tags.png"
)
plot_idf_bars(
    ingr_scores,
    "Ingredient Term Discrimination (IDF) -- blue=high, red=low",
    "idf_ingredients.png"
)


# =============================================================================
# SECTION C: SVD EXPLAINED VARIANCE (Combined feature space)
# =============================================================================

print("\n-- Section C: SVD explained variance --")

mat_numeric_sparse = csr_matrix(scaler.transform(numeric_data))

combined = hstack([mat_ingr, mat_tags, mat_numeric_sparse])
print(f"  Combined matrix shape: {combined.shape}")

n_components = min(SVD_COMPONENTS, combined.shape[1] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd.fit(combined)

exp_var    = svd.explained_variance_ratio_
cumulative = np.cumsum(exp_var)

thresholds = [0.50, 0.70, 0.80, 0.90]
for t in thresholds:
    n = int(np.searchsorted(cumulative, t)) + 1
    print(f"  Components to explain {int(t*100)}% variance: {n}")

# ── SVD plot ──────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.bar(range(1, len(exp_var) + 1), exp_var, color="steelblue", alpha=0.8)
ax1.set_title("Per-Component Explained Variance")
ax1.set_xlabel("SVD Component")
ax1.set_ylabel("Explained Variance Ratio")

ax2.plot(range(1, len(cumulative) + 1), cumulative, color="coral", linewidth=2)
for t in thresholds:
    n = int(np.searchsorted(cumulative, t)) + 1
    ax2.axhline(t, linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.annotate(
        f"{int(t*100)}% @ {n}", xy=(n, t), fontsize=8,
        xytext=(n + 1, t - 0.03)
    )
ax2.set_title("Cumulative Explained Variance")
ax2.set_xlabel("Number of SVD Components")
ax2.set_ylabel("Cumulative Explained Variance")
ax2.set_ylim(0, 1.05)

plt.suptitle("SVD Analysis of Combined Feature Space", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("svd_variance.png", dpi=150)
plt.close()
print("  Saved -> svd_variance.png")


# =============================================================================
# SECTION D: Export JSON for recommender prompt builder
# =============================================================================

print("\n-- Section D: Exporting top_features.json --")

# Numerical: rank by CV descending, exclude dropped features
numeric_out = []
for feat in keep_numeric:
    row = cv_stats.loc[feat]
    numeric_out.append({
        "rank":          None,
        "feature":       feat,
        "feature_type":  "numerical",
        "cv_score":      round(float(row["cv"]), 6),
        "prompt_key":    feat.replace("_", " ").title(),
        "input_type":    "range",
        "source_column": feat,
    })
numeric_out = sorted(numeric_out, key=lambda x: -x["cv_score"])

# Tags: top N by IDF
tag_out = []
for _, row in tags_scores.head(TOP_N_TAGS).iterrows():
    tag_out.append({
        "rank":          None,
        "feature":       f"tag__{row['term']}",
        "feature_type":  "tag",
        "idf_score":     round(float(row["idf"]), 6),
        "doc_freq_pct":  float(row["doc_freq%"]),
        "prompt_key":    f"Tag: {row['term'].replace('_', ' ').title()}",
        "input_type":    "boolean",
        "source_column": "tags",
        "tag_value":     row["term"],
    })

# Ingredients: top N by IDF
ingr_out = []
for _, row in ingr_scores.head(TOP_N_INGR).iterrows():
    ingr_out.append({
        "rank":             None,
        "feature":          f"ingr__{row['term']}",
        "feature_type":     "ingredient",
        "idf_score":        round(float(row["idf"]), 6),
        "doc_freq_pct":     float(row["doc_freq%"]),
        "prompt_key":       f"Contains: {row['term'].replace('_', ' ').title()}",
        "input_type":       "boolean",
        "source_column":    "ingredients",
        "ingredient_value": row["term"],
    })

all_out = numeric_out + tag_out + ingr_out
for i, item in enumerate(all_out, start=1):
    item["rank"] = i

with open("top_features.json", "w") as f:
    json.dump(all_out, f, indent=2)

print(f"  {len(all_out)} features exported -> top_features.json")

# ── Final summary ─────────────────────────────────────────────────────────────

print(f"""
{'='*60}
 FEATURE SELECTION SUMMARY
{'='*60}
 Numerical kept  : {len(keep_numeric)} / {len(NUMERIC_COLS)}
   {keep_numeric}
 Numerical dropped: {sorted(drop_numeric) or 'none'}
   Reason: low CV (< {CV_THRESHOLD}) or correlated (|r| >= {CORR_THRESHOLD})

 Top tag features    : {TOP_N_TAGS}  (ranked by IDF)
 Top ingredient feat.: {TOP_N_INGR}  (ranked by IDF)

 Outputs
   feature_scores.csv     numeric CV + flags
   correlation_matrix.png numeric pairwise correlation heatmap
   top_tfidf_tags.csv     all tag IDF scores
   top_tfidf_ingr.csv     all ingredient IDF scores
   idf_tags.png           tag IDF bar chart
   idf_ingredients.png    ingredient IDF bar chart
   svd_variance.png       cumulative variance curve
   top_features.json      structured output for prompt builder
{'='*60}
""")
