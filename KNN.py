
"""
Alexa: Recipe Recommender: TF-IDF + KNN with Euclidean Distance
========================================================
Pipeline:
  0. Load all datasets
  1. Clean & prepare data
  2. TF-IDF on text fields (ingredients, tags, name, description)
  3. Scale numeric fields (minutes, n_steps, n_ingredients, avg_rating)
  4. Apply feature weights & combine into feature matrix
  5. KNN with Euclidean distance for recommendations
  6. Evaluation + visualizations
"""


import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Import additional libraries needed for KNN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


#Load datasets

recipes_pp_df = pd.read_csv('PP_recipes.csv', on_bad_lines='skip', escapechar='\\')
interactions_raw_df = pd.read_csv('RAW_interactions.csv').dropna()
recipes_raw_df = pd.read_csv('RAW_recipes_clean.csv')
users_pp_df = pd.read_csv('PP_users.csv')



print("Total interactions:", len(interactions_raw_df))
print("Missing ratings:",    interactions_raw_df['rating'].isna().sum())
print("Missing reviews:",    interactions_raw_df['review'].isna().sum())

# Keep only rows with a valid rating
interactions_raw_df = interactions_raw_df[
    interactions_raw_df['rating'].notna()
]

# Count ratings per recipe 
rating_counts    = interactions_raw_df.groupby('recipe_id').size()
rating_counts_df = rating_counts.reset_index()
rating_counts_df.columns = ['recipe_id', 'rating_count']

print("\nRating Count Statistics:")
print(rating_counts_df['rating_count'].describe())

# Examine threshold effects
print("\nRecipes below various thresholds:")
for threshold in [1, 2, 3, 5, 10]:
    count   = (rating_counts_df['rating_count'] < threshold).sum()
    percent = count / len(rating_counts_df) * 100
    print(f"  Recipes with < {threshold} ratings: {count} ({percent:.2f}%)")

# Apply final threshold (>= 2 ratings)
filtered_recipes = recipes_raw_df.merge(
    rating_counts_df,
    left_on='id',
    right_on='recipe_id',
    how='left'
)
filtered_recipes['rating_count'] = filtered_recipes['rating_count'].fillna(0)
filtered_recipes = filtered_recipes[filtered_recipes['rating_count'] >= 2]

print("\nOriginal number of recipes:", len(recipes_raw_df))
print("Number after filtering (>= 2 ratings):", len(filtered_recipes))

# Calculate average ratings
# Get average rating for each recipe
recipe_ratings = interactions_raw_df.groupby('recipe_id')['rating'].mean().reset_index()
recipe_ratings = recipe_ratings.rename(columns={'rating': 'avg_rating'})

# Add average ratings to filtered recipes
filtered_recipes = filtered_recipes.merge(
    recipe_ratings,
    left_on='id',
    right_on='recipe_id',
    how='left'
)
filtered_recipes['avg_rating'] = filtered_recipes['avg_rating'].fillna(0)



# FEATURE EXTRACTION 

# Extract text features with TF-IDF
tfidf_config = {
    'max_features': 1000,
    'min_df': 2,
    'max_df': 0.8,
}

# Save the fitted vectorizers (needed for recommend_from_user_input later)
ingredients_vectorizer = TfidfVectorizer(**tfidf_config)
ingredients_tfidf = ingredients_vectorizer.fit_transform(filtered_recipes['ingredients'].fillna(''))

tags_vectorizer = TfidfVectorizer(**tfidf_config)
tags_tfidf = tags_vectorizer.fit_transform(filtered_recipes['tags'].fillna(''))

name_vectorizer = TfidfVectorizer(**tfidf_config)
name_tfidf = name_vectorizer.fit_transform(filtered_recipes['name'].fillna(''))

description_vectorizer = TfidfVectorizer(**tfidf_config)
description_tfidf = description_vectorizer.fit_transform(filtered_recipes['description'].fillna(''))

# Extract and scale numeric features
numeric_features = filtered_recipes[['minutes', 'n_steps', 'n_ingredients', 'avg_rating']].copy()

# Log-transform time (often skewed)
numeric_features['minutes'] = np.log1p(numeric_features['minutes'])

# Standardize numeric features 
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)


# FEATURE COMBINATION 
from scipy.sparse import hstack as sparse_hstack, csr_matrix

# Feature weights
feature_weights = {
    'ingredients': 2.0,
    'tags': 1.5,
    'name': 1.0,
    'description': 0.5,
    'numeric': 1.5
}

# Multiply sparse matrices by weights directly — no .toarray() needed
weighted_ingredients = ingredients_tfidf * feature_weights['ingredients']
weighted_tags = tags_tfidf * feature_weights['tags']
weighted_name = name_tfidf * feature_weights['name']
weighted_description = description_tfidf * feature_weights['description']

# Convert the small numeric array to sparse too
weighted_numeric = csr_matrix(numeric_scaled * feature_weights['numeric'])

# Combine using sparse hstack
features_combined = sparse_hstack([
    weighted_ingredients,
    weighted_tags,
    weighted_name,
    weighted_description,
    weighted_numeric
]).tocsr()

print(f"Combined feature matrix shape: {features_combined.shape}")

# Free the intermediate weighted matrices
del weighted_ingredients, weighted_tags, weighted_name, weighted_description, weighted_numeric
gc.collect()


# FULL DATASET KNN IMPLEMENTATION

# Create recipe ID to index mappings
recipe_id_to_idx = dict(zip(filtered_recipes['id'], range(len(filtered_recipes))))
idx_to_recipe_id = dict(zip(range(len(filtered_recipes)), filtered_recipes['id']))

# Instead of finding optimal K through cross-validation, use a reasonable value
# Research shows that for recommendation systems, K values between 5-20 work well
k = 10

# BUILD THE KNN MODEL DIRECTLY

print(f"Building KNN model with k={k} using full dataset of {len(filtered_recipes)} recipes...")
print(f"Feature matrix shape: {features_combined.shape}")

# Build the KNN model with fixed K
knn_model = NearestNeighbors(
    n_neighbors=k+1,  # +1 because the first match is the recipe itself
    metric='euclidean',   # Best for high-dimensional sparse data
    algorithm='auto',  # Let scikit-learn choose the best algorithm
    n_jobs=-1         # Use all CPU cores for parallel processing
)

# Fit the model to the full dataset
knn_model.fit(features_combined)
print("KNN model successfully built!")

# CREATE AN EFFICIENT RECOMMENDATION FUNCTION
#

def get_recommendations(recipe_id, n=5):
    """Get recipe recommendations using KNN with the full dataset"""

    # Find recipe index
    if recipe_id not in recipe_id_to_idx:
        print(f"Recipe ID {recipe_id} not found")
        return pd.DataFrame()

    recipe_idx = recipe_id_to_idx[recipe_id]

    # Get recipe vector and find nearest neighbors
    recipe_vector = features_combined[recipe_idx].reshape(1, -1)
    distances, indices = knn_model.kneighbors(recipe_vector)

    # Skip the first result (which is the recipe itself)
    rec_indices = indices.flatten()[1:n+1]

    # Get recipe IDs
    recommended_ids = [idx_to_recipe_id[idx] for idx in rec_indices]

    # Get recipe details
    recommendations = filtered_recipes[filtered_recipes['id'].isin(recommended_ids)].copy()

    # Calculate similarity scores (1 - distance)
    similarities = 1 / (1 + distances.flatten()[1:n+1])
    similarity_dict = dict(zip(recommended_ids, similarities))
    recommendations['similarity'] = recommendations['id'].map(similarity_dict)

    # Sort by similarity
    recommendations = recommendations.sort_values('similarity', ascending=False)

    # Add shared ingredients information
    seed_recipe = filtered_recipes[filtered_recipes['id'] == recipe_id].iloc[0]

    # Parse ingredients from the string representation (list stored as string)
    def parse_ingredients(ing_str):
        if not isinstance(ing_str, str):
            return set()
        try:
            return set(ast.literal_eval(ing_str))
        except (ValueError, SyntaxError):
            # Fallback: split on commas and clean up
            return set(i.strip().strip("'\"[] ") for i in ing_str.split(','))

    seed_ingredients = parse_ingredients(seed_recipe['ingredients'])

    def count_shared_ingredients(rec_ingredients):
        rec_set = parse_ingredients(rec_ingredients)
        return len(seed_ingredients.intersection(rec_set))

    recommendations['shared_ingredients'] = recommendations['ingredients'].apply(count_shared_ingredients)

    # Add simple insight about main reason for recommendation
    def get_insight(row):
        if row['shared_ingredients'] >= 3:
            return f"Shares {row['shared_ingredients']} ingredients"
        elif abs(row['minutes'] - seed_recipe['minutes']) < 10:
            return "Similar cooking time"
        else:
            return "Similar overall profile"

    recommendations['insight'] = recommendations.apply(get_insight, axis=1)

    return recommendations



# STEP 4: TEST RECOMMENDATION FOR A SINGLE RECIPE

# Pick a random recipe (change random seed to change recipe)
test_recipe_id = filtered_recipes.sample(1, random_state=42)['id'].values[0]

# Get seed recipe info
seed = filtered_recipes[filtered_recipes['id'] == test_recipe_id].iloc[0]

print("=" * 60)
print("SEED RECIPE")
print("=" * 60)
print(f"  Name:        {seed['name']}")
print(f"  ID:          {seed['id']}")
print(f"  Minutes:     {seed['minutes']}")
print(f"  Steps:       {seed['n_steps']}")
print(f"  Ingredients: {seed['n_ingredients']}")
print(f"  Avg Rating:  {seed['avg_rating']:.2f}")

# Get recommendations
recs = get_recommendations(test_recipe_id, n=5)

if len(recs) > 0:
    print(f"\n{'=' * 60}")
    print(f"TOP {len(recs)} RECOMMENDATIONS")
    print(f"{'=' * 60}")

    similarity_scores = []
    shared_counts = []

    for j, (_, rec) in enumerate(recs.iterrows(), 1):
        similarity_scores.append(rec['similarity'])
        shared_counts.append(rec['shared_ingredients'])

        print(f"\n  {j}. {rec['name']}")
        print(f"     Similarity:         {rec['similarity']:.4f}")
        print(f"     Shared Ingredients: {rec['shared_ingredients']}")
        print(f"     Minutes: {rec['minutes']} | Steps: {rec['n_steps']} | "
              f"Avg Rating: {rec['avg_rating']:.2f}")
        print(f"     Insight: {rec['insight']}")

    print(f"\n{'─' * 60}")
    print(f"  Avg Similarity:         {np.mean(similarity_scores):.4f}")
    print(f"  Avg Shared Ingredients: {np.mean(shared_counts):.1f}")
else:
    print("\n  No recommendations found.")



def recommend_from_user_input(
    user_ingredients,
    user_tags,
    user_numeric,
    top_k=5
):
    """
    Build a query vector from raw user input and return top_k recommendations
    using the fitted KNN model.

    Args:
        user_ingredients : list, e.g. ['chicken', 'garlic', 'olive oil']
        user_tags        : list, e.g. ['dinner', 'quick', 'low-fat']
        user_numeric     : dict, e.g. {'minutes': 30, 'n_steps': 6, 'n_ingredients': 8}
        top_k            : int, number of recommendations to return

    Returns:
        DataFrame with top_k recipes and similarity scores.
    """
    # Transform text features using fitted vectorizers
    ing_str = " ".join(user_ingredients)
    tag_str = " ".join(user_tags)

    ing_vec  = ingredients_vectorizer.transform([ing_str]).toarray() * feature_weights['ingredients']
    tag_vec  = tags_vectorizer.transform([tag_str]).toarray() * feature_weights['tags']

    # Pass empty strings for name and description to keep vector shape consistent
    name_vec = name_vectorizer.transform([""]).toarray() * feature_weights['name']
    desc_vec = description_vectorizer.transform([""]).toarray() * feature_weights['description']

    # Transform numeric features using fitted scaler
    default_avg_rating = filtered_recipes['avg_rating'].median()

    num_values = pd.DataFrame([{
        'minutes': user_numeric.get('minutes', 0),
        'n_steps': user_numeric.get('n_steps', 0),
        'n_ingredients': user_numeric.get('n_ingredients', 0),
        'avg_rating': user_numeric.get('avg_rating', default_avg_rating)
    }])

    # Apply same log-transform on minutes as training data
    num_values['minutes'] = np.log1p(num_values['minutes'])

    num_scaled = scaler.transform(num_values) * feature_weights['numeric']

    # Combine into single query vector (same order as features_combined)
    query_vec = np.hstack([ing_vec, tag_vec, name_vec, desc_vec, num_scaled])

    # Use KNN model to find nearest neighbors
    distances, indices = knn_model.kneighbors(query_vec, n_neighbors=top_k)

    rec_indices = indices.flatten()
    rec_distances = distances.flatten()

    # Convert distances to similarity scores
    similarities = 1 / (1 + rec_distances)

    # Map indices back to recipe IDs
    recommended_ids = [idx_to_recipe_id[idx] for idx in rec_indices]

    results = filtered_recipes[filtered_recipes['id'].isin(recommended_ids)].copy()

    # Add similarity scores
    similarity_dict = dict(zip(recommended_ids, similarities))
    results['similarity_score'] = results['id'].map(similarity_dict)

    # Sort by similarity
    results = results.sort_values('similarity_score', ascending=False)

    # Return clean output
    results = results[['name', 'id', 'minutes', 'n_steps', 'n_ingredients', 'avg_rating', 'similarity_score']].copy()
    results['similarity_score'] = results['similarity_score'].round(4)
    results.index = range(1, len(results) + 1)

    return results


# Quick test
print("Testing recommend_from_user_input...")
test_results = recommend_from_user_input(
    user_ingredients=["chicken", "garlic", "olive oil", "lemon"],
    user_tags=["dinner", "quick"],
    user_numeric={"minutes": 30, "n_steps": 6, "n_ingredients": 8},
    top_k=5
)
print(test_results.to_string())


#  PRECISION@K EVALUATION

import ast

def parse_ingredients(ing_str):
    """Parse ingredients from string representation to a set."""
    if not isinstance(ing_str, str):
        return set()
    try:
        return set(ast.literal_eval(ing_str))
    except (ValueError, SyntaxError):
        return set(i.strip().strip("'\"[] ") for i in ing_str.split(','))


def precision_at_k(query_indices, k=3, min_shared_ingredients=2):
    """
    Proxy Precision@K using KNN: a recommendation is 'relevant' if it
    shares at least min_shared_ingredients with the query recipe.
    """
    results_log = []

    for idx in query_indices:
        # Get query recipe ingredients
        query_id = idx_to_recipe_id[idx]
        query_row = filtered_recipes[filtered_recipes['id'] == query_id].iloc[0]
        query_ings = parse_ingredients(query_row['ingredients'])

        # Use KNN to get neighbors
        query_vec = features_combined[idx].reshape(1, -1)
        distances, indices = knn_model.kneighbors(query_vec, n_neighbors=k + 1)

        # Skip first result (the recipe itself)
        rec_indices = indices.flatten()[1:k + 1]

        hits = 0
        for rec_idx in rec_indices:
            rec_id = idx_to_recipe_id[rec_idx]
            rec_row = filtered_recipes[filtered_recipes['id'] == rec_id].iloc[0]
            rec_ings = parse_ingredients(rec_row['ingredients'])
            shared = len(query_ings & rec_ings)
            if shared >= min_shared_ingredients:
                hits += 1

        precision = hits / k
        results_log.append({
            "query": query_row['name'],
            f"precision@{k}": precision,
            "hits": hits
        })

    results_df = pd.DataFrame(results_log)
    mean_p = results_df[f"precision@{k}"].mean()
    print(f"\n{'='*50}")
    print(f"PRECISION@{k} (min {min_shared_ingredients} shared ingredients)")
    print(f"{'='*50}")
    display(results_df)
    print(f"\nMean Precision@{k}: {mean_p:.3f}")
    return results_df


# Evaluate on a sample
sample_indices = list(range(0, min(50, len(filtered_recipes)), 5))
prec_df = precision_at_k(sample_indices, k=3, min_shared_ingredients=2)

#PRECISION@K BAR CHART

def plot_precision_at_k(prec_df, k=3):
    """Bar chart of per-query Precision@K scores."""
    plt.figure(figsize=(12, 4))
    colors = ["steelblue" if v >= 0.5 else "salmon" for v in prec_df[f"precision@{k}"]]
    plt.bar(range(len(prec_df)), prec_df[f"precision@{k}"], color=colors)
    plt.axhline(prec_df[f"precision@{k}"].mean(), color="black", linestyle="--",
                label=f"Mean = {prec_df[f'precision@{k}'].mean():.2f}")
    plt.xticks(range(len(prec_df)),
               [n[:20] for n in prec_df["query"]], rotation=45, ha="right", fontsize=8)
    plt.ylabel(f"Precision@{k}")
    plt.title(f"KNN Per-Query Precision@{k} (proxy: ≥2 shared ingredients)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("knn_precision_at_k.png", dpi=150)
    plt.show()
    print("Saved: knn_precision_at_k.png")

plot_precision_at_k(prec_df, k=3)


# KNN DISTANCE DISTRIBUTION

def plot_knn_distance_distribution(sample_size=300):
    """
    Histogram of KNN distances for a random sample of recipes.
    Lower distances = more similar. A tight distribution near 0 means
    the model finds strong matches consistently.
    """
    np.random.seed(42)
    n_recipes = features_combined.shape[0]
    idx = np.random.choice(n_recipes, size=min(sample_size, n_recipes), replace=False)

    all_distances = []
    for i in idx:
        query_vec = features_combined[i].reshape(1, -1)
        distances, _ = knn_model.kneighbors(query_vec, n_neighbors=6)
        # Skip distance to self (0.0), take the 5 neighbor distances
        all_distances.extend(distances.flatten()[1:])

    plt.figure(figsize=(8, 5))
    sns.histplot(all_distances, bins=60, kde=True, color="steelblue")
    plt.title("Distribution of KNN Neighbor Distances")
    plt.xlabel("Distance (lower = more similar)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("knn_distance_distribution.png", dpi=150)
    plt.show()
    print("Saved: knn_distance_distribution.png")

    # Also show as similarity (1 - distance)
    similarities = [1 - d for d in all_distances]
    plt.figure(figsize=(8, 5))
    sns.histplot(similarities, bins=60, kde=True, color="coral")
    plt.title("Distribution of KNN Similarity Scores (1 - distance)")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("knn_similarity_distribution.png", dpi=150)
    plt.show()
    print("Saved: knn_similarity_distribution.png")

plot_knn_distance_distribution()


#  TOP TF-IDF TERMS

def plot_top_tfidf_terms(vectorizer, title="Top TF-IDF Terms", top_n=20, filename="tfidf_terms.png"):
    """Bar chart of the most common terms by inverse IDF score."""
    idf_scores = vectorizer.idf_
    terms = vectorizer.get_feature_names_out()
    importance = 1 / idf_scores
    top_idx = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10, 5))
    plt.barh(
        [terms[i] for i in top_idx][::-1],
        importance[top_idx][::-1],
        color="coral"
    )
    plt.title(title)
    plt.xlabel("Relative Frequency (1 / IDF)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")

plot_top_tfidf_terms(ingredients_vectorizer, title="Top Ingredient Terms (TF-IDF)", filename="top_ingredients.png")
plot_top_tfidf_terms(tags_vectorizer, title="Top Tag Terms (TF-IDF)", filename="top_tags.png")

#  2D RECIPE SPACE VIA t-SNE

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

def plot_2d_recipe_space(sample_size=500):
    """
    Reduce features_combined to 2D with SVD → t-SNE and scatter-plot.
    Highlights KNN neighbors for a few example recipes in color.
    """
    np.random.seed(42)
    n_recipes = features_combined.shape[0]
    idx = np.random.choice(n_recipes, size=min(sample_size, n_recipes), replace=False)
    sample = features_combined[idx]

    # Step 1: reduce to 50 dims (t-SNE can't handle high-dim sparse well)
    svd = TruncatedSVD(n_components=50, random_state=42)
    dense = svd.fit_transform(sample)

    # Step 2: t-SNE to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(dense)

    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.3, s=10, color="grey", label="All recipes")

    # Highlight KNN neighbors for 3 example recipes from the sample
    colors = ["red", "blue", "green"]
    for i, color in enumerate(colors):
        example_idx = idx[i * 10]  # pick spaced-out examples
        query_vec = features_combined[example_idx].reshape(1, -1)
        _, neighbor_indices = knn_model.kneighbors(query_vec, n_neighbors=6)
        neighbor_indices = neighbor_indices.flatten()[1:]  # skip self

        # Find which neighbors are in our sample
        sample_positions = []
        for ni in neighbor_indices:
            positions = np.where(idx == ni)[0]
            if len(positions) > 0:
                sample_positions.append(positions[0])

        # Plot the query recipe
        query_pos = np.where(idx == example_idx)[0][0]
        recipe_name = filtered_recipes.iloc[example_idx]['name'][:25]
        plt.scatter(coords[query_pos, 0], coords[query_pos, 1],
                    color=color, s=100, marker="*", zorder=5,
                    label=f"Query: {recipe_name}")

        # Plot its neighbors
        if sample_positions:
            plt.scatter(coords[sample_positions, 0], coords[sample_positions, 1],
                        color=color, s=40, marker="o", alpha=0.8, zorder=4)

    plt.title("Recipe Feature Space (t-SNE) with KNN Neighbors Highlighted")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=1.5, fontsize=8)
    plt.tight_layout()
    plt.savefig("knn_recipe_2d_space.png", dpi=150)
    plt.show()
    print("Saved: knn_recipe_2d_space.png")

plot_2d_recipe_space()


# ─────────────────────────────────────────────
# FOR REBECCA'S UI
# ─────────────────────────────────────────────

# FUNCTION 1: — user clicks a recipe, gets similar ones
# Input:  a recipe ID (int) from the database
# Output: DataFrame with columns: name, id, minutes, n_steps, n_ingredients,
#         avg_rating, similarity, shared_ingredients, insight

get_recommendations(recipe_id=12345, n=5)


# FUNCTION 2: — user input
# Input:  ingredients (list of strings), tags (list of strings),
#         numeric prefs (dict), optional name/description strings
# Output: DataFrame with columns: name, id, minutes, n_steps, n_ingredients,
#         avg_rating, similarity_score

recommend_from_user_input(
    user_ingredients=["chicken", "garlic", "olive oil"],
    user_tags=["dinner", "quick"],
    user_numeric={"minutes": 30, "n_steps": 6, "n_ingredients": 8},
    user_name="lemon garlic chicken",
    top_k=5
)

# Both return a pandas DataFrame. 

