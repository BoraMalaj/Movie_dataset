#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
links = pd.read_csv('links.csv')

print("All files loaded successfully!")
print(f"Movies  : {movies.shape}")
print(f"Ratings : {ratings.shape}")
print(f"Tags    : {tags.shape}")
print(f"Links   : {links.shape}")

print("\nMovies columns:", movies.columns.tolist())
print("Ratings columns:", ratings.columns.tolist())
print("Tags columns:", tags.columns.tolist())
print("Links columns:", links.columns.tolist())


# In[31]:


# Extract year from title
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['title_clean'] = movies['title'].str.replace(
    r'\s*\(\d{4}\)', '', regex=True).str.strip()

# Split genres
movies['genres_list'] = movies['genres'].str.split('|')

# One Hot Encode genres
genres_encoded = movies['genres'].str.get_dummies(sep='|')

print("Movie features extracted!")
print("Shape:", genres_encoded.shape)
print("\nUnique Genres:", genres_encoded.columns.tolist())


# In[33]:


# Merge movies with ratings
movie_ratings = ratings.merge(movies, on='movieId', how='left')

# Merge with tags
movie_tags = tags.groupby('movieId')['tag'].apply(
    lambda x: ' '.join(x)).reset_index()
movie_ratings = movie_ratings.merge(movie_tags, on='movieId', how='left')

# Merge with links
movie_ratings = movie_ratings.merge(links, on='movieId', how='left')

print("All datasets merged!")
print("Shape:", movie_ratings.shape)
print(movie_ratings.head())


# In[35]:


# Average rating per movie
movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()

movie_stats = movie_stats.merge(movies[['movieId', 'title_clean', 
                                         'genres', 'year']], 
                                  on='movieId', how='left')

movie_stats['avg_rating'] = movie_stats['avg_rating'].round(2)

print("Movie Statistics:")
print(movie_stats.sort_values('num_ratings', ascending=False).head(10))


# In[37]:


from sklearn.metrics.pairwise import cosine_similarity

# Build feature matrix
feature_matrix = genres_encoded.values

# Compute cosine similarity
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

print("Cosine Similarity Matrix:")
print("Shape:", cosine_sim.shape)


# In[53]:


def get_recommendations(movie_title, n=10):
    matches = movies[movies['title_clean'].str.contains(
        movie_title, case=False, na=False)]

    if len(matches) == 0:
        print(f"Movie '{movie_title}' not found!")
        return

    idx = matches.index[0]
    print(f"Finding movies similar to: {movies['title_clean'][idx]}")

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]

    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    recommendations = movies.iloc[movie_indices][
        ['title_clean', 'genres', 'year']].copy()
    recommendations['similarity_score'] = [round(s, 3) for s in scores]

    return recommendations

print(" Recommendations for Heat:")
print(get_recommendations('Heat', n=10))


# In[55]:


def build_user_profile(user_id, min_rating=3.5):
    # Get movies rated highly by user
    user_ratings = ratings[
        (ratings['userId'] == user_id) & 
        (ratings['rating'] >= min_rating)]

    if len(user_ratings) == 0:
        print(f"No ratings found for User {user_id}")
        return None

    # Get movie indices
    movie_ids = user_ratings['movieId'].values
    movie_indices = movies[movies['movieId'].isin(movie_ids)].index

    # Build profile as average of genre vectors
    profile = genres_encoded.iloc[movie_indices].mean().values

    return profile

# Build profiles for sample users
user_ids = [1, 2, 3]
user_profiles = {}

for uid in user_ids:
    profile = build_user_profile(uid)
    if profile is not None:
        user_profiles[uid] = profile
        print(f"\nUser {uid} Profile (genre preferences):")
        profile_series = pd.Series(profile, index=genres_encoded.columns)
        print(profile_series[profile_series > 0].round(2).sort_values(
            ascending=False))


# In[57]:


def recommend_for_user(user_id, n=5):
    if user_id not in user_profiles:
        print(f"No profile for User {user_id}")
        return

    profile = user_profiles[user_id]

    # Get movies already rated by user
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].values

    # Compute similarity
    sim_scores = cosine_similarity([profile], feature_matrix)[0]

    # Create recommendations dataframe
    recs = movies.copy()
    recs['similarity_score'] = sim_scores

    # Exclude already rated movies
    recs = recs[~recs['movieId'].isin(rated_movies)]

    # Sort by similarity
    recs = recs.sort_values('similarity_score', ascending=False).head(n)

    return recs[['title_clean', 'genres', 'year', 'similarity_score']]

for uid in user_ids:
    print(f"\n Recommendations for User {uid}:")
    print(recommend_for_user(uid, n=5))
    print("=" * 60)


# In[59]:


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Genre Distribution
genres_encoded.sum().sort_values(ascending=False).plot(
    kind='bar', ax=axes[0,0], color='steelblue', edgecolor='black')
axes[0,0].set_title('Genre Distribution')
axes[0,0].set_xlabel('Genre')
axes[0,0].set_ylabel('Number of Movies')
axes[0,0].tick_params(axis='x', rotation=45)

# Rating Distribution
ratings['rating'].value_counts().sort_index().plot(
    kind='bar', ax=axes[0,1], color='green', edgecolor='black')
axes[0,1].set_title('Rating Distribution')
axes[0,1].set_xlabel('Rating')
axes[0,1].set_ylabel('Count')
axes[0,1].tick_params(axis='x', rotation=0)

# Top 10 Most Rated Movies
top_movies = movie_stats.sort_values(
    'num_ratings', ascending=False).head(10)
axes[1,0].barh(top_movies['title_clean'], 
               top_movies['num_ratings'],
               color='orange', edgecolor='black')
axes[1,0].set_title('Top 10 Most Rated Movies')
axes[1,0].set_xlabel('Number of Ratings')

# Top 10 Highest Rated Movies (min 50 ratings)
top_rated = movie_stats[movie_stats['num_ratings'] >= 50].sort_values(
    'avg_rating', ascending=False).head(10)
axes[1,1].barh(top_rated['title_clean'],
               top_rated['avg_rating'],
               color='red', edgecolor='black')
axes[1,1].set_title('Top 10 Highest Rated Movies (min 50 ratings)')
axes[1,1].set_xlabel('Average Rating')

plt.suptitle('MovieLens 100K - Dataset Analysis', fontsize=14)
plt.tight_layout()
plt.show()

