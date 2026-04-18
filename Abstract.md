This hand on was completed as part of Group 1 assignment where the 
assigned task was Content Based Filtering. The dataset used is the 
MovieLens 100K dataset which contains 4 files — movies with 9742 
entries, ratings with 100836 user ratings, tags with 3683 entries 
and links connecting movies to IMDB and TMDB. Movie features were 
extracted by parsing genres using one hot encoding and extracting 
release years from movie titles. A cosine similarity matrix was 
computed between all movies based on genre features to measure 
movie to movie similarity. User profiles were constructed by 
averaging the genre feature vectors of movies rated above 3.5 
by each user based on their historical rating preferences. 
Personalized recommendations were generated for each user by 
computing similarity between their profile and all unrated movies. 
The results demonstrate that content based filtering effectively 
captures user preferences and provides meaningful movie recommendations 
based on genre similarity and historical viewing behavior.
