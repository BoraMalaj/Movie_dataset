This project was completed as part of Group 1 assignment where the 
assigned task was Content Based Filtering. The dataset used is the 
MovieLens dataset which contains 9742 movies with features including 
movieId, title and genres. Movie features were extracted by parsing 
genres using one hot encoding and extracting the release year from 
movie titles. A cosine similarity matrix was computed between all 
movies based on their genre features to measure movie similarity. 
A movie recommendation function was built to suggest the top similar 
movies for any given title. User profiles were constructed by averaging 
the genre feature vectors of historically watched movies for each user 
and personalized recommendations were generated based on each user 
profile similarity to all available movies. The results demonstrate 
that content based filtering is an effective approach for movie 
recommendation when user historical preferences are available.
