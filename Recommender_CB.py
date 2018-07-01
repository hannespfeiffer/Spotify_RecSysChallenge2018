import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
      
    def create_mix(self, x):
        ########## BEGIN HERE ##########
        mix = x.genres or []
        mix.extend(x.cast)
        mix.append(x.director)
        mix.extend(x.keywords)
        return " ".join(mix)

    
    def build_movie_profiles(self):
        # Create a new mix feature
        self.movies['mix'] = self.movies.apply(self.create_mix, axis=1)
        
        # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
        ########## BEGIN HERE ##########
        vectorizer = TfidfVectorizer(stop_words="english")
        ##########  END HERE  ##########
        
        if self.movie_profile_type == 'extended':
            # Construct the required TF-IDF matrix by fitting and transforming the mix attribute
            ########## BEGIN HERE ##########
            self.tfidf = vectorizer.fit_transform(self.movies['mix'])
            ##########  END HERE  ##########
            
        elif self.movie_profile_type == 'overview':
            # Construct the required TF-IDF matrix by fitting and transforming the overview attribute
            ########## BEGIN HERE ##########
            self.tfidf = vectorizer.fit_transform(self.movies['overview'])
            ##########  END HERE  ##########
        
        # Get features names (tokens)
        self.feature_names = vectorizer.get_feature_names()
    
    
    def get_movie_profile(self, movieId):
        # Find the profile of the movie with id=movieId in the tfidf_matrix
        ######### BEGIN HERE ##########
        # TIP: Use movie_ids to find index of a movie, 
        # then use that index to find the tfidf representation of the movie
        ##########  END HERE  ##########
        return self.tfidf[self.movie_ids.index(movieId)]

    
    def get_movie_profiles(self, ids):
        if ids.size==1:
            movie_profiles_list = self.get_movie_profile(ids)
        else:
            movie_profiles_list = [self.get_movie_profile(x) for x in ids]
        movie_profiles = scipy.sparse.vstack(movie_profiles_list)
        return movie_profiles

    
    def build_user_profile(self, user_id, positive_ratings):
        ## build a user profile, based on positive ratings provided by the user
        
        ######### BEGIN HERE ##########
        # 1. Retrieve user ratings
        # TIP: use positive_ratings and loc         
        ratings_user = positive_ratings.loc[user_id]
        
        # 2. Retrieve all movie profiles that a given user has rated as positive, name it user_movie_profiles
        user_movie_profiles = self.get_movie_profiles(ratings_user['movieId'])
        
        # 3. Create user movie ratings in the form of an array, name it user_movie_ratings
        # TIP: use reshape(-1,1)
        user_movie_ratings = ratings_user['rating']
        if user_movie_ratings.size != 1:
            user_movie_ratings = user_movie_ratings.values.reshape(-1, 1)
        
        # 4. Now, a user profile should be computed as average of movie profiles that a user has rated as positive
        # TIP: Use previously obtained user_movie_profiles and user_movie_ratings
        user_ratings_wavg = user_movie_profiles.multiply(user_movie_ratings).sum(axis=0)
        ##########  END HERE  ##########
        
        user_profile = sklearn.preprocessing.normalize(user_ratings_wavg)
        
        return user_profile
   
    
    def build_user_profiles(self): 
        positive_ratings = self.ratings_train[self.ratings_train.rating>3].set_index('userId')
        self.user_profiles = {}
        for user_id in positive_ratings.index.unique():
            self.user_profiles[user_id] = self.build_user_profile(user_id, positive_ratings)
        
    
    def get_similar_items_to_user_profile(self, user_id, topN):
        
        ######### BEGIN HERE ##########
        # compute the cosine similarity between the user profile and all item profiles
        # TIP: use linear_kernel function from sklearn.metrics.pairwise
        cosine_sim = linear_kernel(self.user_profiles[user_id], self.tfidf)
        ##########  END HERE  ##########
        
        # Get the top similar items
        similar_indices = cosine_sim.argsort().flatten()[-topN:]
        # Sort the similar items by similarity
        similar_items = sorted([(self.movie_ids[i], cosine_sim[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
    
    
    def __init__(self, movie_profile_type='overview'):
        self.movie_profile_type = movie_profile_type
  

    def build_model(self, ratings_train, movies):
        
        self.ratings_train = ratings_train
        self.movies = movies
        
        self.movieIds = self.ratings_train.movieId.unique()
        self.movieIds.sort()
        self.userIds = self.ratings_train.userId.unique()
        self.userIds.sort()
        
        self.movie_ids = self.movies['movieId'].tolist()
        
        self.build_movie_profiles()
        self.build_user_profiles()
        

    ### recommend up to topN items among those in item_ids for user_id
    def recommend(self, user_id, item_ids=None, topN=20):
        
        ########## BEGIN HERE ##########
        ### get a list of movieIds that user_id has rated in the ratings_train 
        movies_rated_by_user = self.ratings_train[self.ratings_train.userId == user_id].movieId.values
        ##########  END HERE  ##########
        
        similar_items = self.get_similar_items_to_user_profile(user_id, 0) ## 0 means return all 
    
        recommendations = []

        
        if item_ids is None:
            item_ids = self.movieIds
                
        for item_id, strength in similar_items:
            if item_id in movies_rated_by_user:
                continue
            if item_id not in item_ids:
                continue
            recommendations.append((item_id, strength))    
        
        ########## BEGIN HERE ##########
        ### sort recommendations decreasing by their rating, get the topN, and store them back in recommendations
        recommendations = sorted(recommendations, key=lambda x: -x[1])[:topN]
        ##########  END HERE  ##########
        
        return [item_id for item_id, rating in recommendations]