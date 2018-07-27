import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:

    #DONE
    def create_mix(self, x):

        #pnames_list = self.ratings_train.loc[self.ratings_train['track_uri'] == x.track_uri][['name']]

        #print(pnames_list)

        #mix = pnames_list
        mix = [x.artist_name]
        mix.append(str(x.duration_ms))
        return " ".join(mix)

    #DONE
    def build_track_profiles(self):
        # Create a new mix feature
        self.tracks['mix'] = self.tracks.apply(self.create_mix, axis=1)
        print("ok")
        # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
        vectorizer = TfidfVectorizer(stop_words="english")
        
        if self.track_profile_type == 'extended':
            # Construct the required TF-IDF matrix by fitting and transforming the mix attribute
            self.tfidf = vectorizer.fit_transform(self.tracks['mix'])
            
        elif self.track_profile_type == 'overview':
            # Construct the required TF-IDF matrix by fitting and transforming the overview attribute
            self.tfidf = vectorizer.fit_transform(self.tracks['overview'])
        
        # Get features names (tokens)
        self.feature_names = vectorizer.get_feature_names()
    

    #DONE
    def get_track_profile(self, trackId):
        # Find the profile of the track with uri=track_uri in the tfidf_matrix
        # Use track_uris to find index of a track,
        # then use that index to find the tfidf representation of the track
        return self.tfidf[self.track_ids.index(trackId)]


    #DONE
    def get_track_profiles(self, ids):
        if ids.size==1:
            track_profiles_list = self.get_track_profile(ids)
        else:
            track_profiles_list = [self.get_track_profile(x) for x in ids]

        track_profiles = scipy.sparse.vstack(track_profiles_list)

        return track_profiles

    #DONE
    def build_playlist_profile(self, pid, positive_ratings):
        ## build a playlist profile, based on positive ratings (tracks contained in the playlist)
        # 1. Retrieve playlist "ratings"
        ratings_playlist = positive_ratings.loc[pid]
        
        # 2. Retrieve all track profiles that a given playlist has "rated" as positive, name it playlist_track_profiles
        playlist_track_profiles = self.get_track_profiles(ratings_playlist['track_uri'])
        
        # 3. Create playlist track ratings in the form of an array, name it playlist_track_ratings
        # use reshape(-1,1)
        playlist_track_ratings = ratings_playlist['rating']
        if playlist_track_ratings.size != 1:
            playlist_track_ratings = playlist_track_ratings.values.reshape(-1, 1)
        
        # 4. Now, a playlist profile should be computed as average of track profiles that a playlist has rated as positive
        # Use previously obtained playlist_track_profiles and playlist_track_ratings
        playlist_ratings_wavg = playlist_track_profiles.multiply(playlist_track_ratings).sum(axis=0)
        
        playlist_profile = sklearn.preprocessing.normalize(playlist_ratings_wavg)

        return playlist_profile
   
    #DONE?
    def build_playlist_profiles(self):
        positive_ratings = self.ratings_train[self.ratings_train.rating>0].set_index('pid')
        self.playlist_profiles = {}
        for pid in positive_ratings.index.unique():
            self.playlist_profiles[pid] = self.build_playlist_profile(pid, positive_ratings)
        print('ok playlist profiles')
        
    
    def get_similar_tracks_to_playlist_profile(self, pid, topN):
        # compute the cosine similarity between the playlist profile and all track profiles
        # use linear_kernel function from sklearn.metrics.pairwise
        cosine_sim = linear_kernel(self.playlist_profiles[pid], self.tfidf)
        
        # Get the top similar items
        similar_indices = cosine_sim.argsort().flatten()[-topN:]
        # Sort the similar items by similarity
        similar_items = sorted([(self.track_ids[i], cosine_sim[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
    
    
    def __init__(self, profile_type='overview'):
        self.track_profile_type = profile_type
  
    #DONE
    def build_model(self, ratings_train, tracks):
        
        self.ratings_train = ratings_train
        self.tracks = tracks

        self.trackIds = self.ratings_train.track_uri.unique()
        self.trackIds.sort()
        self.playlistIds = self.ratings_train.pid.unique()
        self.playlistIds.sort()
        
        self.track_ids = self.tracks['track_uri'].tolist()
        
        self.build_track_profiles()
        self.build_playlist_profiles()
        

    ### recommend up to topN items among those in track_ids for playlist_id
    def recommend(self, playlist_id, track_ids=None, topN=100):

        ### get a list of trackIds that playlist_id has rated in the ratings_train
        tracks_rated_by_playlist = self.ratings_train[self.ratings_train.pid == playlist_id].track_uri.values
        
        similar_items = self.get_similar_tracks_to_playlist_profile(playlist_id, 0) ## 0 means return all
    
        recommendations = []

        
        if track_ids is None:
            track_ids = self.trackIds
                
        for item_id, strength in similar_items:
            if item_id in tracks_rated_by_playlist:
                continue
            if item_id not in track_ids:
                continue
            recommendations.append((item_id, strength))

        ### sort recommendations decreasing by their rating, get the topN, and store them back in recommendations
        recommendations = sorted(recommendations, key=lambda x: -x[1])[:topN]
        
        return [item_id for item_id, rating in recommendations]