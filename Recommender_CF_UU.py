import pandas as pd
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import norm
import sklearn.preprocessing as pp


class UUCFRecommender:
    UU = {} ## user-user similarities; constructed lazily
    
    def create_Ratings_Matrix(self):
        
        self.track_uris = self.ratings_train.track_uri.unique()
        self.track_uris.sort()
        self.pids = self.ratings_train.pid.unique()
        self.pids.sort()
        self.m = self.pids.size
        
        ## tracks and playlists should have consecutive indexes starting from 0
        self.track_uri_to_track_uriX = dict(zip(self.track_uris, range(0, self.track_uris.size)))
        self.track_uriX_to_track_uri = dict(zip(range(0, self.track_uris.size), self.track_uris))

        self.pid_to_pidX = dict(zip(self.pids, range(0, self.pids.size )))
        self.pidX_to_pid = dict(zip(range(0, self.pids.size), self.pids))
        
        self.R = sp.csr_matrix((self.ratings_train.rating, (self.ratings_train.pid.map(self.pid_to_pidX), self.ratings_train.track_uri.map(self.track_uri_to_track_uriX))))
        
        self.R_dok = self.R.todok()
    
    
    def compute_playlist_avgs(self):
        playlist_sums = self.R.sum(axis=1).A1 ## matrix converted to 1-D array via .A1
        self.playlist_cnts = (self.R != 0).sum(axis=1).A1
        self.playlist_avgs = playlist_sums / self.playlist_cnts
    
    def compute_pairwise_playlist_similarity(self, u_id, v_id):        
        
        u = self.R[u_id,:].copy()
        v = self.R[v_id,:].copy()
        
        ### IMPORTANT: Don't forget to prefix non local variables with 'self.'
        ########## START HERE ##########
        u.data -= self.playlist_avgs[u_id]
        v.data -= self.playlist_avgs[v_id]
        numerator = u.dot(v.T).A.item()
        denominator = norm(u) * norm(v)
        ##########  END HERE  ##########

        if denominator == 0:
            similarity = 0.;
        else:
            similarity = numerator/denominator

        return similarity
    
    def compute_playlist_similarities(self, u_id):
        if u_id in self.UU.keys(): ## persist
            return
        
        uU = np.empty((self.m,))

        ########## START HERE ##########
        R_copy = self.R.copy()

        R_copy.data -= np.repeat(self.playlist_avgs, np.diff(R_copy.indptr))  # mean-center
        R_copy = pp.normalize(R_copy)

        u = self.R[u_id, :].copy()
        u.data -= self.playlist_avgs[u_id]
        u = pp.normalize(u)

        uU = np.squeeze(R_copy.dot(u.T).A)
        ##########  END HERE  ##########

        self.UU[u_id] = uU
        
    
    def create_playlist_neighborhood(self, u_id, i_id):
        nh = {} ## the neighborhood dict with (playlist id: similarity) entries
        ## nh should not contain u_id and only include playlists that have rated i_id; there should be at most k neighbors
        self.compute_playlist_similarities(u_id)
        uU = self.UU[u_id].copy()
        
        uU_copy = uU.copy() ## so that we can modify it, but also keep the original

        ########## START HERE ##########
        if self.with_abs_sim:
            uU_copy = np.absolute(uU_copy)

        idx = np.argsort(uU_copy)
        # only playlists who rated i_id, and not the same playlist u_id
        item_map = np.array([(x, i_id) in self.R_dok and x != u_id for x in idx])
        idx = idx[item_map][-1:-1 * (self.k + 1):-1]  # get last k entries
        nh = dict(zip(idx, uU[idx]))
        ##########  END HERE  ##########

        return nh
    
    
    def predict_rating(self, u_id, i_id):

        nh = self.create_playlist_neighborhood(u_id, i_id)

        neighborhood_weighted_avg = 0.

        ########## START HERE ##########
        ### compute numerator and denominator
        denominator = sum(np.absolute(np.array(list(nh.values()))))
        if self.with_deviations:
            numerator = sum([(self.R[v_id, i_id] - self.playlist_avgs[v_id]) * w_uv for v_id, w_uv in nh.items()])
        else:
            numerator = sum([self.R[v_id, i_id] * w_uv for v_id, w_uv in nh.items()])
        ##########  END HERE  ##########
        
        if denominator != 0: ## avoid division by zero
            neighborhood_weighted_avg = numerator/denominator
        
        
        if self.with_deviations:
            prediction = self.playlist_avgs[u_id] + neighborhood_weighted_avg
#             print("prediction ", prediction, " (playlist_avg ", self.playlist_avgs[u_id], " offset ", neighborhood_weighted_avg, ")", sep="")
        else:
            prediction = neighborhood_weighted_avg
#             print("prediction ", prediction, " (playlist_avg ", self.playlist_avgs[u_id], ")", sep="")

        return prediction
    
    
    def __init__(self, with_abs_sim = True, with_deviations = True, k = 50):
        self.with_abs_sim = with_abs_sim
        self.with_deviations= with_deviations
        self.k = k
  

    def build_model(self, ratings_train, tracks = None):
        self.ratings_train = ratings_train

        self.create_Ratings_Matrix()
        self.compute_playlist_avgs()
    

    ### recommend up to topN items among those in item_ids for playlist_id
    def recommend(self, playlist_id, item_ids=None, topN=20):
        
        ########## START HERE ##########
        ### get a list of track_uris that playlist_id has rated in the ratings_train
        tracks_rated_by_playlist = self.ratings_train[self.ratings_train.pid == playlist_id].track_uri.values
        ##########  END HERE  ##########
        
        u_id = self.pid_to_pidX[playlist_id]
        
        recommendations = []
        
        if item_ids is None: ## recommend among all items
            item_ids = self.track_uris
        
        for item_id in item_ids:
            if item_id in tracks_rated_by_playlist:
                continue
            i_id = self.track_uri_to_track_uriX[item_id]
            rating = self.predict_rating(u_id, i_id)
            recommendations.append((item_id, rating))

        ########## START HERE ##########
        ### sort recommendations decreasing by their rating, get the topN, and store them back in recommendations
        recommendations = sorted(recommendations, key=lambda x: -x[1])[:topN]
        ##########  END HERE  ##########

        return [item_id for item_id, rating in recommendations]