import pandas as pd
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import norm
import sklearn.preprocessing as pp


class UUCFRecommender:
    UU = {} ## user-user similarities; constructed lazily
    
    def create_Ratings_Matrix(self):
        
        self.movieIds = self.ratings_train.movieId.unique()
        self.movieIds.sort()
        self.userIds = self.ratings_train.userId.unique()
        self.userIds.sort()
        self.m = self.userIds.size
        
        ## movies and users should have consecutive indexes starting from 0
        self.movieId_to_movieIDX = dict(zip(self.movieIds, range(0, self.movieIds.size)))
        self.movieIDX_to_movieId = dict(zip(range(0, self.movieIds.size), self.movieIds))

        self.userId_to_userIDX = dict(zip(self.userIds, range(0, self.userIds.size )))
        self.userIDX_to_userId = dict(zip(range(0, self.userIds.size), self.userIds))
        
        self.R = sp.csr_matrix((self.ratings_train.rating, (self.ratings_train.userId.map(self.userId_to_userIDX), self.ratings_train.movieId.map(self.movieId_to_movieIDX))))
        
        self.R_dok = self.R.todok()
    
    
    def compute_user_avgs(self):
        user_sums = self.R.sum(axis=1).A1 ## matrix converted to 1-D array via .A1
        self.user_cnts = (self.R != 0).sum(axis=1).A1
        self.user_avgs = user_sums / self.user_cnts
    
    def compute_pairwise_user_similarity(self, u_id, v_id):        
        
        u = self.R[u_id,:].copy()
        v = self.R[v_id,:].copy()
        
        ### IMPORTANT: Don't forget to prefix non local variables with 'self.'
        ########## START HERE ##########
        u.data -= self.user_avgs[u_id]
        v.data -= self.user_avgs[v_id]
        numerator = u.dot(v.T).A.item()
        denominator = norm(u) * norm(v)
        ##########  END HERE  ##########

        if denominator == 0:
            similarity = 0.;
        else:
            similarity = numerator/denominator

        return similarity
    
    def compute_user_similarities(self, u_id):
        if u_id in self.UU.keys(): ## persist
            return
        
        uU = np.empty((self.m,))

        ########## START HERE ##########
        R_copy = self.R.copy()

        R_copy.data -= np.repeat(self.user_avgs, np.diff(R_copy.indptr))  # mean-center
        R_copy = pp.normalize(R_copy)

        u = self.R[u_id, :].copy()
        u.data -= self.user_avgs[u_id]
        u = pp.normalize(u)

        uU = np.squeeze(R_copy.dot(u.T).A)
        ##########  END HERE  ##########

        self.UU[u_id] = uU
        
    
    def create_user_neighborhood(self, u_id, i_id):
        nh = {} ## the neighborhood dict with (user id: similarity) entries
        ## nh should not contain u_id and only include users that have rated i_id; there should be at most k neighbors
        self.compute_user_similarities(u_id)
        uU = self.UU[u_id].copy()
        
        uU_copy = uU.copy() ## so that we can modify it, but also keep the original

        ########## START HERE ##########
        if self.with_abs_sim:
            uU_copy = np.absolute(uU_copy)

        idx = np.argsort(uU_copy)
        # only users who rated i_id, and not the same user u_id
        item_map = np.array([(x, i_id) in self.R_dok and x != u_id for x in idx])
        idx = idx[item_map][-1:-1 * (self.k + 1):-1]  # get last k entries
        nh = dict(zip(idx, uU[idx]))
        ##########  END HERE  ##########

        return nh
    
    
    def predict_rating(self, u_id, i_id):

        nh = self.create_user_neighborhood(u_id, i_id)

        neighborhood_weighted_avg = 0.

        ########## START HERE ##########
        ### compute numerator and denominator
        denominator = sum(np.absolute(np.array(list(nh.values()))))
        if self.with_deviations:
            numerator = sum([(self.R[v_id, i_id] - self.user_avgs[v_id]) * w_uv for v_id, w_uv in nh.items()])
        else:
            numerator = sum([self.R[v_id, i_id] * w_uv for v_id, w_uv in nh.items()])
        ##########  END HERE  ##########
        
        if denominator != 0: ## avoid division by zero
            neighborhood_weighted_avg = numerator/denominator
        
        
        if self.with_deviations:
            prediction = self.user_avgs[u_id] + neighborhood_weighted_avg
#             print("prediction ", prediction, " (user_avg ", self.user_avgs[u_id], " offset ", neighborhood_weighted_avg, ")", sep="")
        else:
            prediction = neighborhood_weighted_avg
#             print("prediction ", prediction, " (user_avg ", self.user_avgs[u_id], ")", sep="")

        return prediction
    
    
    def __init__(self, with_abs_sim = True, with_deviations = True, k = 50):
        self.with_abs_sim = with_abs_sim
        self.with_deviations= with_deviations
        self.k = k
  

    def build_model(self, ratings_train, movies = None):
        self.ratings_train = ratings_train

        self.create_Ratings_Matrix()
        self.compute_user_avgs()
    

    ### recommend up to topN items among those in item_ids for user_id
    def recommend(self, user_id, item_ids=None, topN=20):
        
        ########## START HERE ##########
        ### get a list of movieIds that user_id has rated in the ratings_train 
        movies_rated_by_user = self.ratings_train[self.ratings_train.userId == user_id].movieId.values
        ##########  END HERE  ##########
        
        u_id = self.userId_to_userIDX[user_id]
        
        recommendations = []
        
        if item_ids is None: ## recommend among all items
            item_ids = self.movieIds
        
        for item_id in item_ids:
            if item_id in movies_rated_by_user:
                continue
            i_id = self.movieId_to_movieIDX[item_id]
            rating = self.predict_rating(u_id, i_id)
            recommendations.append((item_id, rating))

        ########## START HERE ##########
        ### sort recommendations decreasing by their rating, get the topN, and store them back in recommendations
        recommendations = sorted(recommendations, key=lambda x: -x[1])[:topN]
        ##########  END HERE  ##########

        return [item_id for item_id, rating in recommendations]