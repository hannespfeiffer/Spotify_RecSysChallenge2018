import pandas as pd
import numpy as np
import math
from scipy import sparse as sp
from scipy.sparse.linalg import norm
import sklearn.preprocessing as pp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils import shuffle
from numpy.core.umath_tests import inner1d
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error


def get_one_hot(targets):
    lb = pp.LabelBinarizer(sparse_output=True)
    lb.fit(targets.reshape(-1))
    return lb.transform(targets)


def to_Xy_format(R):
    n_users = R.shape[0]
    n_items = R.shape[1]

    users, items = R.nonzero()
    n_ratings = users.size

    Xu = get_one_hot(users)
    Xi = get_one_hot(items)

    R = sp.csr_matrix(R)
    y = R.data
    X = sp.hstack([Xu, Xi])

    return X, y, n_users, n_items


def to_R_format(X, y, n_users, n_items):
    Xu = X.tocsc()[:, :n_users]
    Xi = X.tocsc()[:, n_users:]

    Xu = Xu.tocsr()
    Xi = Xi.tocsr()

    R_rec = sp.coo_matrix( (y, (Xu.indices, Xi.indices)), shape=(n_users, n_items) )

    return R_rec.tocsr()


def to_UI_arrays(X, n_users, n_items):
    Xu = X.tocsc()[:, :n_users]
    Xi = X.tocsc()[:, n_users:]
        
    U = Xu.argmax(axis=1).A1
    I = Xi.argmax(axis=1).A1
    
    return U, I


VERBOSE = False

#### set DEBUG to False when preparing your report
DEBUG = True


class MFRecommender(BaseEstimator, RegressorMixin):
    
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
         
    
    def __init__(self, k = 5, eta = 0.002, lam = 0., n_epochs = 5, s_batch = 1, w_average = True, w_biases = True, rnd_mean = 0, rnd_std = 0.1):
        self.k = k
        self.eta = eta
        self.lam = lam
        self.n_epochs = n_epochs
        self.s_batch = s_batch
        self.w_average = w_average
        self.w_biases = w_biases
        self.rnd_mean = rnd_mean
        self.rnd_std = rnd_std
 
    
    
    def fit_init(self, X, y, n_users, n_items):
        X, y = check_X_y(X, y, accept_sparse=True)

        self.n_users_ = n_users
        self.n_items_ = n_items
        self.n_ratings_ = X.shape[0]

        self.X_ = X
        self.y_ = y


        ## USEFUL for debugging: remove randomness by fixing the random seed
        if DEBUG:
            np.random.seed(42)


        ########## BEGIN HERE ##########

        ## compute average rating
        self.mu_ = np.mean(y)

        ## randomly initialize P_, Q_, bu_, bi_
        self.P_ = np.random.normal(self.rnd_mean, self.rnd_std, (n_users, self.k))
        self.Q_ = np.random.normal(self.rnd_mean, self.rnd_std, (n_items, self.k))
        if self.w_biases:
            self.bu_ = np.random.normal(self.rnd_mean, self.rnd_std, (n_users,))
            self.bi_ = np.random.normal(self.rnd_mean, self.rnd_std, (n_items,))
        ##########  END HERE  ########## 


        ## random shuffle the training data
        self.X_, self.y_ = shuffle(self.X_, self.y_)

        return self
 

    def fit_sgd(self): ## stochastic gradient descent
    
        U, I = to_UI_arrays(self.X_, self.n_users_, self.n_items_)


        if VERBOSE:
            print("start of training")

        for epoch in range(self.n_epochs):

            epoch_loss = 0.

            for index in range(self.y_.shape[0]):
                u = U[index]
                i = I[index]
                r_ui = self.y_[index]


                ########## BEGIN HERE ##########
                prediction = self.P_[u].T.dot(self.Q_[i])
                if self.w_average:
                    prediction += self.mu_
                if self.w_biases:
                    prediction += self.bu_[u] + self.bi_[i]

                err = r_ui - prediction  # calculate difference of real and predicted value

                # update parameters
                # P_val = self.P_[u].copy()
                self.P_[u] += self.eta * (err * self.Q_[i] - self.lam * self.P_[u])
                self.Q_[i] += self.eta * (err * self.P_[u] - self.lam * self.Q_[i])
                if self.w_biases:
                    self.bu_[u] += self.eta * (err - self.lam * self.bu_[u])
                    self.bi_[i] += self.eta * (err - self.lam * self.bi_[i])
                ##########  END HERE  ##########

                epoch_loss += err * err


            ## epoch is done
            epoch_loss /= self.n_ratings_
            if VERBOSE:
                print("epoch", epoch, "loss", epoch_loss)

        return self


    def fit_mgd(self):  ## mini-batch gradient descent

        U, I = to_UI_arrays(self.X_, self.n_users_, self.n_items_)

        if VERBOSE:
            print("start of training")

        for epoch in range(self.n_epochs):

            epoch_loss = 0.

            for i in range(math.ceil(self.n_ratings_ / self.s_batch)):

                # get indices for current batch
                Ub = U[i * self.s_batch: (i + 1) * self.s_batch]
                Ib = I[i * self.s_batch: (i + 1) * self.s_batch]
                r_ui = self.y_[i * self.s_batch: (i + 1) * self.s_batch]

                prediction = inner1d(self.P_[Ub], self.Q_[Ib])
                if self.w_average:
                    prediction += self.mu_
                if self.w_biases:
                    prediction += self.bu_[Ub] + self.bi_[Ib]

                # calculate difference of real and predicted value
                err = r_ui - prediction

                # update parameters
                P_grad = self.eta * (err[:, np.newaxis] * self.Q_[Ib] - self.lam * self.P_[Ub])
                Q_grad = self.eta * (err[:, np.newaxis] * self.P_[Ub] - self.lam * self.Q_[Ib])
                for j in range(P_grad.shape[0]):
                    self.P_[Ub[j]] += P_grad[j]
                    self.Q_[Ib[j]] += Q_grad[j]
                if self.w_biases:
                    self.bu_[Ub] += self.eta * (err - self.lam * self.bu_[Ub])
                    self.bi_[Ib] += self.eta * (err - self.lam * self.bi_[Ib])

                epoch_loss += sum(err * err / self.n_ratings_)

            if VERBOSE:
                print("epoch", epoch, "loss", epoch_loss)

        return self


    def fit(self, X, y, n_users, n_items):

        self.fit_init(X, y, n_users, n_items)

        if VERBOSE:
            print(self.get_params())

        if self.s_batch == 1: ## stochastic gradient descent
            self.fit_sgd()
        else: ## mini-batch stochastic gradient descent -- BONUS point
            self.fit_mgd()

        return self
    
    def predict(self, X, y=None):
        try:
            getattr(self, "n_users_")
        except AttributeError:
            raise RuntimeError("You must train before predicting!")


        U, I = to_UI_arrays(X, self.n_users_, self.n_items_)

        y_pred = np.ndarray(U.shape[0])

        for index in range(U.shape[0]):
            u = U[index]
            i = I[index]

            ########## BEGIN HERE ##########
            prediction = self.P_[u].T.dot(self.Q_[i])
            if self.w_average:
                prediction += self.mu_
            if self.w_biases:
                prediction += self.bu_[u] + self.bi_[i]
            ##########  END HERE  ##########

            y_pred[index] = prediction


        if y is not None:
            mse = mean_squared_error(y_pred, y)
            rmse = math.sqrt(mse)
            print("RMSE", rmse)

        return y_pred

    
    def computeRMSE(self):
        y_pred = self.predict(self.X_)
        mse = mean_squared_error(y_pred, self.y_)
        rmse = math.sqrt(mse)
        return rmse, mse

    
    def score(self, X, y=None):
        if y is None:
            rmse, mse = self.computeRMSE()
            return -rmse
        else:
            y_pred = self.predict(X)
            mse = mean_squared_error(y_pred, y)
            return -math.sqrt(mse)
    
    
    def build_model(self, ratings_train, movies = None): 
        self.ratings_train = ratings_train
        self.create_Ratings_Matrix()
        
        X, y, n_users, n_items = to_Xy_format(self.R)
        self.fit(X, y, n_users, n_items)
        
    
    ### recommend up to topN items among those in item_ids for user_id
    def recommend(self, user_id, item_ids=None, topN=20):
        
        ########## BEGIN HERE ##########
        ### get a list of movieIds that user_id has rated in the ratings_train 
        movies_rated_by_user = self.ratings_train[self.ratings_train.userId == user_id].movieId.values
        ##########  END HERE  ##########
        
        u_id = self.userId_to_userIDX[user_id]
        
        y_pred = self.predict(self.X_)
             
        recommendations = []
        
        if item_ids is None: ## recommend among all items
            item_ids = self.movieIds
        
        for item_id in item_ids:
            if item_id in movies_rated_by_user:
                continue
            i_id = self.movieId_to_movieIDX[item_id]
            rating = y_pred[i_id]
            recommendations.append((item_id, rating))
                
        ########## BEGIN HERE ##########
        ### sort recommendations decreasing by their rating, get the topN, and store them back in recommendations
        recommendations = sorted(recommendations, key=lambda x: -x[1])[:topN]
        ##########  END HERE  ##########
        
        return [item_id for item_id, rating in recommendations]