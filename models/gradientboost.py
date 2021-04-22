from sklearn.tree import DecisionTreeRegressor
import numpy as np

class Gboost():
    def __init__(self,n_trees=100,lr=0.1,max_depth=3):
        self.n_trees=n_trees
        self.lr=lr
        self.max_depth=max_depth
        self.trees=[]
        for i in range(n_trees):
            self.trees.append(DecisionTreeRegressor( max_depth=self.max_depth))

    def fit(self,X,y):
        self.X=X
        self.y=y
        F0 = np.mean(y)
        for m in range(self.n_trees):
            r=np.sum([y,-F0.reshape(-1,1)],axis=0)#negative gradient
            tree=self.trees[m].fit(X,r)
            h=tree.predict(X)
            F=F0+self.lr*h
            F0=F
        return

    def predict(self,x):
        
        pred=np.ones(x.shape[0])*np.mean(self.y)
        for i,tree in enumerate(self.trees):
            pred+= self.lr*tree.predict(x).reshape(-1)
        return pred
if __name__ == "__main__":
   mdl=Gboost(n_trees=100,lr=0.1)