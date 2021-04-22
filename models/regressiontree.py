import numpy as np

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.value = value
class RegressionTree():
    def __init__(self, min_samples_split=2, max_depth=3):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.i=0
    
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = X.shape
        best_split = {}
        if num_samples>=self.min_samples_split and curr_depth<self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)            
            if best_split["var_red"]>1e-7:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        leaf_value = np.mean(Y)
        return Node(value=leaf_value)
    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_var_red = -np.inf
        if num_samples>=self.min_samples_split :
            for feature_index in range(num_features):
                possible_thresholds=[np.quantile(dataset[:, feature_index],1/i) for i in range(1,20)]
                
    #            possible_thresholds=np.unique(dataset[:, feature_index])
                for  threshold in possible_thresholds:
                    dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                    if len(dataset_left)>2 and len(dataset_right)>2:
                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                        #  information gain
                        curr_var_red = self.variance_reduction(y, left_y, right_y)
                        if curr_var_red>max_var_red:
                            best_split["feature_index"] = feature_index
                            best_split["threshold"] = threshold
                            best_split["dataset_left"] = dataset_left
                            best_split["dataset_right"] = dataset_right
                            best_split["var_red"] = curr_var_red
                            max_var_red = curr_var_red
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left=[]
        dataset_right=[]
        for row in dataset:
            if row[feature_index]<=threshold:
                dataset_left.append(row) 
            else : dataset_right.append(row)
        return np.array(dataset_left), np.array(dataset_right)
   
    def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    def make_prediction(self, x, tree):
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):        
        preditions = np.array([self.make_prediction(x, self.root) for x in X])
        return preditions

if __name__=="__main__":     
    mdl=RegressionTree()