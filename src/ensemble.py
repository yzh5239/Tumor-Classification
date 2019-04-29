from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle


class random_forest:
    def __init__(self, data = None, label = None, num_tree = 100, train_ratio = 0.8):
        data, label = shuffle(data, label, random_state=0)
        self.train_data = data[0:int(len(data)*train_ratio)]
        self.train_label = label[0:int(len(label)*train_ratio)]
        self.test_data = data[int(len(data)*train_ratio):]
        self.test_label = label[int(len(label)*train_ratio):]
        self.model = None
        self.num_tree = num_tree
        
    def fit(self):
        clf=RandomForestClassifier(n_estimators=self.num_tree)
        print('Start training')
        clf.fit(self.train_data, self.train_label)
        self.model = clf
        
    def predict(self):
        pred = self.model.predict(self.test_data)
        print("Test Accuracy:",metrics.accuracy_score(pred, self.test_label))        

