from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle


class random_forest:
    def __init__(self, data = None, label = None, num_tree = 100):
        data, label = shuffle(data, label, random_state=0)
        self.cross_data = []
        self.cross_label = []
        self.cross_data.append(data[0:int(len(data)*0.2)])
        self.cross_label.append(label[0:int(len(label)*0.2)])
        self.cross_data.append(data[int(len(data)*0.2):int(len(data)*0.4)])
        self.cross_label.append(label[int(len(label)*0.2):int(len(data)*0.4)])
        self.cross_data.append(data[int(len(data)*0.4):int(len(data)*0.6)])
        self.cross_label.append(label[int(len(label)*0.4):int(len(data)*0.6)])
        self.cross_data.append(data[int(len(data)*0.6):int(len(data)*0.8)])
        self.cross_label.append(label[int(len(label)*0.6):int(len(data)*0.8)])
        self.cross_data.append(data[int(len(data)*0.8):])
        self.cross_label.append(label[int(len(label)*0.8):])
        self.model = None
        self.num_tree = num_tree
        
    def fit(self, cross_valid):
        clf=RandomForestClassifier(n_estimators=self.num_tree)
        train_data = []
        for i in range(5):
            if(i != cross_valid):
                for j in range(len(self.cross_data[i])):
                    train_data.append(self.cross_data[i][j])
        train_label = []
        for i in range(5):
            if(i != cross_valid):
                for j in range(len(self.cross_label[i])):
                    train_label.append(self.cross_label[i][j])
        print('Start training')
        clf.fit(train_data, train_label)
        self.model = clf
        
    def predict(self, cross_valid):
        test_data = []
        for i in range(5):
            if(i == cross_valid):
                for j in range(len(self.cross_data[i])):
                    test_data.append(self.cross_data[i][j])
        test_label = []
        for i in range(5):
            if(i == cross_valid):
                for j in range(len(self.cross_label[i])):
                    test_label.append(self.cross_label[i][j])
        pred = self.model.predict(test_data)
        return metrics.accuracy_score(pred, test_label)   

