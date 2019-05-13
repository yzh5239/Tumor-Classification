import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

class kmeans:
    def __init__(self, data,label,num_classes = 5):
        self.data = data
        self.label = label
        self.num_classes = num_classes
        
    def predict(self):
        kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(self.data)
        
        pred = kmeans.labels_
        
        translate = np.array([-1,-1,-1,-1,-1])
        for i in range(len(pred)):
            if(pred[i] != self.label[i] and translate[pred[i]] == -1):
                translate[pred[i]] = self.label[i]
            if(translate[translate == -1] == []):
                break
        for i in range(len(translate)):
            if(translate[i] == -1):
                translate[i] = i
                
        #Prediction Translation
        pred[pred==0] = translate[0]+5
        pred[pred==1] = translate[1]+5
        pred[pred==2] = translate[2]+5
        pred[pred==3] = translate[3]+5
        pred[pred==4] = translate[4]+5
        pred[pred==5] = 0
        pred[pred==6] = 1
        pred[pred==7] = 2
        pred[pred==8] = 3
        pred[pred==9] = 4
        
        print("Test Accuracy:",metrics.accuracy_score(pred, self.label))