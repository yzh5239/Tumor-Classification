import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics




def read_data(data_path, pca = 100):
    data = []
    count = 0
    with open(data_path, 'r') as f:
        #Read data from csv to np array
        for row in f:
            count+=1
            if(count > 1):
                row = row.replace('\n','')
                row = row.split(',')
                data.append(row[1:])
        data = np.array(data).astype('float32')
        
        #PCA to reduce dimension
        pca = PCA(n_components=pca)
        pca.fit(data)
        data = pca.fit_transform(data)
        
    return data 

def read_label(label_path):
    label_map = {}
    label_map['PRAD'] = 0
    label_map['KIRC'] = 1
    label_map['LUAD'] = 2
    label_map['BRCA'] = 3
    label_map['COAD'] = 4
    label = []
    count = 0
    with open(label_path, 'r') as f:
        #Read data from csv to np array
        for row in f:
            count+=1
            if(count > 1):
                tmp = row.split(',')[1].replace('\n','')
                label.append(label_map[tmp])
            
    label = np.array(label) 
    return label



