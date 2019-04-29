from preprocessing import read_data, read_label
from kmeans import kmeans
from ensemble import random_forest
import sys

data_path = '/home/yzhuang1211/Desktop/466/Tumor-Classification/data/data.csv'
label_path = '/home/yzhuang1211/Desktop/466/Tumor-Classification/data/labels.csv'


data = read_data(data_path, int(sys.argv[2]))
label = read_label(label_path)
print('Reading completed')

#'sup' = rf, 'unsup' = kmeans
method = sys.argv[1]

if(method == 'sup'):
    rf = random_forest(data,label,50)
    rf.fit()
    rf.predict()
    
elif(method == 'unsup'):
    km = kmeans(data,label)
    km.predict()
    
else:
    print('{method} is not supported'.format(method = method))
