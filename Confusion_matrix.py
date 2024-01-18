from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
a=[[966 , 1 ,  6 ,  4 ,  1 ,  0 ,  0 ,  1 , 19 ,  2],
 [  0 , 982 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  1 , 17],
 [ 36 ,  0, 918 , 10 , 16 ,  5 , 13 ,  0 ,  1 ,  1],
 [  3 ,  1 , 12 , 887 ,  6 , 67 , 15 ,  5 ,  4 ,  0],
 [  1 ,  0  , 9 , 10 ,967 ,  4  , 6 ,  3   ,0 ,  0],
 [  1  , 0 , 10 , 57   ,9 ,913 ,  2 ,  7   ,0  , 1],
 [  6  , 1  , 3 ,  3  , 3  , 1, 983 ,  0   ,0  , 0],
 [  3  , 0  , 1 ,  3 , 10   ,9 ,  0 ,974  , 0  , 0],
 [  7  , 6  , 3 ,  0 ,  1   ,0 ,  0 ,  1 ,978  , 4],
 [  2 , 14 ,  2 ,  4 ,  0   ,0 ,  0 ,  0 ,  4 ,974],]


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
 plt.imshow(cm, interpolation='nearest', cmap=cmap)
 plt.title(title)
 plt.colorbar()
 xlocations = np.array(range(len(labels)))
 plt.xticks(xlocations, labels, rotation=90)
 plt.yticks(xlocations, labels)
 plt.ylabel('True label')
 plt.xlabel('Predicted label')


if __name__ == '__main__':
    plt.show()