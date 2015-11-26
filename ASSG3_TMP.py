import numpy as np
import mnist_load_show as mnist
import time
'''
use pdis in order to find the the distance
'''

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix


"""
============================================
DO NOT FORGET TO INCLUDE YOUR STUDENT ID
============================================
"""
<<<<<<< HEAD
student_number = 'sample_solution'
=======
student_ID = ''
>>>>>>> 622382c2ed459a743b28d10dffc1e837df0911a1


X, y = mnist.read_mnist_training_data(N=5000)

X_train = X[0:2500]
X_test = X[2500:5000]
y_train = y[0:2500]
y_test = y[2500:5000]

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)


def sanity_check():
    indices = np.random.choice(5000, 100, replace=False)
    print bmatrix(y[indices].reshape(10,10))
    mnist.visualize(X[indices])

def my_info():
    """
    :return: DO NOT FORGET to include your student ID as a string, this function is used to evaluate your code and results
    """
    return student_ID


def simple_EC_classifier():
    """
    Implement the classifier using KNN and return the confusion matrix
    :return: the confusion matrix regarding the result obtained using knn method
    """
    prototypes = np.array([np.mean(X_train[y_train == i], 0) for i in xrange(10)])
    distance = cdist(X_test, prototypes)
    predicted_digits = np.argmin(distance, 1)
    simple_EC_conf_matrix = confusion_matrix(y_test, predicted_digits)
    return simple_EC_conf_matrix


def KNN():
    """
    Implement the classifier based on the Euclidean distance
    :return: the confusing matrix obtained regarding the result obtained using simple Euclidean distance method
    """
    
    distances = cdist(X_test, X_train)
    predicted_digits = y_train[np.argmin(distances, 1)]
    knn_conf_matrix = confusion_matrix(y_test, predicted_digits)
    return knn_conf_matrix




def main():
    """
    DO NOT TOUCH THIS FUNCTION. IT IS USED FOR COMPUTER EVALUATION OF YOUR CODE
    """
    print bmatrix(KNN())

    '''start = time.time()
    simple_EC_conf_matrix = simple_EC_classifier()
    simple_EC_classifier_precision = np.sum(np.diagonal(simple_EC_conf_matrix)) / 2500.0
    simple_EC_classifier_error = 1 - simple_EC_classifier_precision

    knn_conf_matrix = KNN()
    knn_classifier_precision = np.sum(np.diagonal(knn_conf_matrix))/ 2500.0
    knn_classifier_error = 1 - knn_classifier_precision
    end = time.time()
    print np.array_str(np.diagonal(simple_EC_conf_matrix)) + '\t\t'
    print np.array_str(np.diagonal(knn_conf_matrix)) + '\t\t'
    print 'simple_EC precision is ' + str(simple_EC_classifier_precision) + ' and error is: ' + str(simple_EC_classifier_error)
    print 'KNN precision is ' + str(knn_classifier_precision) + ' and error is: ' + str(knn_classifier_error)
    print('elapsed time WITHOUT sanity check and plotting: ' + str(end - start))
    '''


if __name__ == '__main__':
    main()
