import numpy as np
import mnist_load_show as mnist
from sklearn.metrics import confusion_matrix
"""
============================================
DO NOT FORGET TO INCLUDE YOUR STUDENT ID
============================================
"""
student_ID = ''


X, y = mnist.read_mnist_training_data()


def my_info():
    """
    :return: DO NOT FORGET to include your student ID as a string, this function is used to evaluate your code and results
    """
    return student_ID


def one_vs_all():
    """
    Implement the the multi label classifier using one_vs_all paradigm and return the confusion matrix
    :return: the confusion matrix regarding the result obtained using the classifier
    """
    one_vs_all_conf_matrix = ''
    return one_vs_all_conf_matrix


def all_vs_all():
    """
    Implement the multi label classifier based on the all_vs_all paradigm and return the confusion matrix
    :return: the confusing matrix obtained regarding the result obtained using teh classifier
    """
    all_vs_all_conf_matrix = ''
    return all_vs_all_conf_matrix




def main():
    """
    DO NOT TOUCH THIS FUNCTION. IT IS USED FOR COMPUTER EVALUATION OF YOUR CODE
    """
    results = my_info() + '\t\t'
    results += np.array_str(np.diagonal(one_vs_all())) + '\t\t'
    results += np.array_str(np.diagonal(all_vs_all()))
    print results + '\t\t'

if __name__ == '__main__':
    main()