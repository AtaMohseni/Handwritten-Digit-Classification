# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:13:12 2023

@author: ATA
"""

def data_loader():
    """ function that read train and test text data and store them on a list"""
    try:
        train3 = open('./new_train3.txt')
        train5 = open('./new_train5.txt')
        test3 = open('./new_test3.txt')
        test5 = open('./new_test5.txt')
    except:
        print ('one or more the files do not exist')
        return None
    
    datatrain = []
    datatest = []
    for line in train3:
        datatrain.append([list(map(int,list (line.split()) )),0])
    for line in train5:
        datatrain.append([list(map(int,list (line.split()) )),1])
    for line in test3:
        datatest.append([list(map(int,list (line.split()) )),0])
    for line in test5:
        datatest.append([list(map(int,list (line.split()) )),1])
    return datatrain , datatest    


if __name__ == "__main__" :
    
    datatrain , datatest = data_loader()
   
    