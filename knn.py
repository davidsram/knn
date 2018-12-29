from numpy import *
import operator

# def create_data_set():
#     group=array([1.1,1.0],[1,1],[0,0],[0,0.1])
#     labels=['A','A','B','B']
#     return group,labels
# group=array([[1.1,1.0,1],[1.0,1.0,1],[0.09,0.09,1],[0,0,1],[0,0.1,1],[2,1,1]])
# labels=['A','A','B','B','B','C']
def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    # print('diff:',diffMat)
    sqDiffMat=diffMat**2
    # print('sqdiffmat:',sqDiffMat)
    sqDistances=sqDiffMat.sum(axis=1)

    distances=sqDistances**0.5
    # print('distance:',distances)
    sortedDistances=distances.argsort()
    # print("sorted;",sortedDistances[0:20])
#     选择距离最小
    classCount={}
    for i in range(k):
        votelabel=labels[sortedDistances[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    #排序

    # print(sortedClassCount)
    return sortedClassCount[0][0]


