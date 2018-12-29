import os
import knn
import openFile
import time
import numpy as np
def classNameCut(fileName):
    wholeName=fileName.split('.')[0]
    className=wholeName.split('_')[0]
    return className
# path='C:/Users/user/trainingDigits'
# dirs=os.listdir(path)
# for fileNamr in dirs:
#     print(classNameCut(fileNamr))
def trainingDataSet():
    hwLabels = []
    trainingFileList = os.listdir('C:/Users/user/trainingDigits')           #获取目录内容
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))

    #m维向量的训练集
    for i in range(m):
        fileNameStr = trainingFileList[i]
        hwLabels.append(classNameCut((fileNameStr)))
        trainingMat[i,:] = openFile.img2vector('C:/Users/user/trainingDigits/%s' % fileNameStr)
    return hwLabels,trainingMat

def handwritingTest():
    hwLabels,trainingMat = trainingDataSet()    #构建训练集
    testFileList = os.listdir('C:/Users/user/testDigits')        #获取测试集
    errorCount = 0.0
    mTest = len(testFileList)                   #测试集总样本数
    t1 = time.time()
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = classNameCut(fileNameStr)
        vectorUnderTest = openFile.img2vector('C:/Users/user/testDigits/%s' % fileNameStr)
        classifierResult = knn.classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %s, the real answer is: %s" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of tests is: %d" % mTest  )            #输出测试总样本数
    print ("the total number of errors is: %d" % errorCount)          #输出测试错误样本数
    print ("the total error rate is: %f" % (errorCount/float(mTest)))  #输出错误率
    t2 = time.time()
    print ("Cost time: %.2fmin, %.4fs."%((t2-t1)//60,(t2-t1)%60))     #测试耗时

if __name__ == "__main__":
    handwritingTest()
