#EMBEDDER
import torch
import numpy as np
from sklearn import svm
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

#import files
pinchFile = "EarPinch2.wav"
strokeFile = "EarStroke2.wav"

#embed the file
pinchEmbed = model.forward(pinchFile)
strokeEmbed = model.forward(strokeFile)

#np conversion
numpyPinchEmbed = [ item.detach().numpy() for item in pinchEmbed]
numpyStrokeEmbed = [ item.detach().numpy() for item in strokeEmbed]
fullEmbed = np.concatenate((numpyPinchEmbed,numpyStrokeEmbed), axis=0)

#CLASSIFICATION
#define the target and target names
target = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
numpyTarget = np.array(target)
target_names = ['Pinch','Stroke']

#define classifier and train the data
clf = svm.SVC(kernel='linear')
clf.fit(fullEmbed,numpyTarget)

#set up a test 1 second
testPinch = "EarPinchTestData2.wav"
testPinchEmbed = model.forward(testPinch)
numpyTestPinchEmbed = [ item.detach().numpy() for item in testPinchEmbed]

testStroke = "EarStrokeTestData2.wav"
testStrokeEmbed = model.forward(testStroke)
numpyTestStrokeEmbed = [ item.detach().numpy() for item in testStrokeEmbed]

#print(len(numpyTestEmbed))

#testValue = (numpyTestEmbed[15])
#testValue = testValue.reshape(1,-1)

#predict the result
#testResult = clf.predict(testValue)
overallPinchTest = 0
overallStrokeTest = 0
for item in numpyTestPinchEmbed:
    testValue = (item)
    testValue = testValue.reshape(1,-1)

    testResult = clf.predict(testValue)
    overallPinchTest = overallPinchTest + testResult[0]
overallPinchTest = overallPinchTest / len(numpyTestPinchEmbed)
print("Closer to 0 = Pinch")
print("Closer to 1 = Stroke")
print("The average value for the pinch test data is:")
print(overallPinchTest)

for item in numpyTestStrokeEmbed:
    testValue = (item)
    testValue = testValue.reshape(1,-1)

    testResult = clf.predict(testValue)
    overallStrokeTest = overallStrokeTest + testResult[0]
overallStrokeTest = overallStrokeTest / len(numpyTestStrokeEmbed)
print("The average value for the stroke test data is:")
print(overallStrokeTest)
#print(target_names[testResult[0]])
