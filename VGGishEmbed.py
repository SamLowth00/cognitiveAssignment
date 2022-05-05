print("penis")
#EMBEDDER
import torch
import numpy as np
import time
from sklearn import svm
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

guessFile = "client_audio.wav"
#import files
pinchFile = "pinch5mintraining.wav"
strokeFile = "stroke5mintraining.wav"
backgroundFile = "BackgroundNoise.wav"

#embed the file
pinchEmbed = model.forward(pinchFile)
strokeEmbed = model.forward(strokeFile)
backgroundEmbed = model.forward(backgroundFile)

guessEmbed = model.forward(guessFile)

#np conversion
numpyPinchEmbed = [ item.detach().numpy() for item in pinchEmbed]
numpyBackgroundEmbed = [ item.detach().numpy() for item in backgroundEmbed]
numpyStrokeEmbed = [ item.detach().numpy() for item in strokeEmbed]
fullEmbed = np.concatenate((numpyBackgroundEmbed,numpyPinchEmbed,numpyStrokeEmbed), axis=0)
#CLASSIFICATION
#define the target and target names
#target = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
target = []
numpyTarget = np.array(target)
target_names = ['Pinch','Nothing','Stroke']
for item in numpyBackgroundEmbed:
    target.append(0)
for item in numpyStrokeEmbed:
    target.append(1)
for item in numpyPinchEmbed
    target.append(2)
numpyTarget = np.array(target)    
#define classifier and train the data
clf = svm.SVC(kernel='linear')
clf.fit(fullEmbed,numpyTarget)

#set up a test 1 second
# testPinch = "EarPinchTestData2.wav"
# testPinchEmbed = model.forward(testPinch)
# numpyTestPinchEmbed = [ item.detach().numpy() for item in testPinchEmbed]
#
# testStroke = "EarStrokeTestData2.wav"
# testStrokeEmbed = model.forward(testStroke)
# numpyTestStrokeEmbed = [ item.detach().numpy() for item in testStrokeEmbed]

#print(len(numpyTestEmbed))

#testValue = (numpyTestEmbed[15])
#testValue = testValue.reshape(1,-1)

#predict the result
#testResult = clf.predict(testValue)

###########################################################
# overallPinchTest = 0
# overallNothingTest = 0
# overallStrokeTest = 0
# for item in numpyTestPinchEmbed:
#     testValue = (item)
#     testValue = testValue.reshape(1,-1)
#
#     testResult = clf.predict(testValue)
#     overallPinchTest = overallPinchTest + testResult[0]
# overallPinchTest = overallPinchTest / len(numpyTestPinchEmbed)
# print("Closer to 0 = Pinch")
# print("Closer to 1 = Nothing")
# print("Closer to 2 = Stroke")
# print("The average value for the pinch test data is:")
# print(overallPinchTest)
#
# for item in numpyBackgroundEmbed:
#     testValue = (item)
#     testValue = testValue.reshape(1,-1)
#
#     testResult = clf.predict(testValue)
#     overallNothingTest = overallNothingTest + testResult[0]
# overallNothingTest = overallNothingTest / len(numpyBackgroundEmbed)
# print("The average value for the Background noise test data is:")
# print(overallNothingTest)
#
# for item in numpyTestStrokeEmbed:
#     testValue = (item)
#     testValue = testValue.reshape(1,-1)
#
#     testResult = clf.predict(testValue)
#     overallStrokeTest = overallStrokeTest + testResult[0]
# overallStrokeTest = overallStrokeTest / len(numpyTestStrokeEmbed)
# print("The average value for the stroke test data is:")
# print(overallStrokeTest)
#print(target_names[testResult[0]])

######################################################################
guessFile = "client_audio.wav"
guessEmbed = model.forward(guessFile)
numpyGuessEmbed = [ item.detach().numpy() for item in guessEmbed]
overallGuess = 0
for item in numpyGuessEmbed:
    testValue = (item)
    testValue = testValue.reshape(1,-1)

    testResult = clf.predict(testValue)
    print(testResult[0])
    overallGuess = overallGuess + testResult[0]
overallGuess = overallGuess / len(numpyGuessEmbed)
print("guess value is")
print(overallGuess)
if (overallGuess < 0.5) and (overallGuess > 0):
    print("NOTHING")
elif (overallGuess >= 0.5) and (overallGuess <= 1):
    print("STROKE")
elif (overallGuess >= 1) and (overallGuess <= 2):
    print("PINCH")
else:
    print("error")
