#EMBEDDER
import torch
import numpy as np
from sklearn import svm
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

# For onboard computation, comment out all but return to disable any optimsations
def optimisation(embed):
    embed = quantization(embed)
    embed = pruning(embed)
    embed = fusing(embed)
    return embed

# Converts embed from float to int to improve performance
def quantization(embed):
    return (np.array(embed, dtype='int'))

def pruning(embed):
    THRESHOLD_VALUE = 100
    return embed
    # return (np.where(embed > THRESHOLD_VALUE, 255, 0))

def fusing(embed):
    return embed

guessFile = "EarPinch2.wav" # Change back to client_audio.wav
#import files
pinchFile = "EarPinch2.wav"
strokeFile = "EarStroke2.wav"
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

# Optimisation (1/3)
numpyPinchEmbed = optimisation(numpyPinchEmbed)
numpyBackgroundEmbed = optimisation(numpyBackgroundEmbed)
numpyStrokeEmbed = optimisation(numpyStrokeEmbed)

fullEmbed = np.concatenate((numpyPinchEmbed,numpyBackgroundEmbed,numpyStrokeEmbed), axis=0)

# Optimisation (2/3)
fullEmbed = optimisation(fullEmbed)

#CLASSIFICATION
#define the target and target names
target = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
numpyTarget = np.array(target)
target_names = ['Pinch','Nothing','Stroke']

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

# Optimisation (3/3)
numpyTestPinchEmbed = optimisation(numpyTestPinchEmbed)
numpyTestStrokeEmbed = optimisation(numpyTestStrokeEmbed)

# print(len(numpyTestEmbed))

# testValue = (numpyTestEmbed[15])
# testValue = testValue.reshape(1,-1)

# predict the result
# testResult = clf.predict(testValue)

###########################################################
overallPinchTest = 0
overallNothingTest = 0
overallStrokeTest = 0
for item in numpyTestPinchEmbed:
    testValue = (item)
    testValue = testValue.reshape(1,-1)

    testResult = clf.predict(testValue)
    overallPinchTest = overallPinchTest + testResult[0]
overallPinchTest = overallPinchTest / len(numpyTestPinchEmbed)
print("Closer to 0 = Pinch")
print("Closer to 1 = Nothing")
print("Closer to 2 = Stroke")
print("The average value for the pinch test data is:")
print(overallPinchTest)

for item in numpyBackgroundEmbed:
    testValue = (item)
    testValue = testValue.reshape(1,-1)

    testResult = clf.predict(testValue)
    overallNothingTest = overallNothingTest + testResult[0]
overallNothingTest = overallNothingTest / len(numpyBackgroundEmbed)
print("The average value for the Background noise test data is:")
print(overallNothingTest)

for item in numpyTestStrokeEmbed:
    testValue = (item)
    testValue = testValue.reshape(1,-1)

    testResult = clf.predict(testValue)
    overallStrokeTest = overallStrokeTest + testResult[0]
overallStrokeTest = overallStrokeTest / len(numpyTestStrokeEmbed)
print("The average value for the stroke test data is:")
print(overallStrokeTest)
print(target_names[testResult[0]])

######################################################################
# guessEmbed = model.forward(guessFile)
# numpyGuessEmbed = [ item.detach().numpy() for item in guessEmbed]

# numpyGuessEmbed = optimisation(numpyGuessEmbed)

# overallGuess = 0
# for item in numpyGuessEmbed:
#     testValue = (item)
#     testValue = testValue.reshape(1,-1)

#     testResult = clf.predict(testValue)
#     overallGuess = overallGuess + testResult[0]
# overallGuess = overallGuess / len(numpyGuessEmbed)
# if (overallGuess < 0.5) and (overallGuess > 0):
#     print("STROKE")
# elif (overallGuess >= 0.5) and (overallGuess <= 1.5):
#     print("NOTHING")
# elif (overallGuess >= 1.5) and (overallGuess <= 2):
#     print("PINCH")
# else:
#     print("error")
