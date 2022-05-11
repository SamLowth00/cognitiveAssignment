import os

#EMBEDDER
import torch
import numpy as np
import time
from sklearn import svm
model = torch.hub.load('torchvggish', 'vggish', source = 'local')
model.eval()

#import files
pinchFile = "pinch5mintraining.wav"
strokeFile = "stroke5mintraining.wav"
backgroundFile = "BackgroundNoise.wav"

#embed the file
pinchEmbed = model.forward(pinchFile)
strokeEmbed = model.forward(strokeFile)
backgroundEmbed = model.forward(backgroundFile)


#np conversion
numpyPinchEmbed = [ item.detach().numpy() for item in pinchEmbed]
numpyBackgroundEmbed = [ item.detach().numpy() for item in backgroundEmbed]
numpyStrokeEmbed = [ item.detach().numpy() for item in strokeEmbed]
fullEmbed = np.concatenate((numpyBackgroundEmbed,numpyStrokeEmbed,numpyPinchEmbed), axis=0)
#CLASSIFICATION
target = []
target_names = ['Pinch','Nothing','Stroke']
for item in numpyBackgroundEmbed:
    target.append(0)
for item in numpyStrokeEmbed:
    target.append(1)
for item in numpyPinchEmbed:
    target.append(2)
numpyTarget = np.array(target)
#define classifier and train the data
clf = svm.SVC(kernel='linear')
clf.fit(fullEmbed,numpyTarget)

while True:
	os.system('python3 client_audio.py record')
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
	if (overallGuess < 0.5) and (overallGuess >= 0):
	    print("NOTHING")
	elif (overallGuess >= 0.5) and (overallGuess <= 1):
	    print("STROKE")
	elif (overallGuess >= 1) and (overallGuess <= 2):
	    print("PINCH")
	else:
	    print("error")

#from client_audio_edit import client

#miro = client()

#miro.record("record")
#os.system('python3 VGGishEmbed.py')
