import rospy
import os
import miro2 as miro
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, UInt16MultiArray, UInt8MultiArray, UInt16, Int16MultiArray, String

#Noise reduction imports
import noisereduce as nr
import soundfile as sf
		
#EMBEDDER
import torch
import numpy as np
import time
from sklearn import svm
model = torch.hub.load('torchvggish', 'vggish', source = 'local')
model.eval()

front_left, mid_left, rear_left, front_right, mid_right, rear_right = range(6)
illum = UInt32MultiArray()
illum.data = [0xFFFFFFF0, 0xFFFFFFF0, 0xFFFFFFF0, 0xFFFFFFF0, 0xFFFFFFF0, 0xFFFFFFF0]
topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
pub_illum = rospy.Publisher(topic_base_name + "/control/illum", UInt32MultiArray, queue_size=0)
rospy.init_node("client_illum")

value = 0xFFFF0000
for x in range(6):
	illum.data[x] = value

#Noise Reduction Function
def noise_reduction(dataWav, backgroundWav, outputFile, stationary):
    data, rate = sf.read(dataWav)
    noise_data, noise_rate = sf.read(backgroundWav)
    reduced_noise = np.empty([len(data),0])
    for i in range (0,(data.shape[1])):
        if (stationary):
            index_reduced_noise = nr.reduce_noise(y = (data[:,i]), sr=rate, y_noise = (noise_data[:,i]), n_std_thresh_stationary=1.5,stationary=True)
        else:
            index_reduced_noise = nr.reduce_noise(y = (data[:,i]), sr=rate, thresh_n_mult_nonstationary=2,y_noise = noise_data,stationary=False)
        reduced_noise = np.insert(reduced_noise,i,index_reduced_noise, axis=1)
    sf.write(outputFile, reduced_noise, rate)


#import files
pinchFile = "pinch5mintrainingNR.wav"
strokeFile = "stroke5mintrainingNR.wav"
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
	noise_reduction("client_audio.wav", "BackgroundNoise.wav","client_audio_reduced.wav", True)
	guessFile = "client_audio_reduced.wav"
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
	if (overallGuess < 1) and (overallGuess >= 0):
	    print("NOTHING")
	    value = 0xFF0000FF
	elif (overallGuess >= 1) and (overallGuess <= 1.5):
	    print("STROKE")
	    value = 0xFF00FF00
	elif (overallGuess > 1.5) and (overallGuess <= 2):
	    print("PINCH")
	    value = 0xFFFF0000
	else:
	    print("error")
	
	for x in range(6):
		illum.data[x] = value
		
	pub_illum.publish(illum)

#from client_audio_edit import client

#miro = client()

#miro.record("record")
#os.system('python3 VGGishEmbed.py')
