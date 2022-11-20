# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import time
import math
import wave
from playsound import playsound


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(2).start()
time.sleep(2.0)
list = np.array(range(3000))

def showimage():
	f = open('test_list_jpg.txt', 'r')
	name_data = f.read().splitlines()
	f.close()
	#for i in name_data:
	#	time.sleep(1.0)
	#img = cv2.imread('label.png')
	f = open('emolist.txt', 'r')
	emodata = f.read().splitlines()
	f.close()
	data_len = len(emodata)
	emo = emodata[data_len-1][2:5]
	print(emodata[data_len-1][2:5])
	if (emo=='awa'):
		folder_path = './aware_jpg/'
	else:
		folder_path = './normal_jpg/'
  
	img_0000  = cv2.imread(folder_path+name_data[0])
	img_0001  = cv2.imread(folder_path+name_data[1])
	img_0002  = cv2.imread(folder_path+name_data[2])
	img_0003  = cv2.imread(folder_path+name_data[3])
	img_0004  = cv2.imread(folder_path+name_data[4])
	img_0005  = cv2.imread(folder_path+name_data[5])
	img_0006  = cv2.imread(folder_path+name_data[6])
	img_0007  = cv2.imread(folder_path+name_data[7])
	img_0008  = cv2.imread(folder_path+name_data[8])
	img_0009  = cv2.imread(folder_path+name_data[9])
	img_0010  = cv2.imread(folder_path+name_data[10])
	img_0011  = cv2.imread(folder_path+name_data[11])
	img_0012  = cv2.imread(folder_path+name_data[12])
	img_0013  = cv2.imread(folder_path+name_data[13])
	img_0014  = cv2.imread(folder_path+name_data[14])
	img_0015  = cv2.imread(folder_path+name_data[15])
	img_0016  = cv2.imread(folder_path+name_data[16])
	img_0017  = cv2.imread(folder_path+name_data[17])
	img_0018  = cv2.imread(folder_path+name_data[18])
	img_0019  = cv2.imread(folder_path+name_data[19])
	img_0020  = cv2.imread(folder_path+name_data[20])
	img_0021  = cv2.imread(folder_path+name_data[21])
	img_0022  = cv2.imread(folder_path+name_data[22])
	img_0023  = cv2.imread(folder_path+name_data[23])
	img_0024  = cv2.imread(folder_path+name_data[24])
	img_0025  = cv2.imread(folder_path+name_data[25])
	img_0026  = cv2.imread(folder_path+name_data[26])
	img_0027  = cv2.imread(folder_path+name_data[27])
	img_0028  = cv2.imread(folder_path+name_data[28])
	img_0029  = cv2.imread(folder_path+name_data[29])
	img_0030  = cv2.imread(folder_path+name_data[30])
	img_0031  = cv2.imread(folder_path+name_data[31])
	img_0032  = cv2.imread(folder_path+name_data[32])
	img_0033  = cv2.imread(folder_path+name_data[33])
	img_0034  = cv2.imread(folder_path+name_data[34])
	img_0035  = cv2.imread(folder_path+name_data[35])
	img_0036  = cv2.imread(folder_path+name_data[36])
	img_0037  = cv2.imread(folder_path+name_data[37])
	img_0038  = cv2.imread(folder_path+name_data[38])
	img_0039  = cv2.imread(folder_path+name_data[39])
	img_0040  = cv2.imread(folder_path+name_data[40])
	img_0041  = cv2.imread(folder_path+name_data[41])
	img_0042  = cv2.imread(folder_path+name_data[42])
	img_0043  = cv2.imread(folder_path+name_data[43])
	img_0044  = cv2.imread(folder_path+name_data[44])
	img_0045  = cv2.imread(folder_path+name_data[45])
	img_0046  = cv2.imread(folder_path+name_data[46])
	img_0047  = cv2.imread(folder_path+name_data[47])
	img_0048  = cv2.imread(folder_path+name_data[48])
	img_0049  = cv2.imread(folder_path+name_data[49])
	img_0050  = cv2.imread(folder_path+name_data[50])
	img_0051  = cv2.imread(folder_path+name_data[51])
	img_0052  = cv2.imread(folder_path+name_data[52])
	img_0053  = cv2.imread(folder_path+name_data[53])
	img_0054  = cv2.imread(folder_path+name_data[54])
	img_0055  = cv2.imread(folder_path+name_data[55])
	img_0056  = cv2.imread(folder_path+name_data[56])
	img_0057  = cv2.imread(folder_path+name_data[57])
	img_0058  = cv2.imread(folder_path+name_data[58])
	img_0059  = cv2.imread(folder_path+name_data[59])
	img_0060  = cv2.imread(folder_path+name_data[60])
	img_0061  = cv2.imread(folder_path+name_data[61])
	img_0062  = cv2.imread(folder_path+name_data[62])
	img_0063  = cv2.imread(folder_path+name_data[63])
	img_0064  = cv2.imread(folder_path+name_data[64])
	img_0065  = cv2.imread(folder_path+name_data[65])
	img_0066  = cv2.imread(folder_path+name_data[66])
	img_0067  = cv2.imread(folder_path+name_data[67])
	img_0068  = cv2.imread(folder_path+name_data[68])
	img_0069  = cv2.imread(folder_path+name_data[69])
	img_0070  = cv2.imread(folder_path+name_data[70])
	img_0071  = cv2.imread(folder_path+name_data[71])
	img_0072  = cv2.imread(folder_path+name_data[72])
	img_0073  = cv2.imread(folder_path+name_data[73])
	img_0074  = cv2.imread(folder_path+name_data[74])
	img_0075  = cv2.imread(folder_path+name_data[75])
	img_0076  = cv2.imread(folder_path+name_data[76])
	img_0077  = cv2.imread(folder_path+name_data[77])
	img_0078  = cv2.imread(folder_path+name_data[78])
	img_0079  = cv2.imread(folder_path+name_data[79])
	img_0080  = cv2.imread(folder_path+name_data[80])
	img_0081  = cv2.imread(folder_path+name_data[81])
	img_0082  = cv2.imread(folder_path+name_data[82])
	img_0083  = cv2.imread(folder_path+name_data[83])
	img_0084  = cv2.imread(folder_path+name_data[84])
	img_0085  = cv2.imread(folder_path+name_data[85])
	img_0086  = cv2.imread(folder_path+name_data[86])
	img_0087  = cv2.imread(folder_path+name_data[87])
	img_0088  = cv2.imread(folder_path+name_data[88])
	img_0089  = cv2.imread(folder_path+name_data[89])
	img_0090  = cv2.imread(folder_path+name_data[90])
	img_0091  = cv2.imread(folder_path+name_data[91])
	img_0092  = cv2.imread(folder_path+name_data[92])
	img_0093  = cv2.imread(folder_path+name_data[93])
	img_0094  = cv2.imread(folder_path+name_data[94])
	img_0095  = cv2.imread(folder_path+name_data[95])
	img_0096  = cv2.imread(folder_path+name_data[96])
	img_0097  = cv2.imread(folder_path+name_data[97])
	img_0098  = cv2.imread(folder_path+name_data[98])
	img_0099  = cv2.imread(folder_path+name_data[99])
	return img_0000, img_0001, img_0002, img_0003, img_0004, img_0005, img_0006, img_0007, img_0008, img_0009, img_0010, img_0011, img_0012, img_0013, img_0014, img_0015, img_0016, img_0017, img_0018, img_0019,img_0020, img_0021, img_0022, img_0023, img_0024, img_0025, img_0026, img_0027, img_0028, img_0029,img_0030, img_0031, img_0032, img_0033, img_0034, img_0035, img_0036, img_0037, img_0038, img_0039,img_0040, img_0041, img_0042, img_0043, img_0044, img_0045, img_0046, img_0047, img_0048, img_0049,img_0050, img_0051, img_0052, img_0053, img_0054, img_0055, img_0056, img_0057, img_0058, img_0059,img_0060, img_0061, img_0062, img_0063, img_0064, img_0065, img_0066, img_0067, img_0068, img_0069,img_0070, img_0071, img_0072, img_0073, img_0074, img_0075, img_0076, img_0077, img_0078, img_0079,img_0080, img_0081, img_0082, img_0083, img_0084, img_0085, img_0086, img_0087, img_0088, img_0089,img_0090, img_0091, img_0092, img_0093, img_0094, img_0095, img_0096, img_0097, img_0098, img_0099
def anime_face_func(img, rect):
	(x1, y1, x2, y2) = rect
	w = x2-x1
	h = int((y2-y1)/2)
	ut = math.floor(time.time()*10)
	print(str(ut)[-2:])
	if(str(ut)[-2:] == '00'):
		img_face = cv2.resize(showimage()[0], (w, h))
	elif (str(ut)[-2:] == '01'):
		img_face = cv2.resize(showimage()[1], (w, h))
	elif (str(ut)[-2:] == '02'):
		img_face = cv2.resize(showimage()[2], (w, h))
	elif (str(ut)[-2:] == '03'):
		img_face = cv2.resize(showimage()[3], (w, h))
	elif (str(ut)[-2:] == '04'):
		img_face = cv2.resize(showimage()[4], (w, h))
	elif (str(ut)[-2:] == '05'):
		img_face = cv2.resize(showimage()[5], (w, h))
	elif (str(ut)[-2:] == '06'):
		img_face = cv2.resize(showimage()[6], (w, h))
	elif (str(ut)[-2:] == '07'):
		img_face = cv2.resize(showimage()[7], (w, h))
	elif (str(ut)[-2:] == '08'):
		img_face = cv2.resize(showimage()[8], (w, h))
	elif (str(ut)[-2:] == '09'):
		img_face = cv2.resize(showimage()[9], (w, h))
	elif (str(ut)[-2:] == '10'):
		img_face = cv2.resize(showimage()[10], (w, h))
	elif (str(ut)[-2:] == '11'):
		img_face = cv2.resize(showimage()[11], (w, h))
	elif (str(ut)[-2:] == '12'):
		img_face = cv2.resize(showimage()[12], (w, h))
	elif (str(ut)[-2:] == '13'):
		img_face = cv2.resize(showimage()[13], (w, h))
	elif (str(ut)[-2:] == '14'):
		img_face = cv2.resize(showimage()[14], (w, h))
	elif (str(ut)[-2:] == '15'):
		img_face = cv2.resize(showimage()[15], (w, h))
	elif (str(ut)[-2:] == '16'):
		img_face = cv2.resize(showimage()[16], (w, h))
	elif (str(ut)[-2:] == '17'):
		img_face = cv2.resize(showimage()[17], (w, h))
	elif (str(ut)[-2:] == '18'):
		img_face = cv2.resize(showimage()[18], (w, h))
	elif (str(ut)[-2:] == '19'):
		img_face = cv2.resize(showimage()[19], (w, h))
	elif (str(ut)[-2:] == '20'):
		img_face = cv2.resize(showimage()[20], (w, h))
	elif (str(ut)[-2:] == '21'):
		img_face = cv2.resize(showimage()[21], (w, h))
	elif (str(ut)[-2:] == '22'):
		img_face = cv2.resize(showimage()[22], (w, h))
	elif (str(ut)[-2:] == '23'):
		img_face = cv2.resize(showimage()[23], (w, h))
	elif (str(ut)[-2:] == '24'):
		img_face = cv2.resize(showimage()[24], (w, h))
	elif (str(ut)[-2:] == '25'):
		img_face = cv2.resize(showimage()[25], (w, h))
	elif (str(ut)[-2:] == '26'):
		img_face = cv2.resize(showimage()[26], (w, h))
	elif (str(ut)[-2:] == '27'):
		img_face = cv2.resize(showimage()[27], (w, h))
	elif (str(ut)[-2:] == '28'):
		img_face = cv2.resize(showimage()[28], (w, h))
	elif (str(ut)[-2:] == '29'):
		img_face = cv2.resize(showimage()[29], (w, h))
	elif (str(ut)[-2:] == '30'):
		img_face = cv2.resize(showimage()[30], (w, h))
	elif (str(ut)[-2:] == '31'):
		img_face = cv2.resize(showimage()[31], (w, h))
	elif (str(ut)[-2:] == '32'):
		img_face = cv2.resize(showimage()[32], (w, h))
	elif (str(ut)[-2:] == '33'):
		img_face = cv2.resize(showimage()[33], (w, h))
	elif (str(ut)[-2:] == '34'):
		img_face = cv2.resize(showimage()[34], (w, h))
	elif (str(ut)[-2:] == '35'):
		img_face = cv2.resize(showimage()[35], (w, h))
	elif (str(ut)[-2:] == '36'):
		img_face = cv2.resize(showimage()[36], (w, h))
	elif (str(ut)[-2:] == '37'):
		img_face = cv2.resize(showimage()[37], (w, h))
	elif (str(ut)[-2:] == '38'):
		img_face = cv2.resize(showimage()[38], (w, h))
	elif (str(ut)[-2:] == '39'):
		img_face = cv2.resize(showimage()[39], (w, h))
	elif (str(ut)[-2:] == '40'):
		img_face = cv2.resize(showimage()[40], (w, h))
	elif (str(ut)[-2:] == '41'):
		img_face = cv2.resize(showimage()[41], (w, h))
	elif (str(ut)[-2:] == '42'):
		img_face = cv2.resize(showimage()[42], (w, h))
	elif (str(ut)[-2:] == '43'):
		img_face = cv2.resize(showimage()[43], (w, h))
	elif (str(ut)[-2:] == '44'):
		img_face = cv2.resize(showimage()[44], (w, h))
	elif (str(ut)[-2:] == '45'):
		img_face = cv2.resize(showimage()[45], (w, h))
	elif (str(ut)[-2:] == '46'):
		img_face = cv2.resize(showimage()[46], (w, h))
	elif (str(ut)[-2:] == '47'):
		img_face = cv2.resize(showimage()[47], (w, h))
	elif (str(ut)[-2:] == '48'):
		img_face = cv2.resize(showimage()[48], (w, h))
	elif (str(ut)[-2:] == '49'):
		img_face = cv2.resize(showimage()[49], (w, h))
	elif (str(ut)[-2:] == '50'):
		img_face = cv2.resize(showimage()[50], (w, h))
	elif (str(ut)[-2:] == '51'):
		img_face = cv2.resize(showimage()[51], (w, h))
	elif (str(ut)[-2:] == '52'):
		img_face = cv2.resize(showimage()[52], (w, h))
	elif (str(ut)[-2:] == '53'):
		img_face = cv2.resize(showimage()[53], (w, h))
	elif (str(ut)[-2:] == '54'):
		img_face = cv2.resize(showimage()[54], (w, h))
	elif (str(ut)[-2:] == '55'):
		img_face = cv2.resize(showimage()[55], (w, h))
	elif (str(ut)[-2:] == '56'):
		img_face = cv2.resize(showimage()[56], (w, h))
	elif (str(ut)[-2:] == '57'):
		img_face = cv2.resize(showimage()[57], (w, h))
	elif (str(ut)[-2:] == '58'):
		img_face = cv2.resize(showimage()[58], (w, h))
	elif (str(ut)[-2:] == '59'):
		img_face = cv2.resize(showimage()[59], (w, h))
	elif (str(ut)[-2:] == '60'):
		img_face = cv2.resize(showimage()[60], (w, h))
	elif (str(ut)[-2:] == '61'):
		img_face = cv2.resize(showimage()[61], (w, h))
	elif (str(ut)[-2:] == '62'):
		img_face = cv2.resize(showimage()[62], (w, h))
	elif (str(ut)[-2:] == '63'):
		img_face = cv2.resize(showimage()[63], (w, h))
	elif (str(ut)[-2:] == '64'):
		img_face = cv2.resize(showimage()[64], (w, h))
	elif (str(ut)[-2:] == '65'):
		img_face = cv2.resize(showimage()[65], (w, h))
	elif (str(ut)[-2:] == '66'):
		img_face = cv2.resize(showimage()[66], (w, h))
	elif (str(ut)[-2:] == '67'):
		img_face = cv2.resize(showimage()[67], (w, h))
	elif (str(ut)[-2:] == '68'):
		img_face = cv2.resize(showimage()[68], (w, h))
	elif (str(ut)[-2:] == '69'):
		img_face = cv2.resize(showimage()[69], (w, h))
	elif (str(ut)[-2:] == '70'):
		img_face = cv2.resize(showimage()[70], (w, h))
	elif (str(ut)[-2:] == '71'):
		img_face = cv2.resize(showimage()[71], (w, h))
	elif (str(ut)[-2:] == '72'):
		img_face = cv2.resize(showimage()[72], (w, h))
	elif (str(ut)[-2:] == '73'):
		img_face = cv2.resize(showimage()[73], (w, h))
	elif (str(ut)[-2:] == '74'):
		img_face = cv2.resize(showimage()[74], (w, h))
	elif (str(ut)[-2:] == '75'):
		img_face = cv2.resize(showimage()[75], (w, h))
	elif (str(ut)[-2:] == '76'):
		img_face = cv2.resize(showimage()[76], (w, h))
	elif (str(ut)[-2:] == '77'):
		img_face = cv2.resize(showimage()[77], (w, h))
	elif (str(ut)[-2:] == '78'):
		img_face = cv2.resize(showimage()[78], (w, h))
	elif (str(ut)[-2:] == '79'):
		img_face = cv2.resize(showimage()[79], (w, h))
	elif (str(ut)[-2:] == '80'):
		img_face = cv2.resize(showimage()[80], (w, h))
	elif (str(ut)[-2:] == '81'):
		img_face = cv2.resize(showimage()[81], (w, h))
	elif (str(ut)[-2:] == '82'):
		img_face = cv2.resize(showimage()[82], (w, h))
	elif (str(ut)[-2:] == '83'):
		img_face = cv2.resize(showimage()[83], (w, h))
	elif (str(ut)[-2:] == '84'):
		img_face = cv2.resize(showimage()[84], (w, h))
	elif (str(ut)[-2:] == '85'):
		img_face = cv2.resize(showimage()[85], (w, h))
	elif (str(ut)[-2:] == '86'):
		img_face = cv2.resize(showimage()[86], (w, h))
	elif (str(ut)[-2:] == '87'):
		img_face = cv2.resize(showimage()[87], (w, h))
	elif (str(ut)[-2:] == '88'):
		img_face = cv2.resize(showimage()[88], (w, h))
	elif (str(ut)[-2:] == '89'):
		img_face = cv2.resize(showimage()[89], (w, h))
	elif (str(ut)[-2:] == '90'):
		img_face = cv2.resize(showimage()[90], (w, h))
	elif (str(ut)[-2:] == '91'):
		img_face = cv2.resize(showimage()[91], (w, h))
	elif (str(ut)[-2:] == '92'):
		img_face = cv2.resize(showimage()[92], (w, h))
	elif (str(ut)[-2:] == '93'):
		img_face = cv2.resize(showimage()[93], (w, h))
	elif (str(ut)[-2:] == '94'):
		img_face = cv2.resize(showimage()[94], (w, h))
	elif (str(ut)[-2:] == '95'):
		img_face = cv2.resize(showimage()[95], (w, h))
	elif (str(ut)[-2:] == '96'):
		img_face = cv2.resize(showimage()[96], (w, h))
	elif (str(ut)[-2:] == '97'):
		img_face = cv2.resize(showimage()[97], (w, h))
	elif (str(ut)[-2:] == '98'):
		img_face = cv2.resize(showimage()[98], (w, h))
	elif (str(ut)[-2:] == '99'):
		img_face = cv2.resize(showimage()[99], (w, h))
	img2 = img.copy()
	img2[int((y2+y1+1)/2):y2, x1:x2] = img_face
	return img2




# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		#cv2.putText(frame, label, (startX, startY - 10),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		#cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		print(startX, startY, endX, endY)
		if (color == (0, 255, 0)):
			frame = anime_face_func(frame, (startX, startY, endX, endY))
		#cv2.VideoWriter('out1.m4v', cv2.VideoWriter_fourcc).write(img)

	# show the output frame
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
