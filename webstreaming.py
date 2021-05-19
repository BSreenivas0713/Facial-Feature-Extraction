import imutils
from imutils.video import VideoStream
from imutils import face_utils
from flask import Response
from flask import Flask
from flask import render_template
import argparse
import time
import cv2
import dlib
import csv

outputFrame = None
app = Flask(__name__)

vs = cv2.VideoCapture(0)
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def generate():
    while True:
        ret, frame = vs.read()
        frame = cv2.flip(frame, 1) #flip frame horizontally, since on website, it was originally backwards
        frame = imutils.resize(frame, width=400)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            print(shape.tolist()) #these are the 68 landmarks per frame
            print()

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
# release the video stream pointer
vs.release()
cv2.destroyAllWindows()