# usage
# python face.py --prototxt src\deploy.prototxt.txt --model src\res10_300x300_ssd_iter_140000.caffemodel
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import the necessary packages
import numpy as np
import argparse
import cv2

# import from label image


import sys
import time

import tensorflow as tf


import glob
import os

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def read_tensor_from_image_file(numpyface, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  float_caster = tf.cast(numpyface, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def classify(numpyface):
  model_file = "./../poets/tf_files/retrained_graph.pb"
  label_file = "./../poets/tf_files/retrained_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  input_layer = "Mul"
  output_layer = "final_result"

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(numpyface,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
  template = "{} (score={:0.5f})"
  for i in top_k:
    return(labels[i], results[i])

 # ///////////////////////////////////////////////////////////////////////////////////////

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# loop over image files
for filename in glob.glob("./images/test/*.jpg"):
	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	image = cv2.imread(filename)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
	 
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	 
				
			
			aface = image[startY:endY, startX:endX]
			isglassed = classify(aface)
			if isglassed[0] == 'facewithglasses':
				# draw the bounding box of the face along with the associated
				# probability
				text = "{:.2f}%".format(isglassed[1] * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(image, (startX, startY), (endX, endY),
					(0, 0, 255), 8)
				cv2.putText(image, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 8)

	# show the output image
	# cv2.imshow("orayt", cv2.resize(image,(300,300)))
	# cv2.waitKey(0)
	cv2.imwrite("./images/test/output/" + os.path.basename(filename), image)
