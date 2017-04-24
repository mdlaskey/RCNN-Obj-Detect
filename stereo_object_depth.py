
import hsrb_interface
import rospy
import sys
import math
import tf
import tf2_ros
import tf2_geometry_msgs
import IPython
import cv2
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from hsrb_interface import geometry
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped

import cv2
from cv_bridge import CvBridge, CvBridgeError
import IPython

from sensors import Head_Right_RGB, Head_Left_RGB

sys.path.append('../Faster-RCNN_TF/lib/')
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from networks.factory import get_network
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy.linalg as LA

import tensorflow as tfl
import numpy as np

from image_geometry import StereoCameraModel
from itertools import combinations


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


class Depth_Object():

	def __init__(self,label):
		self.robot = hsrb_interface.Robot()
		self.omni_base = self.robot.get('omni_base')
		self.whole_body = self.robot.get('whole_body')
		self.gripper = self.robot.get('gripper')
		self.label = label
		self.br = tf.TransformBroadcaster()
		self.sift = cv2.SIFT()


		self.focal = 5.0
		self.baseline = 5.0

		self.cam_r = Head_Right_RGB()
		self.cam_l = Head_Left_RGB()

		self.st = StereoCameraModel()

		self.st.fromCameraInfo(self.cam_l.read_info(),self.cam_r.read_info())


		self.count = 0
		self.sess = tfl.Session(config=tfl.ConfigProto(allow_soft_placement=True))
		# load network
		#IPython.embed()
		self.net = get_network('VGGnet_test')
		# load model
		self.saver = tfl.train.Saver(write_version=tfl.train.SaverDef.V1)
		self.saver.restore(self.sess, '../Faster-RCNN_TF/model/VGGnet_fast_rcnn_iter_70000.ckpt')



	def vis_detections(self,im, class_name, dets, thresh=0.5):
		"""Draw detected bounding boxes."""
		inds = np.where(dets[:, -1] >= thresh)[0]
		if len(inds) == 0:
			return None, None

		centers = []
		for i in inds:
			bbox = dets[i, :4]
			score = dets[i, -1]
			im = im.copy()
			cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255))

			center = np.zeros(2)
			center[0] = bbox[0] + (bbox[2] - bbox[0])/2.0
			center[1] = bbox[1] + (bbox[3] - bbox[1])/2.0
			print center
			im[center[1]-20:center[1]+20,center[0]-20:center[0]+20,:] = [0,0,255]

			centers.append(center)
			
		return centers,im


	def detect(self,im):
		timer = Timer()
		timer.tic()
		scores, boxes = im_detect(self.sess, self.net, im)
		timer.toc()
		print ('Detection took {:.3f}s for '
		       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

		# Visualize detections for each class
		#im = im[:, :, (2, 1, 0)]

		# fig, ax = plt.subplots(figsize=(12, 12))
		# ax.imshow(im, aspect='equal')
		im_b = None
		centers = None

		CONF_THRESH = 0.5
		NMS_THRESH = 0.3
		for cls_ind, cls in enumerate(CLASSES[1:]):
			cls_ind += 1 # because we skipped background
			#IPython.embed()
			if(cls == self.label):
				cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
				cls_scores = scores[:, cls_ind]
				dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
				keep = nms(dets, NMS_THRESH)
				dets = dets[keep, :]
				
				centers,im_b = self.vis_detections(im, cls, dets, thresh=CONF_THRESH)
				

		return centers,im_b


	def get_image_box(self,im,center):
		WIDTH = 50
		HEIGHT = 200

		box = im[int(center[1])-WIDTH:int(center[1])+WIDTH,int(center[0])-HEIGHT:int(center[0])+HEIGHT,:]
		#box = cv2.cvtColor(box, cv2.cv.CV_RGB2GRAY)
		# cv2.imshow('box_making',box)
		# cv2.waitKey(300)
		# print "BOX SIZE"
		# print box.shape

		return box

	def match_correspondences(self,p_l,p_r,img_r,img_l):

		l_points = []
		r_points = []
		
		#Compute Histograms for each image
		for p in p_l: 
			box = self.get_image_box(img_l,p)

			# hist = cv2.calcHist([box],[0],None,[255],[0,256])
			# l_points.append([hist,p])
			l_points.append([box,p])

		for p in p_r: 
			box = self.get_image_box(img_r,p)
			# hist = cv2.calcHist([box],[0],None,[255],[0,256])
			# r_points.append([hist,p])
			r_points.append([box,p])

		#IPython.embed()


		sim_matrix = np.zeros([len(p_l),len(p_r)])
		for i in range(len(p_l)):
			for j in range(len(p_r)):
				hist_l = l_points[i][0]
				hist_r = r_points[j][0]
				#sim_matrix[i,j] = cv2.compareHist(hist_l,hist_r, cv2.cv.CV_COMP_CORREL)
				try:
					val = cv2.matchTemplate(hist_l,hist_r,cv2.TM_CCORR_NORMED)
					sim_matrix[i,j] = val[0][0]
				except:
					sim_matrix[i,j] = -10000

	

		new_p_l = []
		new_p_r = []
		for i in range(len(p_l)):
			high_val = 0.0
			best_match = None
			for j in range(len(p_r)):

				var = sim_matrix[i,j]
				if(var > high_val):
					mx_var = np.max(sim_matrix[:,j])
					if(var >= mx_var):
						best_match = j
						high_val = var

			if(not best_match == None):
				new_p_l.append(p_l[i])
				new_p_r.append(p_r[best_match])

		print "SIM MATRIX"
		print sim_matrix


		#prune duplicates
		#IPython.embed()

		return new_p_l,new_p_r


	def get_poses(self,p_r,p_l,img_r,img_l):
		#returns list of left (u,v,disparity)

		p_l.sort(key = lambda x: x[0])
		p_r.sort(key = lambda x: x[0])
		poses = []

		p_l,p_r = self.match_correspondences(p_l,p_r,img_r,img_l)

		for i in range(len(p_l)): 
			p = p_l[i]
			d = p[0] - p_r[i][0]
			poses.append((p[0],p[1],d))

		return poses

	def get_state(self):
		img_r = self.cam_r.read_data()
		img_l = self.cam_l.read_data()

		if(img_r == None or img_l == None):
			rospy.logerr('no image from robot camera')
		

		p_r,img_r_b = self.detect(img_r)

		p_l,img_l_b = self.detect(img_l)

		if(p_r == None or p_l == None):
			return []


		if(not img_r == None):
			cv2.imshow('image_r',img_r_b)
			cv2.waitKey(30)


		if(not img_l == None):
			cv2.imshow('image_l',img_l_b)
			cv2.waitKey(30)
		
		self.count += 1
		poses = self.get_poses(p_r,p_l,img_r,img_l)


		# n_b = self.find_nearest(poses)
		
		return poses

	def broadcast_poses(self):
		while True: 
			poses = self.get_state()
			count = 0
			for pose in poses:
				print "POSE ",pose
				
				td_points = self.st.projectPixelTo3d((pose[0],pose[1]),pose[2])
				self.br.sendTransform((td_points[0], td_points[1]-0.02, td_points[2]),
						(0.0, 0.0, 0.0, 1.0),
						rospy.Time.now(),
						"object_"+str(count),
						"head_l_stereo_camera_link")
				count += 1





if __name__=='__main__':

	
	detector = 'bottle'

	#rgbd = RGBD()

	# while True: 
	# 	img = rgbd.read_data()
	# 	cv2.imshow('rgbd',img)
	# 	cv2.waitKey(30)

	do = Depth_Object(detector)
	while True: 
		do.broadcast_poses()
