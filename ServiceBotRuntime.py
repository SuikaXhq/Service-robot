from __future__ import division, print_function, absolute_import
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import GestureRecognition
import cv2
import os
import sys
import math
import numpy as np
import warnings
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
import tensorflow as tf
# -------------------------------------#
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"#选择哪一块gpu,如果是-1，就是调用cpu

config = tf.ConfigProto()#对session进行参数配置
config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction=0.9#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True#按需分配显存，这个比较重要

session = tf.Session(config=config)
# ------------------------------------- #

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/test.avi")
ap.add_argument("-c", "--class",help="name of class", default = "person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

FLAG_WAVE = 0
FLAG_FOLLOW = 1
FLAG_STOP = 2
FLAG_FINISH = 3

class ServiceBotRuntime(object):
    def __init__(self):
        # Loop until the user clicks the close button.
        self._done = False
        self._flag = FLAG_WAVE

        # Kinect runtime object, we want color, depth and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        self.color_frame_shape = (1080, 1920, 4)
        self.depth_frame_shape = (424, 512)
        self.tracker_init()
        self.last_lhand_pos = None
        self.last_rhand_pos = None
        self.lhand_move = None
        self.rhand_move = None
        self.target_id = None
        self.hand_gesture = [[None]*30]*2
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1920,1080))
    
    def tracker_init(self):
        self.yolo = YOLO()

        # deep_sort
        self.max_cosine_distance = 0.5 #余弦距离的控制阈值
        self.nn_budget = None
        self.nms_max_overlap = 0.3 #非极大抑制的阈值
        model_filename = 'deep_sort_yolov3-master/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename,batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric)
        
        blank_frame = np.zeros(self.color_frame_shape)[..., :3].astype(np.uint8)
        image = Image.fromarray(blank_frame)
        boxs, class_names = self.yolo.detect_image(image)
        features = self.encoder(blank_frame,boxs)
        # score to 1.0 here
        detections = [ddet(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)


    def get_bodies(self, track=None):
        if self._kinect.has_new_body_frame(): 
            bodies = self._kinect.get_last_body_frame().bodies
            return bodies[track] if track is not None else bodies
        else:
            return None

    def get_color_frame(self):
        if self._kinect.has_new_color_frame():
            frame = self._kinect.get_last_color_frame()
            return frame.reshape(self.color_frame_shape)[..., :3].copy()
        else:
            return None
    
    def get_depth_frame(self):
        if self._kinect.has_new_depth_frame():
            frame = self._kinect.get_last_depth_frame()
            return frame.reshape(self.depth_frame_shape).copy()
        else:
            return None

    def recongnize_wave(self, bodies):
        def count_change(array):
            result = []
            last = array[0]
            for item in array:
                if last is None or item == last:
                    result.append(False)
                    continue
                elif item != last:
                    result.append(True)
            return result


        if bodies is None:
            return False
        if self.last_lhand_pos is None:
            self.last_lhand_pos = [None]*self._kinect.max_body_count
            for i, body in enumerate(bodies):
                if body.is_tracked:
                    self.last_lhand_pos[i] = body.joints[PyKinectV2.JointType_HandLeft].Position
        if self.last_rhand_pos is None:
            self.last_rhand_pos = [None]*self._kinect.max_body_count
            for i, body in enumerate(bodies):
                if body.is_tracked:
                    self.last_rhand_pos[i] = body.joints[PyKinectV2.JointType_HandRight].Position
        if self.lhand_move is None: self.lhand_move = [[None]*50]*self._kinect.max_body_count
        if self.rhand_move is None: self.rhand_move = [[None]*50]*self._kinect.max_body_count
        for i, body in enumerate(bodies):
            if body.is_tracked:
                
                if body.joints[PyKinectV2.JointType_HandLeft].Position and body.joints[PyKinectV2.JointType_ElbowLeft].Position and body.joints[PyKinectV2.JointType_HandRight].Position and body.joints[PyKinectV2.JointType_ElbowRight].Position:
                    # left
                    if body.joints[PyKinectV2.JointType_HandLeft].Position.y > body.joints[PyKinectV2.JointType_ElbowLeft].Position.y:
                        if self.last_lhand_pos[i] is None:
                            self.last_lhand_pos[i] = body.joints[PyKinectV2.JointType_HandLeft].Position
                        if body.joints[PyKinectV2.JointType_HandLeft].Position.x > self.last_lhand_pos[i].x:
                            self.lhand_move[i].append(True)
                            self.lhand_move[i].pop(0)
                            self.last_lhand_pos[i] = body.joints[PyKinectV2.JointType_HandLeft].Position
                        elif body.joints[PyKinectV2.JointType_HandLeft].Position.x < self.last_lhand_pos[i].x:
                            self.lhand_move[i].append(False)
                            self.lhand_move[i].pop(0)
                            self.last_lhand_pos[i] = body.joints[PyKinectV2.JointType_HandLeft].Position
                        
                    
                    # right
                    if body.joints[PyKinectV2.JointType_HandRight].Position.y > body.joints[PyKinectV2.JointType_ElbowRight].Position.y:
                        if self.last_rhand_pos[i] is None:
                            self.last_rhand_pos[i] = body.joints[PyKinectV2.JointType_HandRight].Position
                        if body.joints[PyKinectV2.JointType_HandRight].Position.x > self.last_rhand_pos[i].x:
                            self.rhand_move[i].append(True)
                            self.rhand_move[i].pop(0)
                            self.last_rhand_pos[i] = body.joints[PyKinectV2.JointType_HandRight].Position
                        elif body.joints[PyKinectV2.JointType_HandRight].Position.x < self.last_rhand_pos[i].x:
                            self.rhand_move[i].append(False)
                            self.rhand_move[i].pop(0)
                            self.last_rhand_pos[i] = body.joints[PyKinectV2.JointType_HandRight].Position

                    changes = np.max([np.sum(count_change(self.lhand_move[i])), np.sum(count_change(self.rhand_move[i]))])
                    if changes >= 3:
                        self.body_tracker = i
                        return True
        return False
            
    
    def recongnize_hand(self, hand_frame):
        result = GestureRecognition.grdetect(hand_frame)
        return result
        
    def recongnize_follow(self, hand_frames):
        for i, hand_frame in enumerate(hand_frames):
            result = self.recongnize_hand(hand_frame)
            if result != -1:
                self.hand_gesture[i].append(result)
                self.hand_gesture[i].pop(0)
                print('Hand {}: {}'.format(i, result))
                if result == 2:
                    # search for 2s
                    count = 0
                    for gesture in self.hand_gesture[i][::-1]:
                        if gesture == 2:
                            count += 1
                        else:
                            break
                    if count >= 5:
                        return True
        return False

    def recongnize_stop(self, hand_frames):
        for i, hand_frame in enumerate(hand_frames):
            result = self.recongnize_hand(hand_frame)
            if result != -1:
                self.hand_gesture[i].append(result)
                self.hand_gesture[i].pop(0)
                print('Hand {}: {}'.format(i, result))
                if result == 3:
                    # search for 3s
                    count = 0
                    for gesture in self.hand_gesture[i][::-1]:
                        if gesture == 3:
                            count += 1
                        else:
                            break
                    if count >= 5:
                        return True
        return False

    def confirm_track(self, frame, body):
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs, class_names = self.yolo.detect_image(image)
        features = self.encoder(frame,boxs)
        # score to 1.0 here
        detections = [ddet(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        target_position = self._kinect.body_joints_to_color_space(body.joints)[PyKinectV2.JointType_SpineBase]
        target_center = np.asarray([
            target_position.x,
            target_position.y
        ])

        min_distance = np.Inf

        # 确定目标
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            center = np.asarray([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
            distance = np.linalg.norm(target_center - center)
            if distance<min_distance:
                min_distance = distance
                self.target_id = track.track_id



    def tracing(self, frame, body): 
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs, class_names = self.yolo.detect_image(image)
        features = self.encoder(frame,boxs)
        # score to 1.0 here
        detections = [ddet(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if track.track_id != self.target_id:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 3)

        if hasattr(body, 'joints'):
            centerpoint = body.joints[PyKinectV2.JointType_SpineMid].Position
            distance = math.sqrt(centerpoint.x**2 + centerpoint.y**2 + centerpoint.z**2)
            offset = np.arccos(centerpoint.z/distance)
        else:
            distance = 0.0
            offset = 0.0
        return distance, offset

    def get_hand_frames(self, frame, body):
        def cut_frame(frame, box):
            if box is None:
                return np.zeros((2,2,3)).astype(np.uint8)
            for point in box:
                for u in point:
                    if u is None:
                        return np.zeros((2,2,3)).astype(np.uint8)
                if point[0] >= point[1]:
                    return np.zeros((2,2,3)).astype(np.uint8)
            result = frame[
                int(box[0][1]):int(box[1][1]),
                int(box[0][0]):int(box[1][0]),
                :
            ].copy()
            for d in result.shape:
                if d == 0:
                    result = np.zeros((2,2,3)).astype(np.uint8)
            return result

        if body is not None and hasattr(body, 'joints'):
            joints = body.joints 
            # convert joint coordinates to color space 
            joint_points = self._kinect.body_joints_to_color_space(joints)

            # Left
            distances_left = [
                np.abs( joint_points[PyKinectV2.JointType_HandLeft].x - joint_points[PyKinectV2.JointType_HandTipLeft].x ),
                np.abs( joint_points[PyKinectV2.JointType_HandLeft].y - joint_points[PyKinectV2.JointType_HandTipLeft].y ),
                np.abs( joint_points[PyKinectV2.JointType_HandLeft].x - joint_points[PyKinectV2.JointType_ThumbLeft].x ),
                np.abs( joint_points[PyKinectV2.JointType_HandLeft].y - joint_points[PyKinectV2.JointType_ThumbLeft].y )
            ]
            if np.max(distances_left)>=0 and np.max(distances_left)<= 1920:
                radius_left = int(np.max(distances_left) * 2)
                lbox = [
                    [int(joint_points[PyKinectV2.JointType_HandLeft].x - radius_left), int(joint_points[PyKinectV2.JointType_HandLeft].y - radius_left)],
                    [int(joint_points[PyKinectV2.JointType_HandLeft].x + radius_left), int(joint_points[PyKinectV2.JointType_HandLeft].y + radius_left)]
                ]
            else:
                rbox = [[None]*2]*2
            lframe = cut_frame(frame, lbox)

            # right
            distances_right = [
                np.abs( joint_points[PyKinectV2.JointType_HandRight].x - joint_points[PyKinectV2.JointType_HandTipRight].x ),
                np.abs( joint_points[PyKinectV2.JointType_HandRight].y - joint_points[PyKinectV2.JointType_HandTipRight].y ),
                np.abs( joint_points[PyKinectV2.JointType_HandRight].x - joint_points[PyKinectV2.JointType_ThumbRight].x ),
                np.abs( joint_points[PyKinectV2.JointType_HandRight].y - joint_points[PyKinectV2.JointType_ThumbRight].y )
            ]
            if np.max(distances_right)>=0 and np.max(distances_right)<= 1920:
                radius_right = int(np.max(distances_right) * 2)
                rbox = [
                    [int(joint_points[PyKinectV2.JointType_HandRight].x - radius_right), int(joint_points[PyKinectV2.JointType_HandRight].y - radius_right)],
                    [int(joint_points[PyKinectV2.JointType_HandRight].x + radius_right), int(joint_points[PyKinectV2.JointType_HandRight].y + radius_right)]
                ]
            else:
                rbox = [[None]*2]*2
            rframe = cut_frame(frame, rbox)
            cv2.rectangle(frame, lbox[0], lbox[1], (255,255,255), 3)
            lhand_text = 'Unknown' if self.hand_gesture[0][-1] is None else self.hand_gesture[0][-1]
            if lbox[0][0] is not None and lbox[0][1] is not None:
                cv2.putText(frame, '{}'.format(lhand_text), (lbox[0][0], lbox[0][1] -20), 0, 1.5, (255,255,255), 3)
            cv2.rectangle(frame, rbox[0], rbox[1], (255,255,255), 3)
            rhand_text = 'Unknown' if self.hand_gesture[1][-1] is None else self.hand_gesture[1][-1]
            if rbox[0][0] is not None and rbox[0][1] is not None:
                cv2.putText(frame, '{}'.format(rhand_text), (rbox[0][0], rbox[0][1] -20), 0, 1.5, (255,255,255), 3)
            return (lframe, rframe)
        else:
            return (np.zeros((2,2,3)).astype(np.uint8), np.zeros((2,2,3)).astype(np.uint8))

    def confirm_body(self, bodies, target_center):
        if bodies is None:
            return None
        min_dist = np.Inf
        for i,body in enumerate(bodies):
            if body.is_tracked:
                if hasattr(body, 'joints') and body.joints[JointType_SpineBase].Position is not None:
                    body_center_point = self._kinect.body_joints_to_color_space(body.joints)[JointType_SpineBase]
                    body_center = np.asarray([
                        body_center_point.x,
                        body_center_point.y
                    ])
                    dist = np.linalg.norm(body_center - target_center)
                    if dist<min_dist:
                        min_dist = dist
                        self.body_tracker = i

    def draw_body_bone(self, frame, jointPoints, color, joint0, joint1):
        if jointPoints is not None and jointPoints[joint0] is not None and jointPoints[joint1] is not None:
            if jointPoints[joint0].x>=0 and jointPoints[joint0].y>=0 and jointPoints[joint1].x>=0 and jointPoints[joint1].x>=0:
                start = (int(jointPoints[joint0].x), int(jointPoints[joint0].y))
                end = (int(jointPoints[joint1].x), int(jointPoints[joint1].y))
                cv2.line(frame, start, end, color, 5)
    
    def draw_body(self, frame, jointPoints, color):
        # Torso
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft)
    
        # Right Arm    
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight)

        # Left Arm
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft)

        # Right Leg
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight)

        # Left Leg
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft)
        self.draw_body_bone(frame, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft)

    def draw_frame(self, frame):
        cv2.putText(frame, 'Press [Q] to exit', (10, 1060), 0, 2, (0,255,0), 3)
        cv2.imshow('ServiceBot', frame)
        
    

    def run(self):
        cv2.namedWindow("ServiceBot", 0)
        cv2.moveWindow('ServiceBot', 200,50)
        cv2.resizeWindow('ServiceBot', 1600, 900)

        while not self._done:
            frame = self.get_color_frame()
            if frame is None:
                continue

            if self._flag == FLAG_WAVE:
                cv2.putText(frame, 'Waiting for waves...', (10, 50), 0, 2, (0,255,0), 3)
                bodies = self.get_bodies()
                if bodies is not None:
                    for body in bodies:
                        if body.is_tracked:
                            joints = body.joints
                            jointPoints = self._kinect.body_joints_to_color_space(joints)
                            self.draw_body(frame, jointPoints, (0,0,255))
                    if self.recongnize_wave(bodies):
                        self._flag = FLAG_FOLLOW

            elif self._flag == FLAG_FOLLOW:
                cv2.putText(frame, 'Detected wave!', (10, 50), 0, 2, (0,255,0), 3)
                body = self.get_bodies(self.body_tracker)
                hand_frames = self.get_hand_frames(frame, body)
                if self.recongnize_follow(hand_frames):
                    self._flag = FLAG_STOP
                    self.confirm_track(frame, body)

            elif self._flag == FLAG_STOP:
                cv2.putText(frame, 'Following Target...', (10, 50), 0, 2, (0,255,0), 3)
                body = self.get_bodies(self.body_tracker)
                hand_frames = self.get_hand_frames(frame, body)
                if self.recongnize_stop(hand_frames):
                    self._flag = FLAG_FINISH
                
                # Tracing
                if self.target_id is None:
                    cv2.putText(frame, 'Confirming Target...', (10, 100), 0, 2, (0,255,0), 3)
                    self.confirm_track(frame, body)
                else:
                    distance, offset = self.tracing(frame, body)
                    cv2.putText(frame, 'Distance: {}m, Offset: {} degrees'.format(np.round(distance,3), np.round(offset/np.pi*180, 2)), (10, 100), 0, 1.5, (0,255,0), 3)
            
            elif self._flag == FLAG_FINISH:
                cv2.putText(frame, 'Service ended. Please press [Q] to exit.', (500, 300), 0, 2, (0,255,0), 5)

            self.out.write(frame)
            self.draw_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._done = True
                self.out.release()
