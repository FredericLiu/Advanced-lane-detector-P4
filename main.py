import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pipelineClass import Line


'''
read in the distortion coeffects from CamerCali.py
'''
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

#perform the pipeline on the single image
'''
image = mpimg.imread('test_images/straight_lines1.jpg')
ld = Line(mtx,dist)
result = ld.img_process(image)
'''

#perform the pipeline on the video
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

#ffmpeg_extract_subclip('project_video.mp4', 22, 27, targetname=project_video_sample_path)
detector = Line(mtx,dist)
clip1 = VideoFileClip('project_video_sample.mp4')
project_video_clip = clip1.fl_image(detector.img_process) #NOTE: this function expects color images!!
project_video_clip.write_videofile('output_images/lanes_project_video_sample.mp4', audio=False)











