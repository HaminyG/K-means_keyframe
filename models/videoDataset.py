import torch
import cv2
import os
from torch.utils.data import Dataset
import numpy as np
import random
import csv
from crop_keypoint import keypoint
from crop import cropFrames
from keyframe import keyframes
class VideoDataset(Dataset):

    def __init__(self, videodir, jsondir, binarydir, channels, frame_num, mode, RGBtransform, binarytransform):
        # frame: how many frame we want in small list. For example, if i want 3 numbers in each small list. [[1,2,3,],[4,5,6]]
        # random_frame_number:how many random frame we want in small list. For example, if i want 2 numbers in each small list. [[1,2,3,],[4,5,6]]->[[1,2],[5,6]]
        total_number = 0
        for paths, dirs, files in os.walk(videodir):
            total_number += len(files)
        self.total_number = total_number
        self.videodir = videodir
        self.binarydir = binarydir
        self.train = '/home/han006/experiment_v3/CSL5000_100class/dataset/devision-D/train.txt'
        self.test = '/home/han006/experiment_v3/CSL5000_100class/dataset/devision-D/test.txt'
        self.jsondir = jsondir
        self.channels = channels
        self.RGBtransform = RGBtransform
        self.binarytransform = binarytransform
        self.total_frame = frame_num
        self.mode = mode

    def get_path(self, train_or_test_txt):
        all_path = []
        with open(train_or_test_txt, 'r') as f:
            output = f.readlines()
            for data in output:
                data_info = data.split('/')
                label = int(data_info[0])
                video_name = data.split('\n')[0] + '.avi'
                json_name = data.split('\n')[0] + '.json'
                final_path = self.videodir + '/' + video_name
                final_json_path = self.jsondir + '/' + json_name
                binary_path = self.binarydir + '/' + video_name
                all_path.append((final_path, final_json_path, binary_path, label))
            return all_path

    def __len__(self):
        if self.mode == 'train':
            return len(self.get_path(self.train))
        else:
            return len(self.get_path(self.test))

    def readVideo(self, videofile,jsonfile,binaryfile):
        global final_keyframe
        global binary_final_keyframe
        keypoints_class = keypoint(jsonfile)
        #print('class',keypoints_class.get_x_y().shape)
        cropFrame_class = cropFrames()
        cap = cv2.VideoCapture(videofile)
        binary_cap= cv2.VideoCapture(binaryfile)
        min_x = cropFrame_class.get_min_x(keypoints_class.get_x_y())
        max_x = cropFrame_class.get_max_x(keypoints_class.get_x_y())
        min_y = cropFrame_class.get_min_y(keypoints_class.get_x_y())
        max_y = cropFrame_class.get_max_y(keypoints_class.get_x_y())
        grayframes=[]
        RGBframes=[]
        frames = []
        binaryframes=[]

        while (cap.isOpened()):
            binary_ret,binary_frame=binary_cap.read()
            ret, frame = cap.read()

            if ret == True:
                crop_frame = frame[int(min_y - 5):int(max_y + 5), int(min_x - 5):int(max_x + 5)]
                binarycropframe=binary_frame[int(min_y - 5):int(max_y + 5), int(min_x - 5):int(max_x + 5)]
                frames.append(crop_frame)
                binaryframes.append(binarycropframe)
            else:
                break
        if self.channels == 3:
            if self.mode == 'train':
                LUVframes=keyframes(frames,self.total_frame)
                binary_frames=keyframes(binaryframes,self.total_frame)
                for LUVframe in LUVframes:
                    crop_frame = cv2.cvtColor(LUVframe, cv2.COLOR_BGR2RGB)
                    RGBframes.append(crop_frame)
                final_keyframe=np.array(RGBframes)
                binary_final_keyframe=np.array(binary_frames)
            elif self.mode=='val':
                LUVframes=keyframes(frames,self.total_frame)
                binary_frames=keyframes(binaryframes,self.total_frame)
                for LUVframe in LUVframes:
                    crop_frame = cv2.cvtColor(LUVframe, cv2.COLOR_BGR2RGB)
                    RGBframes.append(crop_frame)
                final_keyframe=np.array(RGBframes)
                binary_final_keyframe = np.array(binary_frames)
        elif self.channels==1:
            if self.mode=='train':
                LUVframes=keyframes(frames,self.total_frame)
                binary_frames=keyframes(binaryframes,self.total_frame)
                for LUVframe in LUVframes:
                    crop_frame = cv2.cvtColor(LUVframe, cv2.COLOR_BGR2RGB)
                    crop_frame =cv2.cvtColor(crop_frame,cv2.COLOR_BGR2GRAY)
                    crop_frame = np.expand_dims(crop_frame, axis=2)
                    grayframes.append(crop_frame)
                final_keyframe=np.array(grayframes)
                binary_final_keyframe = np.array(binary_frames)
            elif self.mode=='val':
                LUVframes=keyframes(frames,self.total_frame)
                binary_frames=keyframes(binaryframes,self.total_frame)
                for LUVframe in LUVframes:
                    crop_frame = cv2.cvtColor(LUVframe, cv2.COLOR_BGR2RGB)
                    crop_frame =cv2.cvtColor(crop_frame,cv2.COLOR_BGR2GRAY)
                    crop_frame = np.expand_dims(crop_frame, axis=2)
                    grayframes.append(crop_frame)
                final_keyframe=np.array(grayframes)
                binary_final_keyframe = np.array(binary_frames)

        return final_keyframe,binary_final_keyframe

    def __getitem__(self, index):
        if self.mode == 'train':
            data = self.get_path(self.train)
        else:
            data = self.get_path(self.test)
        videopath, jsonpath, binarypath, videolabel = data[index]
        video_data,binaryvideo_data = self.readVideo(videopath,jsonpath,binarypath)
        video_frame_data_list=[]
        binary_frame_data_list=[]
        for video_frame_data in video_data:
            video_frame_data=self.RGBtransform(video_frame_data)
            video_frame_data_list.append(video_frame_data)
        video = torch.stack(video_frame_data_list,
                                            dim=0)
        for video_frame_data in binaryvideo_data:
            video_frame_data=self.binarytransform(video_frame_data)
            binary_frame_data_list.append(video_frame_data)
        binaryvideo_data = torch.stack(binary_frame_data_list,
                                            dim=0)
        video=video.permute(1,0,2,3)
        binaryvideo_data=binaryvideo_data.permute(1,0,2,3)
        # 用其实现 final_frame.append(in_tensor) 的功能：先构造已经append好的final_frame（此时final_frame为list），然后final_frame = torch.stack(final_frame, dim = 0)
        videolabel = torch.tensor(videolabel)

        return video,binaryvideo_data,videolabel