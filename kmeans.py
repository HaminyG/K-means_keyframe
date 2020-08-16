import json
import numpy as np
import torch
import cv2
from sklearn.cluster import KMeans

class keyframes:
    def __init__(self, json_path,video_frame):
        self.path = json_path
        self.video_frame=video_frame
    def get_x_y(self):
        with open(self.path, 'r') as f:
            data = json.load(f)  # all frames data in one video
            video_coordinate = []
            frame_coordinate = []
            for num_frame in range(1, len(data)-1, 1):
                # bodylist=np.array(list(data[num_frame].values())[0][0])
                # facelist=np.array(list(data[num_frame].values())[1][0])
                lefthandlist = np.array(list(data[num_frame].values())[2][0])
                righthandlist = np.array(list(data[num_frame].values())[3][0])
                # total_keypoint_list=np.concatenate((bodylist,facelist,lefthandlist,righthandlist),axis=0)
                total_keypoint_list = np.concatenate((lefthandlist, righthandlist), axis=0)
                frame_coordinate.append(total_keypoint_list)
            video_coordinate.append(frame_coordinate)
            final_video_coordinate = np.array(video_coordinate)[0]
        return final_video_coordinate

    def keyframes_id(self,keyframe_num):
        global final_keyframe
        del self.video_frame[0]
       # del self.video_frame[-1]#中国的数据集需要uncomment
        video_frame = np.array(self.video_frame)
        frame = self.get_x_y()
        final_keypoint = np.array(frame)
        final_keypoint = np.delete(final_keypoint, 2, 2)
        final = np.concatenate((final_keypoint[:, :, 0], final_keypoint[:, :, 1]), axis=1)
        kmeans = KMeans(n_clusters=keyframe_num,init='k-means', random_state=0,max_iter=100).fit(final)
        index_count = 0
        cluster = 0
        index_list = []
        while cluster != 16:
            for klabel in kmeans.labels_:
                if klabel == cluster:
                    index_list.append(index_count)
                    index_count = 0
                    break
                else:
                    index_count += 1
            cluster += 1
        index_list=sorted(index_list)
        final_keyframe=[]
        for keyframe_index in index_list:
            final_keyframe.append(video_frame[keyframe_index])
        final_keyframe=np.array(final_keyframe)
        #final_keyframe=torch.from_numpy(final_keyframe)

        return final_keyframe
