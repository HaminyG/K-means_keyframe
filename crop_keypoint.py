import json
import cv2
from utils import frame_distance
import numpy as np
# from restore_keypoint import restore_keypoint
json_path='C:\\Users\\han006\\Desktop\\video_json_word\\000000\\P01_01_00_0_color.json'
class keypoint:
    def __init__(self, json_path):
        self.path = json_path

    def get_x_y(self):
        with open(self.path, 'r') as f:
            data = json.load(f)  # all frames data in one video
            video_coordinate=[]
            frame_coordinate = []
            for num_frame in range(1, len(data), 1):

                bodylist=np.array(list(data[num_frame].values())[0][0])
                #facelist=np.array(list(data[num_frame].values())[1][0])
                lefthandlist=np.array(list(data[num_frame].values())[2][0])
                righthandlist=np.array(list(data[num_frame].values())[3][0])
                #total_keypoint_list=np.concatenate((bodylist,facelist,lefthandlist,righthandlist),axis=0)
                total_keypoint_list = np.concatenate((bodylist, lefthandlist, righthandlist), axis=0)
                frame_coordinate.append(total_keypoint_list)
            video_coordinate.append(frame_coordinate)
            final_video_coordinate=np.array(video_coordinate)[0]
        return final_video_coordinate
#
# myclass=keypoint(json_path=json_path)
# keypoint_2d_norm=[]
# frame=myclass.get_x_y()
# final_keypoint=np.array(frame)
# final_keypoint=np.delete(final_keypoint,2,2)
# # for i in range(final_keypoint.shape[0]):
# #             keypoint_2d = scaler.fit_transform(final_keypoint[i,:,:])
# #             #print(keypoint_2d.shape)
# #             keypoint_2d_norm.append(keypoint_2d)
# #         keypoint_2d_norm=np.array(keypoint_2d_norm)
# final=np.concatenate((final_keypoint[:,:,0],final_keypoint[:,:,1]),axis=1)#np.array(keypoint_2d_norm).reshape(20,-1)
# distance=frame_distance(frame)
# print(distance.shape)
# data=restore_keypoint(frame)
#
# from sklearn.cluster import KMeans
#
# kmeans = KMeans(n_clusters=16, random_state=0).fit(final)
# print(kmeans.labels_)




# for frame in range(len(video_frame)):
#     cv2.imwrite('C:\\Users\\ECE-ML\\Desktop\\research\\plan_A_code\\sample data\\video\\output video\\frame{}.jpg'.format(frame),video_frame[frame])
