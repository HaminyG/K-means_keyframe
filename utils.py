import math
import numpy as np
import os
import cv2

        # if len(frames)==0:
        #     i+=1
        #     print(frames)
        #     print(video_path)
        #     print(i)
# for i in dir:
#     if len(i)==7:
#         os.rename(file+'/'+i,file+'/'+i[1:7])

def frame_distance(data):
    global frame_distance
    global one2all_distance
    all_frame_distance=[]
    for frame_count_v1 in range(data.shape[0]):
        one2all_distance=[]
        for frame_count_v2 in range(data.shape[0]):
            frame_distance = []
            if frame_count_v1!=frame_count_v2:
                for point_index in range(data.shape[1]):
                    x1=data[frame_count_v1,point_index][0]
                    y1=data[frame_count_v1,point_index][1]
                    x2=data[frame_count_v2,point_index][0]
                    y2=data[frame_count_v2,point_index][1]
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + ((1 / 30) * frame_count_v2-frame_count_v1) ** 2)
                    frame_distance.append(distance)
                one2all_distance.append(frame_distance)
            else:
                continue
        all_frame_distance.append(one2all_distance)
    all_frame_distance=np.array(all_frame_distance).reshape(data.shape[0],-1)
    return all_frame_distance

#
#     all_frame_distance=np.concatenate((pre_frame_distance,later_frame_distance),axis=0)
#     all_frame_distance=all_frame_distance.reshape(-1)
