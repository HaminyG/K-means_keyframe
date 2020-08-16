import cv2
import numpy as np
def keyframes(video_frame,total_frame):
        del video_frame[0]
        final_frame=[]
        video_frame = np.array(video_frame)
        start = 1
        step = int(len(video_frame)/total_frame)
        for i in range(total_frame):
            video=video_frame[start+i*step]#.convert('L')
            final_frame.append(video)
        return final_frame
