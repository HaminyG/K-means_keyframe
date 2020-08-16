import numpy as np
import cv2

from crop_keypoint import keypoint

#place my py file in the same directory and import your file here
#
# video_path='C:\\Users\\han006\\Desktop\\sample\\000000\\P01_01_00_0_color.avi'
# json_path='D:\\video_json_word\\000000\\P01_01_00_0_color.json'

# vidcap = cv2.VideoCapture(video_path)
# fp= vidcap.get(cv2.CAP_PROP_FPS)

class cropFrames:


     def get_min_x(self,body):
        x_min = body[0, 0][0]  # assign 1st value of x co-ordinate from the result
        for keyList in range(len(body)):
            for innerList in range(len(body[keyList])):
                # x_min = body[keyList,innerList][0]
                if (body[keyList, innerList][2] == 0.0):
                    # print("if")
                    continue
                else:
                    if (x_min > body[keyList, innerList][0]):
                        x_min = body[keyList, innerList][0]

        return x_min

     def get_max_x(self,body):
        x_max = body[0, 0][0]  # assign 1st value of x co-ordinate from the result

        for keyList in range(len(body)):
            for innerList in range(len(body[keyList])):
                # x_min = body[keyList,innerList][0]
                if (body[keyList, innerList][2] == 0.0):
                    # print("if")
                    continue
                else:
                    if (x_max < body[keyList, innerList][0]):
                        x_max = body[keyList, innerList][0]

        return x_max

     def get_min_y(self,body):
        y_min = body[0, 0][1]  # assign 1st value of x co-ordinate from the result

        for keyList in range(len(body)):
            for innerList in range(len(body[keyList])):
                # x_min = body[keyList,innerList][0]
                if (body[keyList, innerList][2] == 0.0):
                    # print("if")
                    continue
                else:
                    if (y_min > body[keyList, innerList][1]):
                        y_min = body[keyList, innerList][1]

        return y_min

     def get_max_y(self,body):
        y_max = body[0, 0][1]  # assign 1st value of x co-ordinate from the result

        for keyList in range(len(body)):
            for innerList in range(len(body[keyList])):
                # x_min = body[keyList,innerList][0]
                if (body[keyList, innerList][2] == 0.0):
                    # print("if")
                    continue
                else:
                    if (y_max < body[keyList, innerList][1]):
                        y_max = body[keyList, innerList][1]
        return y_max
     #
     # def getFrame(self, sec, min_y, max_y, min_x, max_x):
     #    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
     #    hasFrames, image = vidcap.read()
     #    if hasFrames:
     #
     #        # crop the image using array slices -- it's a NumPy array
     #        # after all!
     #        cropped = image[int(min_y-50):int(max_y+50), int(min_x - 50):int(max_x + 50)]
     #        cv2.imshow("cropped", cropped)
     #        cv2.waitKey(15)
     #    return hasFrames


#
#
# cropFrames = cropFrames()
# keypoint1=keypoint(json_path,video_path)
#
# min_x = cropFrames.get_min_x(keypoint1.get_x_y())
# max_x = cropFrames.get_max_x(keypoint1.get_x_y())
# min_y = cropFrames.get_min_y(keypoint1.get_x_y())
# max_y = cropFrames.get_max_y(keypoint1.get_x_y())
#
#
# sec = 0
# frameRate = int(1000/int(fp))/1000
# count=1
# success = cropFrames.getFrame(sec, min_y, max_y, min_x, max_x)
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = cropFrames.getFrame(sec, min_y, max_y, min_x, max_x)