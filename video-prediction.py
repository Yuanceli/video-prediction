import cv2
import numpy as np

mnist_test_seq = np.load('mnist_test_seq.npy').swapaxes(0,1)    # shape:(10000, 20, 64, 64)
height = mnist_test_seq.shape[2]
width = mnist_test_seq.shape[3]

def generate_video(video_array=mnist_test_seq, video_filename='output.avi', fps=5, width=width, height=height):
    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # new frame after each addition of water
    for i in range(100):
        for j in range(20):
        #add this array to the video
            gray = cv2.normalize(video_array[i,j,:,:], None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            gray_3c = cv2.merge([gray, gray, gray])
            out.write(gray_3c)

    # close out the video writer
    out.release()
