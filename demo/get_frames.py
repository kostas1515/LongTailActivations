import cv2
from argparse import ArgumentParser
import os
import numpy as np 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--to_imgs',default=None,type=str, help='path to video')
    parser.add_argument('--to_video',default=None,type=str, help='path to images dir')
    args = parser.parse_args()
    return args

def main(args):
    if (args.to_imgs):
        img2save=os.path.join(os.path.dirname(args.to_imgs),'images')
        if os.path.exists(img2save) is False:
            os.mkdir(img2save)
        vidcap = cv2.VideoCapture(args.to_imgs)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success,image = vidcap.read()
        count = 0
        while success:
          cv2.imwrite(os.path.join(img2save,f'frame{count}.jpg'), image)     # save frame as JPEG file      
          success,image = vidcap.read()
#           print('Read a new frame: ', success)
          count += 1
    
    if (args.to_video):
        image_folder = os.path.join(os.path.dirname(args.to_video),'detections')
        video_name = args.to_video

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        order = [int(f.split('.')[0][5:]) for f in images]
        order = sorted(range(len(order)), key=lambda k: order[k])
        images = np.array(images)[order]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

#         cv2.destroyAllWindows()
        video.release()

    
if __name__ == '__main__':
    args = parse_args()
    main(args)