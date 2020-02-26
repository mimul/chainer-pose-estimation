import os

import cv2
import argparse
import chainer

from media_reader import VideoReader, get_filename_without_extension
from pose_detector import PoseDetector, draw_person_pose

chainer.using_config('enable_backprop', False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--video', type=str, default='', help='video file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if args.video == '':
        raise ValueError('Either --video has to be provided')

    chainer.config.enable_backprop = False
    chainer.config.train = False

    # load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)

    print(args.video)

    resultDir = "data/video_result/"

    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    file_body_name = get_filename_without_extension(args.video)
    file_append_name = '_result.mp4'

    video = cv2.VideoWriter(resultDir + file_body_name + file_append_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30.0, (1280, 720))
    # read video
    video_provider = VideoReader(args.video)
    idx = 0
    for img in video_provider:
        person_pose_array, _ = pose_detector(img)
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)
        # cv2.imshow(file_body_name + '_result', res_img)
        video.write(res_img)
        # print('Saving file into {}{}{}{}'.format(resultDir, file_body_name, str(idx), file_append_name))
        # cv2.imwrite(resultDir + file_body_name + str(idx) + file_append_name, res_img)
        # idx += 1
        cv2.waitKey(1)

    video.release()
