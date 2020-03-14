import os

import cv2
import argparse
import chainer

from media_reader import VideoReader, get_filename_without_extension
from pose_detector import PoseDetector, draw_person_pose
from logging import basicConfig, getLogger, DEBUG

chainer.using_config('enable_backprop', False)

if __name__ == '__main__':
    basicConfig(level=DEBUG)
    logger = getLogger(__name__)
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
        poses, _ = pose_detector(img)
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, poses), 0.4, 0)
        logger.debug("type: {}".format(type(poses)))
        logger.debug("shape: {}".format(poses.shape))
        logger.debug(poses)
        # cv2.imshow(file_body_name + '_result', res_img)
        video.write(res_img)
        # print('Saving file into {}{}{}{}'.format(resultDir, file_body_name, str(idx), file_append_name))
        # cv2.imwrite(resultDir + file_body_name + str(idx) + file_append_name, res_img)
        # idx += 1
        cv2.waitKey(1)

    video.release()
