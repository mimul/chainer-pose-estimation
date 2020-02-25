# Chainer Realtime Multi-Person Pose Estimation

This is an implementation of [Realtime Multi-Person Pose Estimation](https://arxiv.org/abs/1611.08050) with Chainer.
The original project is <a href="https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation">here</a>.

## Content
1. [Converting caffe model](#convert-caffe-model-to-chainer-model)
2. [Testing](#test-using-the-trained-model)

## Requirements

- Python 3.0+
- Chainer 2.0+
- NumPy
- Matplotlib
- OpenCV

## 1. Convert Caffe model to Chainer model
The authors of [the original implementation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) provide trained caffe model
which you can use to extract model weights.
Execute the following commands to download the trained model and convert it to npz file:

```
> cd models
> wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
> wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel
> wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel

> python convert_model.py posenet pose_iter_440000.caffemodel coco_posenet.npz
> python convert_model.py facenet pose_iter_116000.caffemodel facenet.npz
> python convert_model.py handnet pose_iter_102000.caffemodel handnet.npz
```

## 2. Test using the trained model
Execute the following command with the weight parameter file and the image file as arguments for estimating pose.
The resulting image will be saved as `result.png`.

```
python pose_detector.py posenet models/coco_posenet.npz --images data/image/deep_squat.jpg data/image/hurdle_step.jpg data/image/in_line_lunge.jpg data/image/rotary_stability.jpg
```


If you have a gpu device, use the `--gpu` option.

```
python pose_detector.py posenet models/coco_posenet.npz --images data/image/deep_squat.jpg data/image/hurdle_step.jpg data/image/in_line_lunge.jpg data/image/rotary_stability.jpg --gpu 0
```

<div align="center">
<img src="data/image/deep_squat.jpg" width="400">
&nbsp;
<img src="data/image_result/deep_squat_result.png" width="400">
</div>




If you have a web camera, you can execute the following cammand to run real-time demostration mode with the camera activated. Quit with the `q` key.

<b>Real-time pose estimation:</b>

```
python video_pose_demo.py
```

## Related repository
- CVPR'16, [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release).
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).


## Citation
Please cite the original paper in your publications if it helps your research:    

    @InProceedings{cao2017realtime,
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
      }
