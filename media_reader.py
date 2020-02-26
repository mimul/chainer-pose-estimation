import cv2


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        self.filename = self.file_names[self.idx]
        img = cv2.imread(self.filename, cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.filename))
        self.idx = self.idx + 1
        return img, self.filename


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        # try:  # OpenCV needs int to read from webcam
        #     self.file_name = int(file_name)
        # except ValueError:
        #     pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def get_filename_without_extension(path):
    return path.split('\\').pop().split('/').pop().rsplit('.', 1)[0]
