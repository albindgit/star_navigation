import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import numpy as np
import pathlib
import os
import subprocess
import glob

class VideoWriter:

    def __init__(self, file_name, framerate):
        self.file_name = file_name
        self.video = None
        self.framerate = framerate
        self.videos_path = str(pathlib.PurePath(pathlib.Path(__file__).parents[0].joinpath("videos")))

    def add_frame(self, fig, k):
        fig.savefig(str(pathlib.PurePath(self.videos_path, self.file_name + '%02d.png' % k)))
        return None

        fig.set_size_inches(16, 9)
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        if self.video is None:
            file_path = str(pathlib.PurePath(pathlib.Path(__file__).parents[0].joinpath("videos"), self.file_name + '.mp4'))
            self.video = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), self.framerate, (mat.shape[1], mat.shape[0]))

        # write frame to video
        self.video.write(mat)

    def release(self):
        os.chdir(self.videos_path)
        subprocess.call([
            'ffmpeg', '-i', self.file_name + '%02d.png', '-framerate', str(self.framerate), '-pix_fmt', 'yuv420p',
            self.file_name+'.mp4'
        ])
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
        return None
        cv2.destroyAllWindows()
        self.video.release()
