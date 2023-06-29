import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import numpy as np
import pathlib


class VideoWriter:

    def __init__(self, file_name, framerate):
        self.file_name = file_name
        self.video = None
        self.framerate = framerate

    def add_frame(self, fig):
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
        self.video.release()
        cv2.destroyAllWindows()