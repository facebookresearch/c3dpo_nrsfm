"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from PIL import Image
import tempfile
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


class VideoWriter:

    def __init__(self,
                 ffmpeg_bin='ffmpeg',
                 out_path='/tmp/video.mp4',
                 fps=20,
                 output_format='visdom'):

        print("video writer for %s" % out_path)

        self.output_format = output_format
        self.fps = fps
        self.out_path = out_path
        self.ffmpeg_bin = ffmpeg_bin
        self.frames = []
        self.regexp = 'frame_%08d.png'
        self.frame_num = 0

        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = self.temp_dir.name

    def __del__(self):
        self.temp_dir.cleanup()

    def write_frame(self, frame, resize=None):

        outfile = os.path.join(self.cache_dir, self.regexp % self.frame_num)

        ftype = type(frame)
        if ftype == matplotlib.figure.Figure:
            plt.savefig(outfile)
            im = None
        elif ftype == np.array or ftype == np.ndarray:
            im = Image.fromarray(frame)
        elif ftype == Image.Image:
            im = frame
        elif ftype == str:
            im = Image.open(frame).convert('RGB')
        else:
            raise ValueError('cant convert type %s' % str(ftype))

        if im is not None:
            if resize is not None:
                if type(resize) in (float,):
                    resize = [int(resize*s) for s in im.size]
                else:
                    resize = [int(resize[1]), int(resize[0])]
                resize[0] += resize[0] % 2
                resize[1] += resize[1] % 2
                im = im.resize(resize, Image.ANTIALIAS)
            im.save(outfile)

        self.frames.append(outfile)
        self.frame_num += 1

    def get_video(self, silent=True):
        regexp = os.path.join(self.cache_dir, self.regexp)
        if self.output_format == 'visdom':  # works for ppt too
            ffmcmd_ = "%s -r %d -i %s -vcodec h264 -f mp4 \
                       -y -b 2000k -pix_fmt yuv420p %s" % \
                (self.ffmpeg_bin, self.fps, regexp, self.out_path)
        else:
            raise ValueError('no such output type %s' %
                             str(self.output_format))

        print(ffmcmd_)

        if silent:
            ffmcmd_ += ' > /dev/null 2>&1'

        os.system(ffmcmd_)
        return self.out_path
