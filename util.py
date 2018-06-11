import numpy as np
import cv2 as cv
import sys
import time

import model

def add_text(im, txt, pos ,maz=255):
    cv.putText(
        im,
        txt,
        pos,
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, maz),
        2,
        cv.LINE_AA
    )


def read_file(filename):
    content = ""
    with open(filename, "r") as f:
        content += f.read()
    return content
