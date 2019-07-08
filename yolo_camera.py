# -*- coding: UTF-8 -*-
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   Author      : waynamigo
#   Created date: 19-6-14 下午1:24
#   Description :
#
#================================================================
import sys
import argparse
from yolo import YOLO, detect_camera
from PIL import Image
if __name__ == '__main__':
    detect_camera(YOLO())
