#!/usr/bin/env python
import os
import json
import socket
import json
import time
from collections import OrderedDict
import logging
from logging.handlers import RotatingFileHandler
from threading import Thread
from functions import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

## Parameters
nbData = 300; #Number of datapoints
seuil=-200; # Threshold used for computing score in percentage. The more close it is to zero, the more strict is the evaluation
registration=1; # temporal alignment or not
filt=1; # filtering of data or not
est=1; #estimation of orientation from position or kinect quaternions
rem=1; # removal of begining of the sequence (no motion) or not
ws=21; # windows size for segmentation
fastDP=1; #fast temporal alignment (using windows instead of full sequence) or not