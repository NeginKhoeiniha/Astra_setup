import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI, RectBivariateSpline as RBS
from skimage.registration import phase_cross_correlation as pcc
from skimage.transform import rotate
import astra
from numpy.fft import fftshift as SHFT
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import tomosipo as ts
import functions
from skimage.restoration import estimate_sigma
import torch 
import functions
from scipy.ndimage import sobel, shift
import skimage.registration
from skimage.transform import warp_polar, rotate, rescale
import ctools


# reading the data 
num_of_projections= 360
white = 50000
pixels = 1000

P = './DataSet_3103/01/'
pre= '010'

projections = read_data(P, pre, white, num_of_projections, pixels)
print('finished loading data')

print(projections.shape)