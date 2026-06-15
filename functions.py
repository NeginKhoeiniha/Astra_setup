import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import tifffile


def read_log_data(path, prefix, white, angles, pixels_x, pixels_y):
    det_offset = int(63 * 2)  # in bytes
    det_len = int(pixels_x * pixels_y)
    g = np.zeros((pixels_y, angles, pixels_x), dtype='float32')
    def read_proj(pr):
        g[:, pr, :] = np.fromfile(path + prefix + str(pr+1).zfill(4) + '.tif', dtype='uint16', offset=det_offset, count=det_len).reshape(pixels_y, pixels_x)
        g[:, pr, :] = -np.log(np.maximum(0.001, g[:, pr, :] / white))

    pool = ThreadPool(mp.cpu_count() - 10)
    pool.map(read_proj, np.arange(angles))
    pool.close()

    return g


def vectors_astra(pixel, voxel, sod, sdd, rot_step, angles, det_x, det_y, eta, theta, phi):

    # eta theta phi in degrees
    
    pixel /= voxel 
    angles_seq = (np.arange(angles) * rot_step + 0.) #* np.pi / 180 #CTPROinitialAngle
    angles_seq *= np.pi / 180  
    sod /= voxel
    sdd /= voxel
    odd = sdd - sod  
    mgn_i = sod / sdd
    
    det_x /= voxel
    det_y /= voxel 

    eta *= np.pi / 180
    theta *= np.pi / 180 
    phi *= np.pi / 180

    sangles = np.sin(angles_seq)
    cangles = np.cos(angles_seq)

    # rotation matrices
    
    def ss(a):
        return np.sin(a)
    def cc(a):
        return np.cos(a)
    def rot_eta(a):
        return np.array([[1,0,0],[0,cc(a),-ss(a)],[0,ss(a),cc(a)]])
    def rot_theta(a):
        return np.array([[cc(a),0,-ss(a)],[0,1,0],[ss(a),0,cc(a)]])
    def rot_phi(a):
        return np.array([[cc(a),-ss(a),0],[ss(a),cc(a),0],[0,0,1]])

    u_shift = sangles * det_x
    v_shift = cangles * det_x

    rot_det = rot_theta(theta) @ rot_phi(phi) @ rot_eta(eta)

    det_u = rot_det @ np.array([0, pixel, 0])
    det_v = rot_det @ np.array([0, 0, pixel]) 

    vectors = np.zeros((angles, 12))
    # (source, detector center, det01, det10)
    
    vectors[:,0] = cangles * sod     
    vectors[:,1] = -sangles * sod     
    vectors[:,2] = 0
    vectors[:,3] = -cangles * odd + u_shift    
    vectors[:,4] = sangles * odd + v_shift
    vectors[:,5] = det_y
    vectors[:,6] = cangles * det_u[0] + sangles * det_u[1]
    vectors[:,7] = -sangles * det_u[0] + cangles * det_u[1]
    vectors[:,8] = det_u[2]
    vectors[:,9] = cangles * det_v[0] + sangles * det_v[1]
    vectors[:,10] = -sangles * det_v[0] + cangles * det_v[1]
    vectors[:,11] = det_v[2]

    return vectors

