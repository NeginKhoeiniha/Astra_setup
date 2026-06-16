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


def AGD(x0, A, Data, l, alpha, num_iters, eps, sigma):
    # Initialize the solution with zeros
    xk = x0
    xk1 = xk
    y = xk
    t = torch.tensor(1)

    # Gradient descent loop
    for k in range(num_iters):

        t1 = (1 + torch.sqrt(1 + 4 * t**2)) / 2
        y = xk + ((t-1)/t1) * (xk - xk1)
        
        xk1 = xk
        t = t1
        xk = y - alpha * grad_post_tv(y, Data, l, eps, sigma)

        print(f"Step: {k}", end="\r")

    return xk

def grad_post_tv(x, Data, l, eps, sigma):
    return adjoint((forward_model(x, A) - Data), A)/ (sigma**2) + l *(smooth_tv_grad(x, eps))

def smooth_tv_grad(x, eps):
    Dx1 = gradient(x, 0) 
    Dx2 = gradient(x, 1)
    Norm = torch.sqrt(gradient(x, 0) ** 2 + gradient(x, 1) ** 2)
    P1 = Dx1 / torch.max(Norm, eps)
    P2 = Dx2 / torch.max(Norm, eps)
    return (div(P1, axis = 0) + div(P2, axis = 1))

def adjoint(x, A):
    xx = x.to(device)
    bp = A.transpose()
    c = bp(xx)
    return c

def forward_model(x, A):
    xx = x.to(device)
    aa = A(xx)
    return aa.to(device)

def gradient(x, axis):
    return torch.roll(x, -1, dims= axis) - x

def div(x, axis):
    return torch.roll(x, 1, dims= axis) - x


def shift_x_2d(g, sod, sdd, pixel, rot_step):

    # x shift of the center of rotation in mm
    # based on fanbeam symmetry relantionship
    # g can be fan or conebeam

    if len(g.shape) == 3:
        g = g[g.shape[0]//2 - 1]*0 + g[g.shape[0]//2]

    angles, columns = g.shape
    
    magn_i = sod / sdd
    s_M = (columns * pixel * magn_i - 1) * 0.5
    beta = np.arange(angles) * rot_step * np.pi / 180
    s = np.linspace(-s_M, s_M, columns)# - shift_0 * magn_i

    interpolator = RBS(beta, s, g, kx = 1, ky = 1)
  
    upsampling = 1 / 0.01  # for image registration, 1 / pixels

    twopi = 2 * np.pi
    f2 = np.zeros((angles, columns))
    sh_0 = 2*np.arctan2(s, sod) + np.pi
        
    for bb in np.arange(angles):
        beta_s = sh_0 + beta[bb]
        mask = beta_s > twopi
        beta_s[mask] -= twopi
        f2[bb,:] = interpolator(beta_s, -s , grid = False)
    #plt.figure(1); plt.imshow(g)
    #plt.figure(2); plt.imshow(f2); plt.show(); exit()

    #plt.figure(12); plt.plot(g[0]); plt.plot(f2[0])

    shifts = pcc(g, f2, upsample_factor = upsampling, normalization = None)[0] * pixel * 0.5       

    return shifts[1]  # this is in mm