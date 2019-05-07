import numpy as np
import os
import cv2
import math

seq = "02"

def mat2euler(M, cy_thresh=None, seq='zyx'):
    '''
    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
     threshold below which to give up on straightforward arctan for
     estimating x rotation.  If None (default), estimate from
     precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
     Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
     z = atan2(-r12, r11)
     y = asin(r13)
     x = atan2(-r23, r33)
    for x,y,z order
    y = asin(-r31)
    x = atan2(r32, r33)
    z = atan2(r21, r11)
    Problems arise when cos(y) is close to zero, because both of::
     z = atan2(cos(y)*sin(z), cos(y)*cos(z))
     x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if seq=='zyx':
        if cy > cy_thresh: # cos(y) not close to zero, standard form
            z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21,  r22)
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = 0.0
    elif seq=='xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi/2
                x = atan2(r12, r13)
            else:
                y = -np.pi/2
    else:
        raise Exception('Sequence not recognized')
    return z, y, x

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def getDisplay(position, scale=0.001, origin = [200, 650]):
    display = scale*position
    return (int(display[0]) + origin[0], -int(display[1]) + origin[1])

def odometry(map, position, direction, t, theta, color = (255, 255, 255), thickness=2, title = 'map', is_plot=True, wait=0):
    Rdirection = cv2.getRotationMatrix2D((0, 0), (180/math.pi)*direction[-1], 1)[:, 0:-1]
    position.append(position[-1] + np.matmul(Rdirection, t))
    direction.append(direction[-1] + theta)
    i = len(position)-1
    cv2.line(map, getDisplay(position[i - 1]), getDisplay(position[i]), color, thickness=thickness)
    cv2.line(map, getDisplay(position[-1]), getDisplay(position[-1] + np.matmul(Rdirection, np.array([0, 1000]))), color, thickness=thickness)
    if is_plot:
        cv2.imshow(title, map)
        cv2.waitKey(wait)
    return position, direction, map


predPathDir = os.path.join('C:\\Users\\jwlim\\PycharmProjects\\geonet_release\\predictions\\pose\\rigid_obd_in_posenet_seq%s'%seq)
gtPathDir = os.path.join('C:\\Users\\jwlim\\PycharmProjects\\geonet_release\\pose_gtruth\\pose_gtruth_snippets_seq%s'%seq)

fpred_list = os.listdir(predPathDir)
fgt_list = os.listdir(gtPathDir)

pred_list = []
for i in range(len(fpred_list)):
    f = open(os.path.join(predPathDir, fpred_list[i]))
    lines = f.readlines()
    data = lines[1].split()
    data = [float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7])] # tx, ty, tz
    pred_list.append(data)

gt_list = []
for i in range(len(fgt_list)):
    f = open(os.path.join(gtPathDir, fgt_list[i]))
    lines = f.readlines()
    data = lines[1].split()
    data = [float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7])] # tx, ty, tz
    gt_list.append(data)

pred_array = np.array(pred_list)
gt_array = np.array(gt_list)

whilte_background = np.ones([1000, 1000, 3], dtype=np.uint8) * 255
position = [np.ones([2], dtype=np.float32) * np.array([50, 950])]
direction = [np.array([0], dtype=np.float32)]
gt_position = [np.ones([2], dtype=np.float32) * np.array([50, 950])]
gt_direction = [np.array([0], dtype=np.float32)]

length = len(pred_list)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('odometry_%s.avi'%seq, fourcc, 30.0, (1000, 1000))

for i in range(length):
    # # GT
    gt_txtytz = gt_array[i, 0:3] * 2.0
    gt_arr=gt_array[i, 3:]
    gt_mat = quat2mat([gt_arr[3],gt_arr[0],gt_arr[1],gt_arr[2]]) #* (180/np.pi)
    gt_rxryrz = np.array(mat2euler(gt_mat))


    # # input
    txtytz = pred_array[i, 0:3]
    arr = pred_array[i, 3:]
    mat = quat2mat([arr[3], arr[0], arr[1], arr[2]])  # * (180/np.pi)
    rxryrz = np.array(mat2euler(mat))
    r_scale = np.sum(rxryrz * gt_rxryrz) / np.sum(rxryrz**2)
    scale = np.sum(txtytz * gt_txtytz) / np.sum(txtytz**2)


    gt_theta, gt_t = gt_rxryrz[1] * 1, 600 * np.array([1. * gt_txtytz[0], 1. * gt_txtytz[2]])
    theta, t = rxryrz[1] * 1, 600 * np.array([1. * txtytz[0], 1. * txtytz[2]])
    t = t * scale
    theta = theta * r_scale
    position, direction, whilte_background = odometry(whilte_background, position, direction, t, theta, color=(255, 0, 0), thickness=2, title='map', is_plot=True, wait=1)


    # print(position, direction)
    gt_position, gt_direction, whilte_background = odometry(whilte_background, gt_position, gt_direction, gt_t, gt_theta, color=(0, 0, 255), thickness=2, title='map', is_plot=True, wait=1)

    out.write(whilte_background)
out.release()