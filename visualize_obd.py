import cv2
import numpy as np
import os
import math

def get_pose_by_obd(obd):
    fps = 9.64764  # 10.0
    alpha = 1.0 / (3.6 * fps)
    beta = 0.000901
    gamma = 0.0
    delta = 0.05 * math.pi / 180.0 * -0.0062
    L_car = 2.7
    bias_steering = 0.0
    angle_scale = 540.0 / 35.0

    seq_result = []
    a = obd[1] / angle_scale * np.pi / 180.0 + bias_steering
    s = obd[0] * alpha

    theta = obd[1] / angle_scale / L_car * s * np.pi / 180.0

    if theta < 0.000001:
        theta = 0.000001

    phy = 0.0
    tx = -2.0 / theta * np.sin(theta / 2.0) * s * np.sin(a + theta / 2.0 + phy)
    tz = 2.0 / theta * np.sin(theta / 2.0) * s * np.cos(a + theta / 2.0 + phy)

    # seq_result.append([tx, 0.0, tz, 0.0, theta, 0.0])
    # seq_result = np.stack(seq_result)
    return tx, tz, theta

img = cv2.imread('./etc/steering_wheel.png', cv2.IMREAD_UNCHANGED)
trans_mask = img[:,:,3] == 0
img[trans_mask] = [255, 255, 255, 255]
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


rows, cols = img.shape[:2]
obd_text = np.genfromtxt('E:\\all_dataset\\KITTI_dump\\virtual_obd\\09.txt', delimiter=' ')
rows, cols = img.shape[:2]

blank = np.ones((700, 1500, 3), dtype=np.uint8)
blank = blank * 255

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (2200, 700))

for i, data in enumerate(obd_text):
    tx, tz, theta = get_pose_by_obd(data)
    speed = data[0]
    angle = data[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)

    # dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_TRANSPARENT, borderValue=(255, 255, 255))
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

    dst = np.concatenate((dst, blank), axis=1)

    org = ((int)(cols/2)+650, (int)(rows/2)-100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 3.5
    thickness = 4
    green = (0, 255, 0)
    str_speed = "RPM : " + str(speed) + " "
    cv2.putText(dst, str_speed, org, font, fontscale, green, thickness, cv2.LINE_AA)

    org = ((int)(cols/2)+650, (int)(rows/2)+25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 3.5
    thickness = 4
    red = (0, 0, 255)
    str_angle = "Degree : " + str(angle)
    cv2.putText(dst, str_angle, org, font, fontscale, red, thickness, cv2.LINE_AA)



    org = ((int)(cols/2)+650, (int)(rows/2)+135)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1.5
    thickness = 4
    red = (255, 0, 0)
    str_tx = "tx : " + str(tx)
    cv2.putText(dst, str_tx, org, font, fontscale, red, thickness, cv2.LINE_AA)

    org = ((int)(cols/2)+650, (int)(rows/2)+185)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1.5
    thickness = 4
    red = (255, 0, 0)
    str_tz = "tz : " + str(tz)
    cv2.putText(dst, str_tz, org, font, fontscale, red, thickness, cv2.LINE_AA)

    org = ((int)(cols/2)+650, (int)(rows/2)+235)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1.5
    thickness = 4
    red = (255, 0, 0)
    str_theta = "theta : " + str(theta)
    cv2.putText(dst, str_theta, org, font, fontscale, red, thickness, cv2.LINE_AA)

    cv2.imshow('Sequence 09', dst)
    cv2.waitKey(0) # 33
    # out.write(dst)

out.release()
cv2.destroyAllWindows()