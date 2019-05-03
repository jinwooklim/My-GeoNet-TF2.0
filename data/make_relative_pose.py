#-*- coding: utf-8 -*-
import os
import numpy as np

'''
dir= './pose_new'
result_dir= './relative_pose'
for category in os.listdir(dir):
    f= open(os.path.join(dir,category),'r')
    f1= open(os.path.join(result_dir,category),'w')
    lines= f.readlines()
    c=1
    cur = []
    d = []
    for one_line in lines:

        one_line=one_line.split(' ')[0:6]
        for i in range(len(one_line)):
            if  c!= 1:
                past = cur[i]
                cur[i] = float(one_line[i])
                d[i]=(cur[i]-past)
            elif c ==1:
                cur.append(float(one_line[i]))
                d.append(cur[i])
        if c!=1:
            for j in range(len(d)):
                f1.write(str(d[j]) + ' ')
            f1.write('\n')
        c=2
    f1.close()
    f.close()
'''


def convert(num_of_train, dump_dir, filename):
    idx = int(filename)
    
    raw_pose_txt = open(os.path.join(dump_dir, "%06d_pose.txt"%(idx)))
    raw_pose_data = raw_pose_txt.readlines()
    raw_pose_data = raw_pose_data[0].split(',')

    tx = float(raw_pose_data[0])
    ty = float(raw_pose_data[1])
    tz = float(raw_pose_data[2])
    pitch = float(raw_pose_data[3])
    yaw = float(raw_pose_data[4])
    roll = float(raw_pose_data[5])

    if(idx == 0 or idx-1 == 0):
        before_tx = tx
        before_ty = ty
        before_tz = tz
        before_pitch = pitch
        before_yaw = yaw
        before_roll = roll
    else:
        before_raw_pose_txt = open(os.path.join(dump_dir, "%06d_pose.txt"%(idx-1)))
        before_raw_pose_data = before_raw_pose_txt.readlines()
        before_raw_pose_data = before_raw_pose_data[0].split(',')
        before_tx = float(before_raw_pose_data[0])
        before_ty = float(before_raw_pose_data[1])
        before_tz = float(before_raw_pose_data[2])
        before_pitch = float(before_raw_pose_data[3])
        before_yaw = float(before_raw_pose_data[4])
        before_roll = float(before_raw_pose_data[5])
 
    rel_tx = tx - before_tx
    rel_ty = ty - before_ty
    rel_tz = tz - before_tz
    rel_pitch = pitch - before_pitch
    rel_yaw = yaw - before_yaw
    rel_roll = roll - before_roll

    return [rel_tx, rel_ty, rel_tz, rel_pitch, rel_yaw, rel_roll]
