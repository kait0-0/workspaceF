import cv2
import matplotlib.pyplot as plt
import copy

import numpy
import numpy as np
import torch

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

print(f"Torch device: {torch.cuda.get_device_name()}")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
parts = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist",
         "RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar"]

def cvpaste(img, imgback, x, y, angle, scale):
    # x and y are the distance from the center of the background image
    r = img.shape[0]
    # r = int(np.linalg.norm(r_wr_bector)) * scale
    c = img.shape[1]
    rb = imgback.shape[0]
    cb = imgback.shape[1]
    hrb=round(rb/2)
    hcb=round(cb/2)
    hr=round(r/2)
    hc=round(c/2)

    # Copy the forward image and move to the center of the background image
    imgrot = np.zeros((rb,cb,3),np.uint8)

    img_resized = cv2.resize(img, (hc * 2, hr * 2 - 1))
    imgrot[hrb - hr:hrb + hr - 1, hcb - hc:hcb + hc, :] = img_resized

    # Rotation and scaling
    M = cv2.getRotationMatrix2D((hcb,hrb),angle,scale)
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))
    # Translation
    M = np.float32([[1,0,x],[0,1,y]])
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))

    # Makeing mask
    imggray = cv2.cvtColor(imgrot,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imggray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of the forward image in the background image
    img1_bg = cv2.bitwise_and(imgback,imgback,mask = mask_inv)

    # Take only region of the forward image.
    img2_fg = cv2.bitwise_and(imgrot,imgrot,mask = mask)

    # Paste the forward image on the background image
    imgpaste = cv2.add(img1_bg,img2_fg)
    return imgpaste


while True:
    ret, oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    try:
        r_wr_bector = np.array([candidate[int(subset[0][4])][0] - candidate[int(subset[0][3])][0],
                            candidate[int(subset[0][4])][1] - candidate[int(subset[0][3])][1]])
        l_wl_bector = np.array([candidate[int(subset[0][6])][0] - candidate[int(subset[0][7])][0],
                                candidate[int(subset[0][6])][1] - candidate[int(subset[0][7])][1]])
        r_sr_bector = np.array([candidate[int(subset[0][3])][0] - candidate[int(subset[0][2])][0],
                            candidate[int(subset[0][3])][1] - candidate[int(subset[0][2])][1]])
        l_sl_bector = np.array([candidate[int(subset[0][5])][0] - candidate[int(subset[0][6])][0],
                            candidate[int(subset[0][5])][1] - candidate[int(subset[0][6])][1]])
        r_kr_bector = np.array([candidate[int(subset[0][9])][0] - candidate[int(subset[0][8])][0],
                                candidate[int(subset[0][9])][1] - candidate[int(subset[0][8])][1]])
        l_kl_bector = np.array([candidate[int(subset[0][11])][0] - candidate[int(subset[0][12])][0],
                                candidate[int(subset[0][11])][1] - candidate[int(subset[0][12])][1]])
        r_ar_bector = np.array([candidate[int(subset[0][10])][0] - candidate[int(subset[0][9])][0],
                                candidate[int(subset[0][10])][1] - candidate[int(subset[0][9])][1]])
        l_al_bector = np.array([candidate[int(subset[0][12])][0] - candidate[int(subset[0][13])][0],
                                candidate[int(subset[0][12])][1] - candidate[int(subset[0][13])][1]])
        body_bector = np.array([candidate[int(subset[0][2])][0] - candidate[int(subset[0][5])][0],
                                candidate[int(subset[0][2])][1] - candidate[int(subset[0][5])][1]])



    except IndexError:
        continue
    y_bector = np.array([1, 0])
    x_bector = np.array([0, -1])
    r_ratio = np.arccos(r_wr_bector @ x_bector / (np.linalg.norm(r_wr_bector) * np.linalg.norm(x_bector))) * 180 / np.pi
    l_ratio = np.arccos(l_wl_bector @ x_bector / (np.linalg.norm(l_wl_bector) * np.linalg.norm(x_bector))) * 180 / np.pi + 180

    rs_ratio = np.arccos(r_sr_bector @ x_bector / (np.linalg.norm(r_sr_bector) * np.linalg.norm(x_bector))) * 180 / np.pi
    ls_ratio = np.arccos(l_sl_bector @ x_bector / (np.linalg.norm(l_sl_bector) * np.linalg.norm(x_bector))) * 180 / np.pi + 180

    rk_ratio = np.arccos(r_kr_bector @ x_bector / (np.linalg.norm(r_kr_bector) * np.linalg.norm(x_bector))) * 180 / np.pi
    lk_ratio = np.arccos(l_kl_bector @ x_bector / (np.linalg.norm(l_kl_bector) * np.linalg.norm(x_bector))) * 180 / np.pi + 180

    ra_ratio = np.arccos(r_ar_bector @ x_bector / (np.linalg.norm(r_ar_bector) * np.linalg.norm(x_bector))) * 180 / np.pi
    la_ratio = np.arccos(l_al_bector @ x_bector / (np.linalg.norm(l_al_bector) * np.linalg.norm(x_bector))) * 180 / np.pi + 180

    body_ratio = np.arccos(body_bector @ x_bector / (np.linalg.norm(body_bector) * np.linalg.norm(x_bector))) * 180 / np.pi - 90
#左右の手首から肘まで
    r_x = int((candidate[int(subset[0][3])][0] + candidate[int(subset[0][4])][0]) / 2) - 320
    r_y = int((candidate[int(subset[0][3])][1] + candidate[int(subset[0][4])][1]) / 2) - 240
    l_x = int((candidate[int(subset[0][6])][0] + candidate[int(subset[0][7])][0]) / 2) - 320
    l_y = int((candidate[int(subset[0][6])][1] + candidate[int(subset[0][7])][1]) / 2) - 240
#左右の肘から肩まで
    r_s_x = int((candidate[int(subset[0][2])][0] + candidate[int(subset[0][3])][0]) / 2) - 320
    r_s_y = int((candidate[int(subset[0][2])][1] + candidate[int(subset[0][3])][1]) / 2) - 240
    l_s_x = int((candidate[int(subset[0][5])][0] + candidate[int(subset[0][6])][0]) / 2) - 320
    l_s_y = int((candidate[int(subset[0][5])][1] + candidate[int(subset[0][6])][1]) / 2) - 240
#左右の腰から膝まで
    r_k_x = int((candidate[int(subset[0][8])][0] + candidate[int(subset[0][9])][0]) / 2) - 320
    r_k_y = int((candidate[int(subset[0][8])][1] + candidate[int(subset[0][9])][1]) / 2) - 240
    l_k_x = int((candidate[int(subset[0][11])][0] + candidate[int(subset[0][12])][0]) / 2) - 320
    l_k_y = int((candidate[int(subset[0][11])][1] + candidate[int(subset[0][12])][1]) / 2) - 240
#左右の膝から足首まで
    r_a_x = int((candidate[int(subset[0][9])][0] + candidate[int(subset[0][10])][0]) / 2) - 320
    r_a_y = int((candidate[int(subset[0][9])][1] + candidate[int(subset[0][10])][1]) / 2) - 240
    l_a_x = int((candidate[int(subset[0][12])][0] + candidate[int(subset[0][13])][0]) / 2) - 320
    l_a_y = int((candidate[int(subset[0][12])][1] + candidate[int(subset[0][13])][1]) / 2) - 240
#胴体
    body_x = int((candidate[int(subset[0][2])][0] + candidate[int(subset[0][11])][0]) / 2) - 320
    body_y = int((candidate[int(subset[0][8])][1] + candidate[int(subset[0][1])][1]) /2) - 240


    if (len(subset) > 0):
        if int(subset[0][1]) != -1 and int(subset[0][2]) != -1 and int(subset[0][5]) != -1 and int(subset[0][8]) != -1:
            img = cv2.imread(r"body.PNG")
            if img.shape[0] > 640 or img.shape[1] > 480:
                img = cv2.resize(img, (200, int(img.shape[1] * 350 / img.shape[0])), interpolation=cv2.INTER_AREA)
            body_angle = body_ratio
            body_scale = float(np.linalg.norm(body_bector)) / img.shape[1]
            canvas = cvpaste(img, canvas, body_x, body_y, body_angle, body_scale)
        if int(subset[0][3]) != -1 and int(subset[0][4]) != -1:  #右手首から肘まで
            img = cv2.imread(r"right-arm.PNG")
            if img.shape[0] > 640 or img.shape[1] > 480:
                img = cv2.resize(img, (200, int(img.shape[1] * 350 / img.shape[0])), interpolation=cv2.INTER_AREA)
            r_angle = r_ratio
            r_scale = float(np.linalg.norm(r_wr_bector)) / img.shape[0]
            canvas = cvpaste(img, canvas, r_x, r_y, r_angle, r_scale)
        if int(subset[0][6]) != -1 and int(subset[0][7]) != -1:  #左手首から肘まで
            img = cv2.imread(r"left-arm.JPG")
            if img.shape[0] > 640 or img.shape[1] > 480:
                img = cv2.resize(img, (200, int(img.shape[1] * 350 / img.shape[0])), interpolation=cv2.INTER_AREA)
            l_angle = l_ratio
            l_scale = float(np.linalg.norm(l_wl_bector)) / img.shape[0]
            canvas = cvpaste(img, canvas, l_x, l_y, l_angle, l_scale)
        if int(subset[0][2]) != -1 and int(subset[0][3]) != -1:  #右肘から肩まで
            img = cv2.imread(r"right-s.PNG")
            if img.shape[0] > 640 or img.shape[1] > 480:
                img = cv2.resize(img, (200, int(img.shape[1] * 350 / img.shape[0])), interpolation=cv2.INTER_AREA)
            rs_angle = rs_ratio
            rs_scale = float(np.linalg.norm(r_sr_bector)) / img.shape[0]
            canvas = cvpaste(img, canvas, r_s_x, r_s_y, rs_angle, rs_scale)
        if int(subset[0][5]) != -1 and int(subset[0][6]) != -1:  #左肘から肩まで
            img = cv2.imread(r"left-s.PNG")
            if img.shape[0] > 640 or img.shape[1] > 480:
                img = cv2.resize(img, (200, int(img.shape[1] * 350 / img.shape[0])), interpolation=cv2.INTER_AREA)
            ls_angle = ls_ratio
            ls_scale = float(np.linalg.norm(l_sl_bector)) / img.shape[0]
            canvas = cvpaste(img, canvas, l_s_x, l_s_y, ls_angle, ls_scale)
        if int(subset[0][8]) != -1 and int(subset[0][9]) != -1:  #右腰から膝まで
            img = cv2.imread(r"right-k.PNG")
            if img.shape[0] > 640 or img.shape[1] > 480:
                img = cv2.resize(img, (200, int(img.shape[1] * 350 / img.shape[0])), interpolation=cv2.INTER_AREA)
            rk_angle = rk_ratio
            rk_scale = float(np.linalg.norm(r_kr_bector)) / img.shape[0]
            canvas = cvpaste(img, canvas, r_k_x, r_k_y, rk_angle, rk_scale)

        if int(subset[0][10]) != -1 and int(subset[0][11]) != -1:  #左腰から膝まで
            img = cv2.imread(r"left-k.PNG")
            if img.shape[0] > 640 or img.shape[1] > 480:
                img = cv2.resize(img, (200,int(img.shape[1]*350/img.shape[0])), interpolation=cv2.INTER_AREA)
            lk_angle = lk_ratio
            lk_scale = float(np.linalg.norm(l_kl_bector)) / img.shape[0]
            canvas = cvpaste(img, canvas, l_k_x, l_k_y, lk_angle, lk_scale)


        if int(subset[0][11]) != -1 and int(subset[0][12]) != -1 and candidate[int(subset[0][8])][1] > candidate[int(subset[0][11])][1]:   #右膝から足首まで
            img = cv2.imread(r"right-a.png")
            if img.shape[0] > 640 or img.shape[1] > 480:
                img = cv2.resize(img, (200, int(img.shape[1] * 350 / img.shape[0])), interpolation=cv2.INTER_AREA)
            ra_angle = ra_ratio
            ra_scale = float(np.linalg.norm(r_ar_bector)) / img.shape[0]
            canvas = cvpaste(img, canvas, r_a_x, r_a_y, ra_angle, ra_scale)


        if int(subset[0][12]) != -1 and int(subset[0][13]) != -1 and candidate[int(subset[0][8])][1] > candidate[int(subset[0][12])][1]:   #左膝から足首まで
            img = cv2.imread(r"left-a.png")
            if img.shape[0] > 640 or img.shape[1] > 480:
                img = cv2.resize(img, (200, int(img.shape[1] * 350 / img.shape[0])), interpolation=cv2.INTER_AREA)
            la_angle = la_ratio
            la_scale = float(np.linalg.norm(l_al_bector)) / img.shape[0]
            canvas = cvpaste(img, canvas, l_a_x, l_a_y, la_angle, la_scale)



    cv2.imshow('demo', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()