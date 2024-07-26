import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from src import model
from src import util
from src.body import Body
from src.hand import Hand
import clothe_affine_translation

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

print(f"Torch device: {torch.cuda.get_device_name()}")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
    ret, oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)

    new_joint_points = np.array([candidate[int(subset[0][4])][0:2],
                                 candidate[int(subset[0][3])][0:2],
                                 candidate[int(subset[0][2])][0:2],
                                 candidate[int(subset[0][1])][0:2],
                                 candidate[int(subset[0][5])][0:2],
                                 candidate[int(subset[0][6])][0:2],
                                 candidate[int(subset[0][7])][0:2],
                                 candidate[int(subset[0][8])][0:2],
                                 candidate[int(subset[0][11])][0:2],])
    # アフィン変換を実行
    affine_translation = clothe_affine_translation.Affine_translation()
    canvas = affine_translation.affine_translation(new_joint_points, canvas)

    cv2.imshow('demo', canvas)#一个窗口用以显示原视频
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

