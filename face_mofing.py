import dlib
import cv2
import os
import sys
from imutils import face_utils

def Face_landmarks(image_path):
  print("[INFO] loading facial landmark predictor...")
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  image = cv2.imread(image_path)
  size = image.shape
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rects = detector(gray, 0)
  if len(rects) > 2:
    print("[ERR] too many faces fount...")
    # print("[Error] {} faces found...".format(len(rect)))
    sys.exit(1)
  if len(rects) < 1:
    print("[ERR] face not found...")
    # print("[Error] face not found...".format(len(rect))
    sys.exit(1)
  for rect in rects:
    (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
    print("[INFO] face frame {}".format(bX, bY, bW, bH))
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    points = shape.tolist()
    # (0,0),(x,0),(0,y),(x,y)
    points.append([0, 0])
    points.append([int(size[1]-1), 0])
    points.append([0, int(size[0]-1)])
    points.append([int(size[1]-1), int(size[0]-1)])
    # (x/2,0),(0,y/2),(x/2,y),(x,y/2)
    points.append([int(size[1]/2), 0])
    points.append([0, int(size[0]/2)])
    points.append([int(size[1]/2), int(size[0]-1)])
    points.append([int(size[1]-1), int(size[0]/2)])
    cv2.destroyAllWindows()
  return points


if __name__ == '__main__':
    filename = 'image/MarilynMonroe.jpg'
    name,ext = os.path.splitext(filename)
    img = cv2.imread(filename)
    points = Face_landmarks(filename)
    stringToWrite = ''
    for (i, (x, y)) in enumerate(points):
        print(x, y)
        stringToWrite += str(x) + " " + str(y) + "n"
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imwrite('%s-points.jpg' %name,img)

import argparse
import cv2
import numpy as np
import random


def Face_delaunay(rect, points1, points2, alpha):
    points = []
    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x, y))
    triangles, delaunay = calculateDelaunayTriangles(rect, points)
    cv2.destroyAllWindows()
    return triangles, delaunay


def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return triangleList, delaunayTri


def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True