from scipy.spatial import Delaunay
import numpy as np
import cv2


class Born:
    def __init__(self, joint_point):
        # 0:右手首, 1:右ひじ, 2:右肩, 3:首, 4:左肩, 5:左ひじ, 6:左手首, 7:右腰, 8:左腰
        # ボーンの生成
        self.born = np.array([[joint_point[0], joint_point[1]],
                              [joint_point[1], joint_point[2]],
                              [joint_point[3], joint_point[3]],
                              [joint_point[3], joint_point[4]],
                              [joint_point[4], joint_point[5]],
                              [joint_point[5], joint_point[6]],
                              [joint_point[3], joint_point[7]],
                              [joint_point[3], joint_point[8]]])
        # ボーンを点で分割
        self.born_points = []
        for i in range(8):
            point = np.floor(np.linspace(start=self.born[i][0], stop=self.born[i][1], num=5)).astype(int)
            self.born_points.append(point)


def calculate_weights(contours, joint_points):
    weights = np.zeros((contours.shape[0], joint_points.shape[0]))
    for i, contour in enumerate(contours):
        for j, joint in enumerate(joint_points):
            distance = np.linalg.norm(contour - joint)
            weight = 1 / (distance + 1)
            weights[i, j] = weight
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    return weights

def apply_bone_movements(contours, joint_points, new_joint_points, weights):
    movements = new_joint_points - joint_points
    new_contours = contours + np.dot(weights, movements)
    return new_contours

def image_affine(src, dst, src_points, dst_points):
    # src_pointsとdst_pointsのバウンディングボックスを取得
    src_rect = cv2.boundingRect(src_points)
    dst_rect = cv2.boundingRect(dst_points.astype(np.float32))

    # srcとdstのクロップ領域を取得
    src_crop = src[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
    dst_crop = dst[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]]

    # クロップ領域内のポイントを計算
    src_pts_crop = src_points - src_rect[:2]
    dst_pts_crop = dst_points - dst_rect[:2]

    # アフィン変換行列を取得
    mat = cv2.getAffineTransform(src_pts_crop.astype(np.float32), dst_pts_crop.astype(np.float32))

    # アフィン変換を適用
    affine_img = cv2.warpAffine(src_crop, mat, (dst_crop.shape[1], dst_crop.shape[0]))

    # マスクを作成
    mask = np.zeros_like(dst_crop, dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts_crop.astype(np.int32), (1, 1, 1), cv2.LINE_AA)

    # アフィン変換後の画像をリサイズ
    affine_img = affine_img[:, :, :3]
    affine_img = cv2.resize(affine_img, (dst_crop.shape[1], dst_crop.shape[0]))

    # マスクを適用して画像をマージ
    dst_crop_merge = affine_img * mask + dst_crop * (1 - mask)

    # マージした画像を元の画像に戻す
    dst[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] = dst_crop_merge
    return dst


class Affine_translation:
    def __init__(self):
        # ボーンの生成関数
        self.img = cv2.imread('long_clothe.png', cv2.IMREAD_UNCHANGED)

        self.contours, hierarchy = cv2.findContours(self.img[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = np.vstack([contour.squeeze() for contour in self.contours])

        # 0:右手首, 1:右ひじ, 2:右肩, 3:首, 4:左肩, 5:左ひじ, 6:左手首, 7:右腰, 8:左腰
        self.joint_points = np.array([[47, 284], [77, 178], [105, 54], [184, 32], [263, 54], [294, 167], [328, 278], [129, 292], [249, 287]])

        # ボーンの生成
        born = Born(self.joint_points)

        # ボーンと輪郭点の結合
        for born_point in born.born_points:
            for point in born_point:
                self.contours = np.vstack([self.contours, point])

        self.tri = Delaunay(self.contours)
        i = 0
        for simplex in self.tri.simplices:
            pt1 = tuple(self.contours[simplex[0]])
            pt2 = tuple(self.contours[simplex[1]])
            pt3 = tuple(self.contours[simplex[2]])
            if self.img[int((pt1[1] + pt2[1] + pt3[1]) / 3)][int((pt1[0] + pt2[0] + pt3[0]) / 3)][3] == 0:
                self.tri.simplices = np.delete(self.tri.simplices, i, axis=0)
                continue
            else:
                i += 1
    def affine_translation(self, new_joint_points,canvas):
        # 動きの算出
        weights = calculate_weights(self.contours, self.joint_points)
        new_contours = apply_bone_movements(self.contours, self.joint_points, new_joint_points, weights)

        for i, simplex in enumerate(self.tri.simplices):
            pt1 = (int(new_contours[simplex[0]][0]), int(new_contours[simplex[0]][1]))
            pt2 = (int(new_contours[simplex[1]][0]), int(new_contours[simplex[1]][1]))
            pt3 = (int(new_contours[simplex[2]][0]), int(new_contours[simplex[2]][1]))
            canvas = image_affine(self.img, canvas, np.array([self.contours[i] for i in simplex]), np.array([new_contours[i] for i in simplex]))

        print(self.img)
        return canvas
if __name__ == '__main__':
    affine_translation = Affine_translation()
    affine_translation.affine_translation()
