import numpy as np

# メッシュの頂点データ
vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0]
])

# ボーンの位置（この例では単純化のため2Dで表現）
bone_positions = np.array([
    [0.5, 0.5]
])

# 各頂点に対するウェイトの初期化
weights = np.zeros((vertices.shape[0], bone_positions.shape[0]))

# ウェイトペインティングのシミュレーション
# ここでは、ボーンからの距離に基づいてウェイトを計算します
for i, vertex in enumerate(vertices):
    for j, bone_pos in enumerate(bone_positions):
        # 2D距離を計算
        distance = np.linalg.norm(vertex[:2] - bone_pos)
        # 距離に基づいてウェイトを割り当てる（単純化のため）
        weight = 1 / (distance + 1)
        weights[i, j] = weight

# ウェイトの正規化
weights /= np.sum(weights, axis=1)[:, np.newaxis]

print("Vertices:")
print(vertices)
print("Weights:")
print(weights)