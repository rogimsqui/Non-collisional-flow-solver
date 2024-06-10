import numpy as np

conec = np.array([
    [1, 2, 4], [2, 5, 4], [2, 3, 5], [3, 6, 5],
    [1, 4, 7]
])

xyz = np.array([
    [-25, -15, 0], [-5, -15, 0], [15, -15, 0],
    [-15, 15, 0], [5, 15, 0], [25, 15, 0],
    [-25, 5, 0]
])

# Subtract 1 from conec indices to get 0-based indices
conec -= 1

# Compute normals, loop by element
normals = np.zeros((conec.shape[0], 3), dtype=np.double)
for iel, c in enumerate(conec):
    v1 = xyz[c[2], :] - xyz[c[0], :]
    v2 = xyz[c[1], :] - xyz[c[0], :]
    normals[iel, :] = 0.5 * np.cross(v1, v2)

print(normals)

