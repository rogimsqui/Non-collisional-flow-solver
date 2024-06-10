import numpy as np
from stl import mesh

# Load STL file
mesh_data = mesh.Mesh.from_file('walls.stl')

# Extract vertices and create faces array
vertices = mesh_data.vectors.reshape(-1, 3, 3)
num_triangles = len(vertices)
faces = np.arange(num_triangles * 3).reshape(-1, 3)

# Compute normals
normals = np.zeros((len(faces), 3), dtype=np.double)
for i, face in enumerate(faces):
    v1 = vertices[face[2] // 3, face[2] % 3] - vertices[face[0] // 3, face[0] % 3]
    v2 = vertices[face[1] // 3, face[1] % 3] - vertices[face[0] // 3, face[0] % 3]
    normals[i] = 0.5 * np.cross(v1, v2)

print(normals)

