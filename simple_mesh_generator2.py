import numpy as np
from stl import mesh

# Define vertices and faces for a simple triangular mesh
vertices = np.array([
    [-25, -15, 0], [-5, -15, 0], [15, -15, 0],
    [-15, 15, 0], [5, 15, 0], [25, 15, 0],
    [-25, 5, 0]    # Additional vertex for the 5th triangle
])

faces = np.array([
    [1, 2, 4], [2, 5, 4], [2, 3, 5], [3, 6, 5],
    [1, 4, 7]    # Face for the 5th triangle
])

# Create a mesh object
simple_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, face in enumerate(faces):
    for j in range(3):
        simple_mesh.vectors[i][j] = vertices[face[j]-1]

# Write the mesh to an STL file
simple_mesh.save('simple_mesh2.stl')

