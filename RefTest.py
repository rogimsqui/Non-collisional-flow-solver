# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the vertices and faces of a simple flat mesh with 2 triangles
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0]
])

faces = np.array([
    [0, 1, 2],
    [1, 3, 2]
])

# Define the ray origins and directions
ray_origins = np.array([
    [0.5, 0.5, 1],
    [0.2, 0.2, 1],
    [0.8, 0.8, 1]
])

ray_directions = np.array([
    [0, 0, -1],
    [0.5, 0.5, -1],
    [-0.5, -0.5, -1]
])

# Function to calculate reflection
def calculate_reflection(direction, normals):
    theta = np.arccos(np.sqrt(np.random.rand(len(normals))))
    phi = 2 * np.pi * np.random.rand(len(normals))
    
    reflectionsT = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    reflections = np.transpose(reflectionsT)
    return reflections

# Function to plot mesh and rays
def plot_mesh_and_rays(vertices, faces, ray_origins, ray_directions, reflections):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh
    mesh_collection = Poly3DCollection(vertices[faces], alpha=0.3, facecolor='blue', linewidths=0.5, edgecolors='darkblue')
    ax.add_collection3d(mesh_collection)
    
    # Plot rays
    for origin, direction in zip(ray_origins, ray_directions):
        ray_end = origin + direction * 2  # Extend the ray for visualization
        ax.plot([origin[0], ray_end[0]], [origin[1], ray_end[1]], [origin[2], ray_end[2]], 'r-', linewidth=0.5)
    
    # Plot reflections
    for origin, reflection in zip(ray_origins, reflections):
        reflection_end = origin + reflection * 2  # Extend the reflection for visualization
        ax.plot([origin[0], reflection_end[0]], [origin[1], reflection_end[1]], [origin[2], reflection_end[2]], 'b-', linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=90, elev=0)
    plt.show()

# Calculate reflections
normals = np.array([[0, 0, 1], [0, 0, 1]])  # Assuming both triangles are flat and facing up
reflections = calculate_reflection(ray_directions, normals)

# Plot mesh and rays
plot_mesh_and_rays(vertices, faces, ray_origins, ray_directions, reflections)

