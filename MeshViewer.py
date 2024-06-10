import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def plot_mesh(mesh):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    mesh_collection = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.3, facecolor='cyan', linewidths=0.5,
                                       edgecolors='darkblue')
    ax.add_collection3d(mesh_collection)

    # Adjust the axis limits
    scale = np.array([coord for vertex in mesh.vertices for coord in vertex]).flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=90, elev=0)
    plt.show()

# Load the STL file
mesh = trimesh.load_mesh("Intake.stl")
# Visualize the mesh
plot_mesh(mesh)




