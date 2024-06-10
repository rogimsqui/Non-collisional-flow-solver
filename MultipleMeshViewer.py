import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def plot_mesh(ax, mesh, color):
    # Plot the mesh
    mesh_collection = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.3, facecolor=color, linewidths=0.5,
                                       edgecolors='darkblue')
    ax.add_collection3d(mesh_collection)

    # Adjust the axis limits
    scale = np.array([coord for vertex in mesh.vertices for coord in vertex]).flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=90, elev=0)

# Load the STL files
walls_mesh = trimesh.load_mesh("walls.stl")
inlet_mesh = trimesh.load_mesh("inlet.stl")
outlet_mesh = trimesh.load_mesh("outlet.stl")

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the meshes with different colors
plot_mesh(ax, walls_mesh, 'blue')   # Walls mesh in blue
plot_mesh(ax, inlet_mesh, 'green')  # Inlet mesh in green
plot_mesh(ax, outlet_mesh, 'red')   # Outlet mesh in red

plt.show()

