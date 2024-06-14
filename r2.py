import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def random_point_in_triangle(v1, v2, v3):
    """Generate a random point within a triangle defined by its vertices."""
    s = np.random.uniform(0, 1, 2)
    sqrt_s0 = np.sqrt(s[0])
    u = 1 - sqrt_s0
    v = s[1] * sqrt_s0
    return (u * v1 + v * v2 + (1 - u - v) * v3)

def random_point_in_mesh(mesh):
    """Generate a random point within a trimesh object."""
    # Calculate the surface area of each triangle
    areas = mesh.area_faces

    # Select a triangle with a probability proportional to its area
    selected_face_idx = np.random.choice(len(areas), p=areas/np.sum(areas))

    # Get the vertices of the selected triangle
    v1, v2, v3 = mesh.vertices[mesh.faces[selected_face_idx]]

    # Generate a random point within the selected triangle
    point = random_point_in_triangle(v1, v2, v3)

    return point

# Load the .stl file
mesh = trimesh.load_mesh('simple_mesh2.stl')

# Generate 1000 random points within the mesh
random_points = [random_point_in_mesh(mesh) for _ in range(5000)]

# Function to plot the mesh and random points
def plot_mesh_and_points(mesh, random_points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    mesh_collection = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.3, facecolor='cyan', linewidths=0.5,
                                       edgecolors='darkblue')
    ax.add_collection3d(mesh_collection)

    # Plot the random points
    random_points = np.array(random_points)
    ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], color='r', s=20, label='Random Points')

    # Adjust the axis limits
    scale = np.array([coord for vertex in mesh.vertices for coord in vertex]).flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Mesh with Random Points')
    plt.show()

# Plot the mesh and the random points
plot_mesh_and_points(mesh, random_points)

