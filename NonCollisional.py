import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# Function to plot the mesh and rays
def plot_mesh_and_rays(mesh, rays_origins, rays_directions, intersections=None, reflections=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ray extension
    length = 100

    # Plot the mesh
    mesh_collection = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.3, facecolor='cyan', linewidths=0.5,
                                       edgecolors='darkblue')
    ax.add_collection3d(mesh_collection)

    # Adjust the axis limits
    scale = np.array([coord for vertex in mesh.vertices for coord in vertex]).flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Plot all rays
    for origin, direction in zip(rays_origins, rays_directions):
        ray_end = origin + direction * length  # Arbitrary extension for visualization
        ax.plot([origin[0], ray_end[0]], [origin[1], ray_end[1]], [origin[2], ray_end[2]], 'r-', linewidth=0.5)
    
    # Plot intersections
    if intersections is not None:
        ax.scatter(intersections[:, 0], intersections[:, 1], intersections[:, 2], color='green', s=25)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=90, elev=0)
    plt.show()


# Load the STL file
mesh = trimesh.load_mesh("simple_mesh.stl")

# Generate 100 random rays
num_rays = 100
rays_origins = np.random.rand(num_rays, 3) * 10 - 5  # Random origins around the mesh
rays_directions = np.random.rand(num_rays, 3) - 0.5  # Random directions
rays_directions = rays_directions / np.linalg.norm(rays_directions, axis=1)[:, None]  # Normalize directions

# Perform ray-mesh intersection for all rays at once
locations, ray_indices, triangle_indices = mesh.ray.intersects_location(
    ray_origins=rays_origins,
    ray_directions=rays_directions
)

print(locations)
print(ray_indices)
print(triangle_indices)

# Visualize the mesh, the rays, and the intersection points
plot_mesh_and_rays(mesh, rays_origins, rays_directions, intersections=locations if len(locations) > 0 else None)
