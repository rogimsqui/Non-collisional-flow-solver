import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# Generate random reflections based on Lambert's cosine law (diffuse surfaces)
def calculate_reflection(direction, normals):
    theta = np.arccos(np.sqrt(np.random.rand(len(normals))))
    phi = 2 * np.pi * np.random.rand(len(normals))
    reflectionsT = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    reflections = np.transpose(reflectionsT)

    # Ensure the angle between the normal and the reflection is less than 90 degrees
    dot_product = np.einsum('ij,ij->i', reflections, normals)
    for i in range(len(reflections)):
        if dot_product[i] <= 0:
            reflections[i] = -reflections[i]  # Reflect in the opposite direction

    return reflections

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
        ray_end = origin + direction * length *10  # Arbitrary extension for visualization
        ax.plot([origin[0], ray_end[0]], [origin[1], ray_end[1]], [origin[2], ray_end[2]], 'r-', linewidth=0.5)

 # Plot reflections
    if reflections is not None:
        for origin, direction in zip(intersections, reflections):
            reflection_end = origin + direction * length
            ax.plot([origin[0], reflection_end[0]], [origin[1], reflection_end[1]], [origin[2], reflection_end[2]], 'b-', linewidth=0.5)  # Reflections are plotted in blue
    
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

# Set origin and direction for all rays
num_rays = 10
ray_origin = np.array([0, 0, -5])  # Origin for all rays
ray_direction = np.array([1, 1, 1])  # Direction for all rays

# Generate ray origins and directions
rays_origins = np.tile(ray_origin, (num_rays, 1))  # Tile the origin to create an array of the same shape as num_rays
rays_directions = np.tile(ray_direction, (num_rays, 1))  # Tile the direction similarly

# Perform ray-mesh intersection for all rays at once
locations, ray_indices, triangle_indices = mesh.ray.intersects_location(
    ray_origins=rays_origins,
    ray_directions=rays_directions
)

# Calculate diffuse reflections
normals = mesh.face_normals[triangle_indices]
reflections = calculate_reflection(rays_directions[ray_indices], normals)

# Update ray origins and directions based on intersections and reflections
rays_origins[ray_indices] = locations
rays_directions[ray_indices] = reflections

# Visualize the mesh, the rays, and the intersection points
plot_mesh_and_rays(mesh, rays_origins, rays_directions, intersections=locations if len(locations) > 0 else None,
                    reflections=reflections)

