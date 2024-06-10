import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# Function to calculate particle reflection after collision
def calculate_reflection (direction, normals):
    # Reflect directions using the specular reflection formula
    reflections = direction - 2 * np.sum(direction * normals, axis=1)[:,None] * normals
    return reflections

    # Generate random reflections based on Lambert's cosine law (diffuse surfaces)
    '''
    theta = np.arcos(np.sqrt(np.random.rand(len(normals))))
    phi = 2 * np.pi * np.random.rand(len(normals))
    reflections = np.array([
        np.sin(theta) * np.cos(phi).
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    return reflections
    '''

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
        ax.quiver(origin[0], origin[1], origin[2], direction[0], direction[1], direction[2], color='r')
        ax.scatter(origin[0], origin[1], origin[2], color='blue', s=25)  # Blue point for ray origin

    # Plot reflections
    if reflections is not None:
        for origin, direction in zip(intersections, reflections):
            reflection_end = origin + direction * length
            ax.plot([origin[0], reflection_end[0]], [origin[1], reflection_end[1]], [origin[2], reflection_end[2]],
                    'b-', linewidth=0.5)  # Reflections are plotted in blue

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
num_rays = 10
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

# Calculate diffuse reflections
normals = mesh.face_normals[triangle_indices]
reflections = calculate_reflection(rays_directions[ray_indices], normals)

# Update ray origins and directions based on intersections and reflections
rays_origins[ray_indices] = locations
rays_directions[ray_indices] = reflections

# Visualize the mesh, the rays, and the intersection points
plot_mesh_and_rays(mesh, rays_origins, rays_directions, intersections=locations if len(locations) > 0 else None,
                    reflections=reflections)

