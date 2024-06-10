import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# Function to calculate specular reflection
def calculate_specular_reflection(directions, normals):
    reflections = directions - 2 * np.sum(directions * normals, axis=1)[:, None] * normals
    return reflections

# Function to calculate diffuse reflection
def calculate_diffuse_reflection(normals):
    theta = np.arccos(np.sqrt(np.random.rand(len(normals))))
    phi = 2 * np.pi * np.random.rand(len(normals))
    reflections = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ]).T
    return reflections

# Function to calculate new particle position for outlet surface
def generate_new_particle(inlet_mesh):
    inlet_faces = inlet_mesh.faces
    random_face = inlet_faces[np.random.randint(0, len(inlet_faces))]
    random_point = np.mean(inlet_mesh.vertices[random_face], axis=0)
    return random_point

# Function to plot the mesh, rays, and reflections
def plot_mesh_rays_and_reflections(mesh, rays_origins, rays_directions, intersections=None, reflections=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    mesh_collection = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.3, facecolor='cyan', linewidths=0.5,
                                       edgecolors='darkblue')
    ax.add_collection3d(mesh_collection)

    # Adjust the axis limits
    scale = np.array([coord for vertex in mesh.vertices for coord in vertex]).flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Plot all rays
    for origin, direction in zip(rays_origins, rays_directions):
        ray_end = origin + direction * 100  # Arbitrary extension for visualization
        ax.plot([origin[0], ray_end[0]], [origin[1], ray_end[1]], [origin[2], ray_end[2]], 'r-', linewidth=0.5)

    # Plot reflections
    if reflections is not None:
        for origin, direction in zip(intersections, reflections):
            reflection_end = origin + direction * 100
            ax.plot([origin[0], reflection_end[0]], [origin[1], reflection_end[1]], [origin[2], reflection_end[2]],
                    'b-', linewidth=0.5)

    # Plot intersections
    if intersections is not None:
        ax.scatter(intersections[:, 0], intersections[:, 1], intersections[:, 2], color='green', s=25)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=90, elev=0)
    plt.show()

# Generate or load the STL mesh with surface type information
# For simplicity, surface types are randomly assigned in this example
mesh = trimesh.creation.box((5, 5, 5))
mesh.visual.face_colors = np.random.randint(0, 255, size=(len(mesh.faces), 3))



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

# Access surface types 
# Això s'hauria de revisar, i veure si es pot assignar un atribut a cada cara que no tingui res a veure amb el seu color
surface_types = mesh.visual.face_colors

# Calculate reflections based on surface types
reflections = []
for i, surface_type in zip(ray_indices, surface_types[triangle_indices]):
    if surface_type[0] == 255:  # Specular surface
        normals = mesh.face_normals[triangle_indices]
        reflections.append(calculate_specular_reflection(rays_directions[i], normals[i]))
    elif surface_type[1] == 255:  # Diffuse surface
        normals = mesh.face_normals[triangle_indices]
        reflections.append(calculate_diffuse_reflection(normals[i]))
    elif surface_type[2] == 255:  # Outlet surface
        # Generate a new particle within an inlet surface
        inlet_mesh = trimesh.creation.box((5, 5, 5))
        new_particle = generate_new_particle(inlet_mesh)
        rays_origins[i] = new_particle
        reflections.append(np.random.rand(3) - 0.5)  # Random direction for visualization
    else:
        reflections.append(np.zeros(3))  # No reflection for other surfaces

# Update ray origins and directions based on intersections and reflections
rays_origins[ray_indices] = locations
rays_directions[ray_indices] = reflections

# Visualize the mesh, the rays, and the intersection points with reflections
plot_mesh_rays_and_reflections(mesh, rays_origins, rays_directions, intersections=locations if len(locations) > 0 else None, reflections=reflections)

