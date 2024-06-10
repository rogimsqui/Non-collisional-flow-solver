import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# Function to generate a random point within a triangle
def random_point_in_triangle(v1, v2, v3):
    s = np.random.uniform(0, 1, 2)
    sqrt_s0 = np.sqrt(s[0])
    u = 1 - sqrt_s0
    v = s[1] * sqrt_s0
    return (u * v1 + v * v2 + (1 - u - v) * v3)

# Function to generate a random point within a trimesh object
def random_point_in_mesh(mesh):
    areas = mesh.area_faces
    selected_face_idx = np.random.choice(len(areas), p=areas/np.sum(areas))
    v1, v2, v3 = mesh.vertices[mesh.faces[selected_face_idx]]
    point = random_point_in_triangle(v1, v2, v3)
    return point

# Function to calculate particle reflection after collision
def calculate_reflection(direction, normals):
    reflections = direction - 2 * np.sum(direction * normals, axis=1)[:, None] * normals
    return reflections

# Function to plot the mesh and rays
def plot_mesh_and_rays(mesh, rays_origins, rays_directions, intersections=None, reflections=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ray extension
    length = 12

    # Plot the meshes
    mesh_collection_walls = Poly3DCollection(mesh['walls'].vertices[mesh['walls'].faces], alpha=0.3, facecolor='blue', linewidths=0.5,
                                       edgecolors='darkblue')
    mesh_collection_inlet = Poly3DCollection(mesh['inlet'].vertices[mesh['inlet'].faces], alpha=0.3, facecolor='green', linewidths=0.5,
                                       edgecolors='darkblue')
    mesh_collection_outlet = Poly3DCollection(mesh['outlet'].vertices[mesh['outlet'].faces], alpha=0.3, facecolor='red', linewidths=0.5,
                                       edgecolors='darkblue')
    ax.add_collection3d(mesh_collection_walls)
    ax.add_collection3d(mesh_collection_inlet)
    ax.add_collection3d(mesh_collection_outlet)

    # Plot all rays
    for i, (origin, direction) in enumerate(zip(rays_origins, rays_directions)):
        if i in intersections:  # Check if the ray intersects a mesh
            continue
        ray_end = origin + direction * length  # Arbitrary extension for visualization
        ax.plot([origin[0], ray_end[0]], [origin[1], ray_end[1]], [origin[2], ray_end[2]], 'r-', linewidth=0.5)

    # Plot reflections
    if reflections is not None:
        for origin, direction in zip(rays_origins[intersections], reflections):
            reflection_end = origin + direction * length
            ax.plot([origin[0], reflection_end[0]], [origin[1], reflection_end[1]], [origin[2], reflection_end[2]], 'b-', linewidth=0.5)  # Reflections are plotted in blue
    
    # Plot intersections
    if intersections is not None:
        ax.scatter(rays_origins[intersections][:, 0], rays_origins[intersections][:, 1], rays_origins[intersections][:, 2], color='green', s=5)  # Intersection points are plotted smaller

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=90, elev=0)
    plt.show()

# Function to count the number of rays reflecting on each triangle of the "walls.stl" mesh
def count_reflections_per_triangle(mesh, ray_origins, ray_directions, intersections, reflections):
    triangle_counts = np.zeros(len(mesh.faces))
    
    for i, (origin, direction) in enumerate(zip(ray_origins, ray_directions)):
        if i in intersections:
            continue
        
        locations, _, triangle_indices = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        
        if len(locations) > 0:
            triangle_counts[triangle_indices[0]] += 1
            
    return triangle_counts

# Function to count the number of rays crossing through each triangle of the "outlet.stl" mesh
def count_crossings_per_triangle(mesh, ray_origins, ray_directions):
    triangle_counts = np.zeros(len(mesh.faces))
    
    for i, (origin, direction) in enumerate(zip(ray_origins, ray_directions)):
        locations, _, triangle_indices = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        
        if len(locations) > 0:
            triangle_counts[triangle_indices[0]] += 1
            
    return triangle_counts

# Load the STL files
walls_mesh = trimesh.load_mesh("walls.stl")
inlet_mesh = trimesh.load_mesh("inlet.stl")
outlet_mesh = trimesh.load_mesh("outlet.stl")

meshes = {
    'walls': walls_mesh,
    'inlet': inlet_mesh,
    'outlet': outlet_mesh
}

# Generate 100 random rays with origins at the "inlet.stl" surfaces
ray_origins = np.array([random_point_in_mesh(inlet_mesh) for _ in range(100)])
satellite_velocity = np.array([7800, 0, 0])  # Satellite velocity

# Calculate the speed of gas particles
R = 8.31446  # Ideal gas constant
particle_speed = np.sqrt(3 * R * 900 / 28)

# Generate random directions for the particle speed vector
random_directions = np.random.normal(size=(100, 3))
random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]

# Calculate ray directions
ray_directions = satellite_velocity + particle_speed * random_directions
ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, None]  # Normalize the ray directions

# Perform ray-mesh intersection for all rays at once
locations, ray_indices, triangle_indices = meshes['walls'].ray.intersects_location(
    ray_origins=ray_origins,
    ray_directions=ray_directions
)

# Calculate diffuse reflections
normals = meshes['walls'].face_normals[triangle_indices]
reflections = calculate_reflection(ray_directions[ray_indices], normals)

# Update ray origins and directions based on intersections and reflections
if len(ray_indices) > 0:
    ray_origins[ray_indices] = locations
    ray_directions[ray_indices] = reflections

# Count the number of rays reflecting on each triangle of the "walls.stl" mesh
reflection_counts = count_reflections_per_triangle(meshes['walls'], ray_origins, ray_directions, ray_indices, reflections)
print("Reflection counts per triangle on 'walls.stl':")
print(reflection_counts)

# Count the number of rays crossing through each triangle of the "outlet.stl" mesh
crossing_counts = count_crossings_per_triangle(meshes['outlet'], ray_origins, ray_directions)
print("\nCrossing counts per triangle on 'outlet.stl':")
print(crossing_counts)

# Visualize the meshes, the rays, the intersections, and the reflections
plot_mesh_and_rays(meshes, ray_origins, ray_directions, intersections=ray_indices, reflections=reflections)

