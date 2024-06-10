import numpy as np
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

# Generate 10,000 random rays with origins at the "inlet.stl" surfaces
ray_origins = np.array([random_point_in_mesh(inlet_mesh) for _ in range(10000)])
satellite_velocity = np.array([7800, 0, 0])  # Satellite velocity

# Calculate the speed of gas particles
R = 8.31446  # Ideal gas constant
particle_speed = np.sqrt(3 * R * 900 / 28)

# Generate random directions for the particle speed vector
random_directions = np.random.normal(size=(10000, 3))
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

