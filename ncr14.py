import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import h5py

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
    selected_face_idx = np.random.choice(len(areas), p=areas / np.sum(areas))
    v1, v2, v3 = mesh.vertices[mesh.faces[selected_face_idx]]
    point = random_point_in_triangle(v1, v2, v3)
    return point

# Function to calculate particle reflection after collision
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
def plot_mesh_and_rays(mesh, original_origins, original_directions, reflection_origins, reflection_directions, intersections=None, reflections=None, intersection_points=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the meshes
    mesh_collection_walls = Poly3DCollection(mesh['walls'].vertices[mesh['walls'].faces], alpha=0.3, facecolor='blue', linewidths=0.5, edgecolors='darkblue')
    mesh_collection_inlet = Poly3DCollection(mesh['inlet'].vertices[mesh['inlet'].faces], alpha=0.3, facecolor='green', linewidths=0.5, edgecolors='darkblue')
    mesh_collection_outlet = Poly3DCollection(mesh['outlet'].vertices[mesh['outlet'].faces], alpha=0.3, facecolor='red', linewidths=0.5, edgecolors='darkblue')
    ax.add_collection3d(mesh_collection_walls)
    ax.add_collection3d(mesh_collection_inlet)
    ax.add_collection3d(mesh_collection_outlet)

    # Plot original rays (in red)
    length = 12
    for origin, direction in zip(original_origins, original_directions):
        ray_end = origin + direction * length
        ax.plot([origin[0], ray_end[0]], [origin[1], ray_end[1]], [origin[2], ray_end[2]], 'r-', linewidth=0.5)

    # Plot reflected rays (in blue)
    if reflection_origins is not None and reflection_directions is not None:
        for origin, direction in zip(reflection_origins, reflection_directions):
            ray_end = origin + direction * length
            ax.plot([origin[0], ray_end[0]], [origin[1], ray_end[1]], [origin[2], ray_end[2]], 'b-', linewidth=0.5)
    
    # Plot intersections
    if intersection_points is not None and len(intersection_points) > 0:
        ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], color='green', s=5)  # Intersection points are plotted smaller

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=90, elev=0)
    plt.show()

# Function to count the number of rays reflecting on each triangle of the "walls.stl" mesh
def count_reflections_per_triangle(mesh, ray_origins, ray_directions, intersections, reflections):
    triangle_counts = np.zeros(len(mesh.faces))
    triangle_momentum = np.zeros(len(mesh.faces))    
    for i, (origin, direction) in enumerate(zip(ray_origins, ray_directions)):
        
        locations, _, triangle_indices = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        
        if len(locations) > 0:
            triangle_counts[triangle_indices[0]] += 1
            triangle_momentum[triangle_indices[0]] += 7800*1.660538921e-24*14
    return triangle_momentum

# Function to count the number of rays crossing through each triangle of the "outlet.stl" mesh
def count_crossings_per_triangle(mesh, ray_origins, ray_directions):
    triangle_counts = np.zeros(len(mesh.faces))
    triangle_mass_flow = np.zeros(len(mesh.faces))
    for i, (origin, direction) in enumerate(zip(ray_origins, ray_directions)):
        locations, _, triangle_indices = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        
        if len(locations) > 0:
            triangle_counts[triangle_indices[0]] += 1
            triangle_mass_flow[triangle_indices[0]] += 1.660538921e-24*14

    return triangle_mass_flow

def compare_nested_arrays(arr1, arr2):
    # Initialize an empty array to store the elements present in one array but not the other
    difference_arr = []

    # Define a helper function to check if a nested array is present in another array
    def is_sub_array_in_array(sub_array, array):
        for arr in array:
            if np.array_equal(sub_array, arr):
                return True
        return False

    # Find sub-arrays in arr1 but not in arr2
    for sub_array in arr1:
        if not is_sub_array_in_array(sub_array, arr2):
            difference_arr.append(sub_array)
    return difference_arr

# Load the STL files
walls_mesh = trimesh.load_mesh("walls.stl")
inlet_mesh = trimesh.load_mesh("inlet.stl")
outlet_mesh = trimesh.load_mesh("outlet.stl")

meshes = {
    'walls': walls_mesh,
    'inlet': inlet_mesh,
    'outlet': outlet_mesh
}

# Generate 3 random rays with origins at the "inlet.stl" surfaces
ray_origins = np.array([random_point_in_mesh(inlet_mesh) for _ in range(500)])

satellite_velocity = np.array([7800, 0, 0])  # Satellite velocity

# Calculate the speed of gas particles
R = 8.31446  # Ideal gas constant
particle_speed = np.sqrt(3 * R * 900 / 28)

# Generate random directions for the particle speed vector
random_directions = np.random.normal(size=(500, 3))
random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]

# Calculate ray directions
ray_directions = satellite_velocity + particle_speed * random_directions
ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, None]  # Normalize the ray directions

# Ensure ray_origins and ray_directions are 2D arrays
ray_origins = np.atleast_2d(ray_origins)
ray_directions = np.atleast_2d(ray_directions)

# Function to simulate ray tracing with reflections
def trace_rays(meshes, ray_origins, ray_directions, max_reflections=100):
    all_origins = []
    all_directions = []
    reflections = []
    intersection_points = []
    intersection_indices = []

    current_origins = ray_origins
    current_directions = ray_directions

    old_origins = np.array([])
    old_directions = np.array([])

    for _ in range(max_reflections):
        # Ensure current_origins and current_directions are 2D arrays
        current_origins = np.atleast_2d(current_origins)
        current_directions = np.atleast_2d(current_directions)

        locations, ray_indices, triangle_indices = meshes['walls'].ray.intersects_location(
            ray_origins=current_origins,
            ray_directions=current_directions
        )

        if len(locations) == 0:
            break
        normals = meshes['walls'].face_normals[triangle_indices]
        new_reflections = calculate_reflection(current_directions[ray_indices], normals)

        intersection_points.append(locations)
        intersection_indices.extend(ray_indices)

        if len(old_origins) > 0:
            current_origins = np.concatenate([old_origins, locations])
            current_directions = np.concatenate([old_directions, new_reflections])
        else:
            current_origins = locations
            current_directions = new_reflections

        reflections.append(new_reflections)

        all_origins.append(current_origins)
        all_directions.append(current_directions)

    all_origins = np.concatenate(all_origins)
    all_directions = np.concatenate(all_directions)
    reflections = np.concatenate(reflections)
    intersection_points = np.concatenate(intersection_points)

    return all_origins, all_directions, reflections, intersection_points

all_origins, all_directions, reflections, intersection_points = trace_rays(meshes, ray_origins, ray_directions)

# Count reflections per triangle for walls
triangle_momentum = count_reflections_per_triangle(walls_mesh, all_origins, all_directions, intersection_points, reflections)

# Count crossings per triangle for outlet
crossing_counts = count_crossings_per_triangle(outlet_mesh, all_origins, all_directions)

# Function to save mesh to VTKHDF file
def vtkh5_save_mesh(filename, vertices, faces, cell_type):
    with h5py.File(filename, 'w') as f:
        f.create_dataset("vertices", data=vertices)
        f.create_dataset("faces", data=faces)
        f.create_dataset("cell_type", data=cell_type)

# Function to save field data to VTKHDF file
def vtkh5_save_field(filename, time, field_data):
    with h5py.File(filename, 'a') as f:
        group = f.create_group(f"fields/{time}")
        for name, data in field_data.items():
            group.create_dataset(name, data=data)

# Save the mesh and field data to a VTKHDF file
output_file = "mesh_data.hdf"
vtkh5_save_mesh(output_file, walls_mesh.vertices, walls_mesh.faces, 5 * np.ones((walls_mesh.faces.shape[0],), np.uint8))
vtkh5_save_field(output_file, 0.0, {'momentum': triangle_momentum})
vtkh5_save_mesh(output_file, outlet_mesh.vertices, outlet_mesh.faces, 5 * np.ones((outlet_mesh.faces.shape[0],), np.uint8))
vtkh5_save_field(output_file, 0.0, {'crossings': crossing_counts})

# Plot the mesh and rays
plot_mesh_and_rays(meshes, ray_origins, ray_directions, all_origins, all_directions, intersection_points, reflections)

