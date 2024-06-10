import numpy as np
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
        normals = mesh.face_normals[triangle_indices]
        print("Normals:")
        print(normals)
        print("\nDirection:")
        print(direction)
        dot_product = np.dot(normals, direction.T)
        print("\nDot product:")
        print(dot_product)
        if len(locations) > 0:
            triangle_counts[triangle_indices[0]] += 1
            triangle_momentum[triangle_indices[0]] += -7800*dot_product*1.660538921e-24*14
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

    return triangle_mass_flow, triangle_counts

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
ray_origins = np.array([random_point_in_mesh(inlet_mesh) for _ in range(100)])

satellite_velocity = np.array([7800, 0, 0])  # Satellite velocity

# Calculate the speed of gas particles
R = 8.31446  # Ideal gas constant
particle_speed = np.sqrt(3 * R * 900 / 28)*200

# Generate random directions for the particle speed vector
random_directions = np.random.normal(size=(100, 3))
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
            current_origins = np.array(compare_nested_arrays(locations, old_origins))
            current_directions = np.array(compare_nested_arrays(new_reflections, old_directions))
        else:
            current_origins = locations
            current_directions = new_reflections

        if len(current_origins) == 0:
            break

        current_directions = current_directions[:len(current_origins)]  # Ensure current_directions has the same length as current_origins

        all_origins.append(current_origins)
        all_directions.append(current_directions)
        reflections.append(current_directions)
        
        old_origins = np.vstack([old_origins, current_origins]) if old_origins.size else current_origins
        old_directions = np.vstack([old_directions, current_directions]) if old_directions.size else current_directions

    return (np.vstack(all_origins) if all_origins else np.array([]), 
            np.vstack(all_directions) if all_directions else np.array([]), 
            reflections, 
            np.vstack(intersection_points) if intersection_points else np.array([]), 
            np.array(intersection_indices))

# Perform ray tracing with reflections
all_origins, all_directions, reflections, intersection_points, intersection_indices = trace_rays(meshes, ray_origins, ray_directions)

all_origins = np.vstack((ray_origins, all_origins))
all_directions = np.vstack((ray_directions, all_directions))
# Count the number of rays reflecting on each triangle of the "walls.stl" mesh
momentum_exchange = count_reflections_per_triangle(meshes['walls'], all_origins, all_directions, intersection_indices, reflections)
print("Momentum exchange per triangle on 'walls.stl':")
print(momentum_exchange)

# Count the number of rays crossing through each triangle of the "outlet.stl" mesh
mass_flow, crossing_counts = count_crossings_per_triangle(meshes['outlet'], all_origins, all_directions)
print("\nCrossings per triangle on 'outlet.stl':")
print(crossing_counts)
print("\nMass flow per triangle on 'outlet.stl':")
print(mass_flow)

# Visualize the meshes, the rays, the intersections, and the reflections
plot_mesh_and_rays(meshes, ray_origins, ray_directions, all_origins, all_directions, intersections=intersection_indices, reflections=np.vstack(reflections) if reflections else np.array([]), intersection_points=intersection_points)

