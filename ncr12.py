import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# GMSH reader function
def GMSH_reader_surftri(filename):
    file = open(filename, 'r')
    vers = np.genfromtxt(file, comments='$', max_rows=1)
    if not vers[0] == 2.2:
        raise ValueError('This parser can only understand version 2.2 of the Gmsh file format')
    nzones = int(np.genfromtxt(file, comments='$', max_rows=1))
    data = np.genfromtxt(file, dtype=('i8', 'i8', '<U256'), comments='$', max_rows=nzones)
    zones = {
        'name': np.array([z['f2'].replace('"', '') for z in data] if data.ndim > 0 else [data['f2'].tolist().replace('"', '')]),
        'code': np.array([z['f1'] for z in data] if data.ndim > 0 else [data['f1'].tolist()]),
        'dim': np.array([z['f0'] for z in data] if data.ndim > 0 else [data['f0'].tolist()]),
    }
    nnodes = int(np.genfromtxt(file, comments='$', max_rows=1))
    xyz = np.genfromtxt(file, comments='$', max_rows=nnodes)[:, 1:].copy()
    nelems = int(np.genfromtxt(file, comments='$', max_rows=1))
    data = np.genfromtxt(file, comments='$', max_rows=nelems)[:, :].astype(np.int32)
    conec = data[:, 5:].copy() - 1
    zoneid = data[:, 3].copy()
    file.close()
    return xyz, conec, zoneid, zones

# Function to generate a random point within a triangle
def random_point_in_triangle(v1, v2, v3):
    s = np.random.uniform(0, 1, 2)
    sqrt_s0 = np.sqrt(s[0])
    u = 1 - sqrt_s0
    v = s[1] * sqrt_s0
    return (u * v1 + v * v2 + (1 - u - v) * v3)

# Function to generate a random point within a mesh
def random_point_in_mesh(vertices, faces):
    areas = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) / 2
    selected_face_idx = np.random.choice(len(areas), p=areas / np.sum(areas))
    v1, v2, v3 = vertices[faces[selected_face_idx]]
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
    dot_product = np.einsum('ij,ij->i', reflections, normals)
    for i in range(len(reflections)):
        if dot_product[i] >= 0:
            reflections[i] = -reflections[i]
    return reflections

# Function to plot the mesh and rays
def plot_mesh_and_rays(vertices, faces_dict, original_origins, original_directions, reflection_origins, reflection_directions, intersections=None, reflections=None, intersection_points=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the meshes
    colors = {'walls': 'blue', 'inlet': 'green', 'outlet': 'red'}
    for name, faces in faces_dict.items():
        mesh_collection = Poly3DCollection(vertices[faces], alpha=0.3, facecolor=colors[name], linewidths=0.5, edgecolors='darkblue')
        ax.add_collection3d(mesh_collection)

    # Plot original rays (in red)
    length = 25
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
        ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], color='green', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=90, elev=0)
    plt.show()

# Function to count the number of rays reflecting on each triangle of a mesh
def count_reflections_per_triangle(mesh, ray_origins, ray_directions):
    triangle_counts = np.zeros(len(mesh.faces))
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    for i, (origin, direction) in enumerate(zip(ray_origins, ray_directions)):
        locations, ray_indices, triangle_indices = intersector.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        if len(locations) > 0:
            triangle_counts[triangle_indices[0]] += 1
    return triangle_counts

# Function to count the number of rays crossing through each triangle of a mesh
def count_crossings_per_triangle(mesh, ray_origins, ray_directions):
    triangle_counts = np.zeros(len(mesh.faces))
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    for i, (origin, direction) in enumerate(zip(ray_origins, ray_directions)):
        locations, ray_indices, triangle_indices = intersector.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        if len(locations) > 0:
            triangle_counts[triangle_indices[0]] += 1
    return triangle_counts

def compare_nested_arrays(arr1, arr2):
    difference_arr = []
    def is_sub_array_in_array(sub_array, array):
        for arr in array:
            if np.array_equal(sub_array, arr):
                return True
        return False
    for sub_array in arr1:
        if not is_sub_array_in_array(sub_array, arr2):
            difference_arr.append(sub_array)
    return difference_arr

# Load the MSH file
vertices, faces, zoneid, zones = GMSH_reader_surftri("cube.msh")

# Separate faces based on zoneid
walls_faces = faces[zoneid == 3]
inlet_faces = faces[zoneid == 1]
outlet_faces = faces[zoneid == 2]

faces_dict = {
    'walls': walls_faces,
    'inlet': inlet_faces,
    'outlet': outlet_faces
}

# Generate 3 random rays with origins at the "inlet" surfaces
ray_origins = np.array([random_point_in_mesh(vertices, inlet_faces) for _ in range(5)])

satellite_velocity = np.array([0, 0, 7800])

R = 8.31446
particle_speed = np.sqrt(3 * R * 900 / 28)*200

random_directions = np.random.normal(size=(5, 3))
random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]

ray_directions = satellite_velocity + particle_speed * random_directions
ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, None]

ray_origins = np.atleast_2d(ray_origins)
ray_directions = np.atleast_2d(ray_directions)

# Function to simulate ray tracing with reflections
def trace_rays(mesh, ray_origins, ray_directions, max_reflections=100):
    all_origins = []
    all_directions = []
    reflections = []
    intersection_points = []
    intersection_indices = []

    current_origins = ray_origins
    current_directions = ray_directions

    old_origins = np.array([])
    old_directions = np.array([])

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    for _ in range(max_reflections):
        current_origins = np.atleast_2d(current_origins)
        current_directions = np.atleast_2d(current_directions)

        locations, ray_indices, triangle_indices = intersector.intersects_location(
            ray_origins=current_origins,
            ray_directions=current_directions
        )

        if len(locations) == 0:
            break
        normals = mesh.face_normals[triangle_indices]
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

        current_directions = current_directions[:len(current_origins)]

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
all_origins, all_directions, reflections, intersection_points, intersection_indices = trace_rays(trimesh.Trimesh(vertices, faces), ray_origins, ray_directions)

# Count the number of rays reflecting on each triangle of the "walls" mesh
reflection_counts = count_reflections_per_triangle(trimesh.Trimesh(vertices, faces), all_origins, all_directions)
print("Reflection counts per triangle on 'walls':")
print(reflection_counts)

# Count the number of rays crossing through each triangle of the "outlet" mesh
crossing_counts = count_crossings_per_triangle(trimesh.Trimesh(vertices, faces), all_origins, all_directions)
print("\nCrossing counts per triangle on 'outlet':")
print(crossing_counts)

# Visualize the meshes, the rays, the intersections, and the reflections
plot_mesh_and_rays(vertices, faces_dict, ray_origins, ray_directions, all_origins, all_directions, intersections=intersection_indices, reflections=np.vstack(reflections) if reflections else np.array([]), intersection_points=intersection_points)

