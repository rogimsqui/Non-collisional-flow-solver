import numpy as np
import trimesh
import meshio

# Function to read .msh file
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

# Function to generate a random point within a trimesh object
def random_point_in_mesh(mesh):
    areas = mesh.area_faces
    selected_face_idx = np.random.choice(len(areas), p=areas/np.sum(areas))
    v1, v2, v3 = mesh.vertices[mesh.faces[selected_face_idx]]
    point = random_point_in_triangle(v1, v2, v3)
    return point, selected_face_idx

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

# Function to count the number of rays reflecting on each triangle of the "walls" mesh
def count_reflections_per_triangle(mesh, ray_origins, ray_directions):
    triangle_momentum = np.zeros((len(mesh.faces), 3))
    for i, (origin, direction) in enumerate(zip(ray_origins, ray_directions)):
        locations, _, triangle_indices = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        if len(locations) > 0:
            normal = mesh.face_normals[triangle_indices[0]]
            momentum_exchange = -7800 * np.dot(normal, direction.T) * 1.660538921e-24 * 14
            triangle_momentum[triangle_indices[0]] += momentum_exchange * normal
    return triangle_momentum

# Function to count the number of rays crossing through each triangle of the "outlet" mesh
def count_crossings_per_triangle(mesh, ray_origins, ray_directions):
    triangle_mass_flow = np.zeros(len(mesh.faces))
    for i, (origin, direction) in enumerate(zip(ray_origins, ray_directions)):
        locations, _, triangle_indices = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        
        if len(locations) > 0:
            triangle_mass_flow[triangle_indices[0]] += 1.660538921e-24 * 14

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

# Load the MSH file
vertices, faces, zoneid, zones = GMSH_reader_surftri("cube.msh")

# Separate faces based on zoneid
walls_faces = faces[zoneid == 3]
inlet_faces = faces[zoneid == 1]
outlet_faces = faces[zoneid == 2]

# Create trimesh objects for each zone
walls_mesh = trimesh.Trimesh(vertices=vertices, faces=walls_faces)
inlet_mesh = trimesh.Trimesh(vertices=vertices, faces=inlet_faces)
outlet_mesh = trimesh.Trimesh(vertices=vertices, faces=outlet_faces)

meshes = {
    'walls': walls_mesh,
    'inlet': inlet_mesh,
    'outlet': outlet_mesh
}

# Generate random rays with origins at the "inlet" surfaces
num_rays = 845
ray_origins = np.zeros((num_rays, 3))
ray_origins_indices = np.zeros(num_rays, dtype=int)

for i in range(num_rays):
    ray_origins[i], ray_origins_indices[i] = random_point_in_mesh(inlet_mesh)

satellite_velocity = np.array([7800, 0, 0])  # Satellite velocity

# Calculate the speed of gas particles
R = 8.31446  # Ideal gas constant
particle_speed = np.sqrt(3 * R * 900 / 28)

# Generate random directions for the particle speed vector
random_directions = np.random.normal(size=(num_rays, 3))
random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]

# Calculate ray directions
ray_directions = satellite_velocity + particle_speed * random_directions
ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, None]  # Normalize the ray directions

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
        reflections.append(new_reflections)

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

# Count the number of rays reflecting on each triangle of the "walls" mesh
momentum_exchange = count_reflections_per_triangle(meshes['walls'], all_origins, all_directions)
print("Momentum exchange per triangle on 'walls':")
print(momentum_exchange)

# Calculate the total force exerted on the walls
total_force = np.sum(momentum_exchange, axis=0)
print("\nDrag force:")
print(total_force)

# Count the number of rays crossing through each triangle of the "outlet" mesh
mass_flow = count_crossings_per_triangle(meshes['outlet'], all_origins, all_directions)
print("\nMass flow per triangle on 'outlet':")
print(mass_flow)

# Calculate the total mass flow at the outlet
total_mass_flow = np.sum(mass_flow)
print("\nTotal mass flow at the outlet:")
print(total_mass_flow)

# Count the number of origins generated at each triangle of the "inlet" mesh
origins_per_triangle = np.zeros(len(meshes['inlet'].faces))
for idx in ray_origins_indices:
    origins_per_triangle[idx] += 1

print("\nNumber of origins per triangle on 'inlet':")
print(origins_per_triangle)

# Function to save data to VTK files
def save_to_vtk(filename, meshes, momentum_exchange, mass_flow, origins_per_triangle):
    for name, mesh in meshes.items():
        points = mesh.vertices
        cells = [("triangle", mesh.faces)]

        cell_data = {"normals": [mesh.face_normals]}

        if name == 'walls':
            cell_data["momentum_exchange"] = [momentum_exchange]
        elif name == 'outlet':
            cell_data["mass_flow"] = [mass_flow]
        elif name == 'inlet':
            cell_data["origins_count"] = [origins_per_triangle]

        point_data = {"normals": mesh.vertex_normals} if hasattr(mesh, 'vertex_normals') else None

        meshio.write_points_cells(
            f"vtk/{filename}_{name}.17.5.vtk",
            points,
            cells,
            cell_data=cell_data,
            point_data=point_data,
        )

# Save data to VTK files for visualization in ParaView
save_to_vtk("mesh_data", meshes, momentum_exchange, mass_flow, origins_per_triangle)

print("Data for ParaView visualization has been saved to 'mesh_data_walls.vtk', 'mesh_data_outlet.vtk', and 'mesh_data_inlet.vtk'.")

