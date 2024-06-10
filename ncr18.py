import numpy as np
import trimesh
import meshio

# Function to read .msh file
def GMSH_reader_surftri(filename):
    with open(filename, 'r') as file:
        vers = np.genfromtxt(file, comments='$', max_rows=1)
        if vers[0] != 2.2:
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
vertices, faces, zoneid, zones = GMSH_reader_surftri("intake_ext.msh")

# Separate faces based on zoneid
walls_faces = faces[zoneid == 1]
inlet_faces = faces[zoneid == 3]
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

# Function to generate initial random rays with origins at the "inlet" surfaces
def generate_initial_rays(num_rays, inlet_mesh):
    ray_origins = np.zeros((num_rays, 3))
    ray_origins_indices = np.zeros(num_rays, dtype=int)

    for i in range(num_rays):
        ray_origins[i], ray_origins_indices[i] = random_point_in_mesh(inlet_mesh)

    return ray_origins, ray_origins_indices

# Function to simulate the rays within the intake geometry
def simulate_rays_continuous(meshes, simulation_time, num_rays, particle_speed):
    # Generate initial rays
    ray_origins, ray_origins_indices = generate_initial_rays(num_rays, meshes['inlet'])
    satellite_velocity = np.array([7800, 0, 0])

    random_directions = np.random.normal(size=(num_rays, 3))
    random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
    ray_directions = satellite_velocity + particle_speed * random_directions
    ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, None]

    total_traveled_distances = np.zeros(num_rays)
    total_time = 0

    triangle_momentum = np.zeros((len(meshes['walls'].faces), 3))
    triangle_mass_flow = np.zeros(len(meshes['outlet'].faces))
    origins_per_triangle = np.zeros(len(meshes['inlet'].faces))

    while total_time < simulation_time:
        all_origins = []
        all_directions = []
        current_origins = ray_origins
        current_directions = ray_directions

        while len(current_origins) > 0:
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

            all_origins.append(current_origins)
            all_directions.append(current_directions)

            traveled_distances = np.linalg.norm(locations - current_origins[ray_indices], axis=1)
            if np.any(ray_indices >= len(total_traveled_distances)):
                break  # Prevent index error

            total_traveled_distances[ray_indices] += traveled_distances

            intersection_points, _, outlet_indices = meshes['outlet'].ray.intersects_location(
                ray_origins=locations,
                ray_directions=new_reflections
            )
            origins_per_triangle += np.histogram(ray_origins_indices, bins=np.arange(len(meshes['inlet'].faces)+1))[0]

            momentum_exchange = count_reflections_per_triangle(
                meshes['walls'],
                current_origins[ray_indices],
                current_directions[ray_indices]
            )
            triangle_momentum += momentum_exchange

            mass_flow = count_crossings_per_triangle(meshes['outlet'], locations, new_reflections)
            triangle_mass_flow += mass_flow

            current_origins = locations
            current_directions = new_reflections

        total_time += np.min(traveled_distances) / 7800  # Update the total simulation time

        if len(current_origins) < num_rays:
            # Regenerate particles that have exited the intake
            new_rays_needed = num_rays - len(current_origins)
            if new_rays_needed > 0:
                new_ray_origins, new_ray_origins_indices = generate_initial_rays(new_rays_needed, meshes['inlet'])
                satellite_velocity = np.array([7800, 0, 0])

                random_directions = np.random.normal(size=(len(new_ray_origins), 3))
                random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
                new_ray_directions = satellite_velocity + particle_speed * random_directions
                new_ray_directions /= np.linalg.norm(new_ray_directions, axis=1)[:, None]

                current_origins = np.vstack((current_origins, new_ray_origins))
                current_directions = np.vstack((current_directions, new_ray_directions))

    return triangle_momentum, triangle_mass_flow, origins_per_triangle

# Define simulation parameters
num_rays = 100
simulation_time = 0.00005  # seconds
particle_speed = np.sqrt(3 * 8.31446 * 900 / 28)

# Perform the simulation
momentum_exchange, mass_flow, origins_per_triangle = simulate_rays_continuous(meshes, simulation_time, num_rays, particle_speed)

print("Momentum exchange per triangle on 'walls':")
print(momentum_exchange)

print("\nMass flow per triangle on 'outlet':")
print(mass_flow)

print("\nNumber of origins per triangle on 'inlet':")
print(origins_per_triangle)

# Save data to VTK files for visualization in ParaView
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
            f"vtk/{filename}_{name}18.vtk",
            points,
            cells,
            cell_data=cell_data,
            point_data=point_data,
        )

save_to_vtk("mesh_data", meshes, momentum_exchange, mass_flow, origins_per_triangle)

print("Data for ParaView visualization has been saved to 'mesh_data_walls.vtk', 'mesh_data_outlet.vtk', and 'mesh_data_inlet.vtk'.")

