import numpy as np
import trimesh
import meshio
import pyembree  # Ensure pyembree is used for faster ray-tracing
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
    locations, ray_indices, triangle_indices = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )
    if len(locations) > 0:
        normals = mesh.face_normals[triangle_indices]
        momentum_exchanges = -7800 * np.einsum('ij,ij->i', normals, ray_directions[ray_indices]) * 1.660538921e-24 * 14
        np.add.at(triangle_momentum, triangle_indices, momentum_exchanges[:, None] * normals)
    return triangle_momentum

# Function to count the number of rays crossing through each triangle of the "outlet" mesh
def count_crossings_per_triangle(mesh, ray_origins, ray_directions):
    triangle_mass_flow = np.zeros(len(mesh.faces))
    locations, ray_indices, triangle_indices = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )
    if len(locations) > 0:
        mass_flow = np.bincount(triangle_indices, minlength=len(mesh.faces)) * 1.660538921e-24 * 14
        triangle_mass_flow += mass_flow
    return triangle_mass_flow

# Load the MSH file
vertices, faces, zoneid, zones = GMSH_reader_surftri("intake_ext.msh")

# Separate faces based on zoneid
walls_faces = faces[zoneid == 1]
inlet_faces = faces[zoneid == 3]
outlet_faces = faces[zoneid == 2]

# Create trimesh objects for each zone
walls_mesh = trimesh.Trimesh(vertices=vertices, faces=walls_faces, process=False)
inlet_mesh = trimesh.Trimesh(vertices=vertices, faces=inlet_faces, process=False)
outlet_mesh = trimesh.Trimesh(vertices=vertices, faces=outlet_faces, process=False)

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

    active_rays = np.arange(num_rays)

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
            total_traveled_distances[active_rays[ray_indices]] += traveled_distances

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

            mass_flow = count_crossings_per_triangle(
                meshes['outlet'],
                current_origins[ray_indices],
                new_reflections
            )
            triangle_mass_flow += mass_flow

            current_origins = locations
            current_directions = new_reflections
            active_rays = active_rays[ray_indices]
            ray_origins_indices = ray_origins_indices[ray_indices]

        total_time += np.min(total_traveled_distances) / particle_speed

        if len(current_origins) < num_rays:
            new_rays_needed = num_rays - len(current_origins)
            new_ray_origins, new_ray_origins_indices = generate_initial_rays(new_rays_needed, meshes['inlet'])

            random_directions = np.random.normal(size=(new_rays_needed, 3))
            random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
            new_ray_directions = satellite_velocity + particle_speed * random_directions
            new_ray_directions /= np.linalg.norm(new_ray_directions, axis=1)[:, None]

            current_origins = np.vstack((current_origins, new_ray_origins))
            current_directions = np.vstack((current_directions, new_ray_directions))

    return triangle_momentum, triangle_mass_flow, origins_per_triangle

# Define simulation parameters
num_rays = 1000
simulation_time = 0.5  # seconds
particle_speed = np.sqrt(3 * 8.31446 * 900 / 28)

# Perform the simulation on each MPI rank
local_num_rays = num_rays // size
momentum_exchange, mass_flow, origins_per_triangle = simulate_rays_continuous(meshes, simulation_time, local_num_rays, particle_speed)

# Gather results from all ranks
total_momentum_exchange = np.zeros_like(momentum_exchange)
total_mass_flow = np.zeros_like(mass_flow)
total_origins_per_triangle = np.zeros_like(origins_per_triangle)

comm.Reduce(momentum_exchange, total_momentum_exchange, op=MPI.SUM, root=0)
comm.Reduce(mass_flow, total_mass_flow, op=MPI.SUM, root=0)
comm.Reduce(origins_per_triangle, total_origins_per_triangle, op=MPI.SUM, root=0)

if rank == 0:
    print("Momentum exchange per triangle on 'walls':")
    print(total_momentum_exchange)

    print("\nMass flow per triangle on 'outlet':")
    print(total_mass_flow)

    print("\nNumber of origins per triangle on 'inlet':")
    print(total_origins_per_triangle)

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
                f"vtk/{filename}_{name}19.vtk",
                points,
                cells,
                cell_data=cell_data,
                point_data=point_data,
            )

    save_to_vtk("mesh_data", meshes, total_momentum_exchange, total_mass_flow, total_origins_per_triangle)

    print("Data for ParaView visualization has been saved to 'mesh_data_walls.vtk', 'mesh_data_outlet.vtk', and 'mesh_data_inlet.vtk'.")

