import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def plot_mesh(ax, mesh, color):
    # Plot the mesh
    mesh_collection = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.3, facecolor=color, linewidths=0.5,
                                       edgecolors='darkblue')
    ax.add_collection3d(mesh_collection)

    # Adjust the axis limits
    scale = np.array([coord for vertex in mesh.vertices for coord in vertex]).flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=90, elev=0)

def GMSH_reader_surftri(filename):
    '''
    Read .msh files with surface triangular meshes
    '''
    # Open file for reading
    file = open(filename,'r')
    # Check GMSH version
    vers = np.genfromtxt(file,comments='$',max_rows=1)
    if not vers[0] == 2.2:
        raise ValueError('This parser can only understand version 2.2 of the Gmsh file format')
    # At this point we have checked that the file version is 2.2
    # Read the number of zones
    nzones = int(np.genfromtxt(file,comments='$',max_rows=1))
    data   = np.genfromtxt(file,dtype=('i8','i8','<U256'),comments='$',max_rows=nzones)
    # Generate a dictionary containing the boundary information
    zones = {
        'name'  : np.array([z['f2'].replace('"','') for z in data] if data.ndim > 0 else [data['f2'].tolist().replace('"','')]),
        'code'  : np.array([z['f1'] for z in data] if data.ndim > 0 else [data['f1'].tolist()]),
        'dim'   : np.array([z['f0'] for z in data] if data.ndim > 0 else [data['f0'].tolist()]),
    }
    # Now read the number of nodes
    nnodes = int(np.genfromtxt(file,comments='$',max_rows=1))
    xyz    = np.genfromtxt(file,comments='$',max_rows=nnodes)[:,1:].copy()
    # Now read the number of elements
    nelems = int(np.genfromtxt(file,comments='$',max_rows=1))
    # Read element conenctivity and elem physical id
    data   = np.genfromtxt(file,comments='$',max_rows=nelems)[:,:].astype(np.int32)
    conec  = data[:,5:].copy() - 1 # Good ol' python starting to count at 0
    zoneid = data[:,3].copy()
    # Close file
    file.close()
    # Return
    return xyz, conec, zoneid, zones

# Load the mesh from the .msh file
xyz, conec, zoneid, zones = GMSH_reader_surftri('cube.msh')

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Separate the mesh into different surface types
intake_mask = zoneid == 1
outlet_mask = zoneid == 2
walls_mask = zoneid == 3

intake_mesh = trimesh.Trimesh(vertices=xyz, faces=conec[intake_mask])
outlet_mesh = trimesh.Trimesh(vertices=xyz, faces=conec[outlet_mask])
walls_mesh = trimesh.Trimesh(vertices=xyz, faces=conec[walls_mask])

# Plot the meshes with different colors
plot_mesh(ax, walls_mesh, 'blue')   # Walls mesh in blue
plot_mesh(ax, intake_mesh, 'green')  # Inlet mesh in green
plot_mesh(ax, outlet_mesh, 'red')   # Outlet mesh in red

plt.show()

