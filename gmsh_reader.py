#!/bin/env python
#
# GMSH reader
import numpy as np


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
filename = 'cube.msh'
# Read GMSH
xyz, conec, zoneid, zones = GMSH_reader_surftri(filename)
print(zones)
print(zoneid.min(),zoneid.max())

