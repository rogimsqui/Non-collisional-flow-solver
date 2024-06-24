"NonCollisional.py" is the non-collisional flow preliminary code which does not include particle trajectories after their collision with the solid surface and the momentum exchanges in such collisions.

"NonCollisional.py" is the same code, now including particle trajectories after the collisions.

"simple_mesh.stl" is a very simple 4 cell parallelogram mesh.

"Cylinder.stl" is the intake geometry .stl file sent by Max.

"simple_mesh_generator.py" is the code that generates the "simple_mesh.stl" file.

"simple_mesh_generator2.py" generates a simple mesh with a smaller triangle

"Cylinder.stl" is a cylindrical mesh

"ncr2.py" (which stands for Non Collisional with Reflections 2" is a code for reflections with arrows

"ncr3.py" is a code for reflections with origins marked as blue points

"ncr4.py" is an incorrect version of the diffuse reflection demonstrator code (reflection directions are random, but within the plane of the reflecting surface)

"ncr5.py" is the reflections demonstrator. The incident ray is not plotted correctly

"ncr6.py" is an effort to implement particle velocities on a cube mesh

"MeshViewer.py" is a quick .stl file visualizer

"MultipleMeshViewer.py" allows the visualization of 3 meshes at the same time, in the same plot

"walls.stl" is the mesh consisting of the lateral walls of the simple cube

"inlet.stl" is the mesh consisting of the inlet surface of the cube

"outlet.stl" is the mesh consisting of the outlet surface of the cube

(note that all three meshes have their respective .py generating codes)

"ncr7.py" implements particle velocities (satellite+kinetic) on the triple mesh cube. 100 particles are generated randomly at the inlet

"ncr8.py" is the same as ncr7.py but with 10000 particles

"ncr9.py" is the same as ncr7.py but now counts the amount of particles reflecting on each triangle of the walls, and the amount of particles crossing each triangle of the outlet

"ncr10.py" now eliminates all graphic plot (to make the code less computationally heavy) and simply adds the counters, this time with 10000 particles

"ncr11.py" implements particle velocities and multiple reflections (ray_trace function)

"ncr11.5.py" is a demonstrator of the ray_trace function

"ncr11.6.py" is the same as ncr11.5.py but with a better plot of the reflections

"ncr12.py" is ncr11.py but implementing Arnau's .msh reader code

"ncr13.py" is ncr11.py but computing the mass flow at the outlet and the momentum exchanges at the walls

"ncr14.py" is ncr13.py (with incorrect computing of the mass flow)  but exporting the results into vtkhdf

"ncr15.py" is ncr13.py without the plots

"ncr15.5.py" is ncr15.py computing the drag force and the total mass flow at the outlet

"ncr16.py" is ncr15.5.py with data being exported to VTKHDF format

"ncr16.5.py" is ncr16.py including the computation of the amount of particles generated at each triangle of the inlet mesh 

"ncr17.py" is ncr16.5 is ncr16.py but with the mesh cube.msh

"ncr17.5.py" is ncr17.py including the normal vectors in the mesh data

"ncr17.6.py" is ncr 17.5 adapted to read the intake.msh files

"ncr18.py" is ncr17.6.py with the continuous approach (setting the simulation time)
