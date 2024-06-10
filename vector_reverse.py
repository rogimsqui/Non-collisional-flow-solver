import numpy as np

def reverse_normals(faces, face_indices):
    """
    Reverse the normals of the specified faces by flipping their vertices.
    
    Parameters:
    - faces: The faces of the mesh.
    - face_indices: Indices of the faces to reverse normals.
    
    Returns:
    - Updated faces with normals reversed for the specified face indices.
    """
    for idx in face_indices:
        faces[idx] = faces[idx][::-1]
    return faces

# Read the mesh file
with open("cubebad.msh", "r") as f:
    lines = f.readlines()

# Identify the start and end of the vertices section
vertex_start_idx = lines.index("$Nodes\n") + 1
vertex_end_idx = lines.index("$EndNodes\n")

# Extract the vertices from the lines
vertices = []
for line in lines[vertex_start_idx:vertex_end_idx]:
    parts = line.strip().split(" ")
    if len(parts) == 4:
        vertices.append(list(map(float, parts[1:])))

# Convert vertices to a numpy array
vertices = np.array(vertices)

# Identify the start and end of the elements (faces) section
face_start_idx = lines.index("$Elements\n") + 1
face_end_idx = lines.index("$EndElements\n")

# Extract the faces from the lines
faces = []
for line in lines[face_start_idx:face_end_idx]:
    parts = line.strip().split(" ")
    if len(parts) > 4 and parts[1] == "2":  # Check if it's a triangle (type 2)
        face_vertices = list(map(int, parts[5:]))
        # Subtract 1 from each vertex index since indices start from 1 in the .msh file
        face_vertices = [v - 1 for v in face_vertices]
        faces.append(face_vertices)

# Convert faces to a numpy array
faces = np.array(faces)

# Identify the faces to reverse based on specific criteria
# You need to adjust this logic based on your specific requirements
# For now, let's assume we want to reverse the normals of all faces
faces_to_reverse = list(range(len(faces)))

# Reverse the normals of the specified faces
updated_faces = reverse_normals(faces, faces_to_reverse)

# Write the modified mesh to a new .msh file
with open("cube.msh", "w") as f:
    # Write the vertices
    f.write("$Nodes\n")
    f.write(str(len(vertices)) + "\n")
    for i, vertex in enumerate(vertices):
        f.write(f"{i+1} {' '.join(map(str, vertex))}\n")
    f.write("$EndNodes\n")

    # Write the elements (faces)
    f.write("$Elements\n")
    f.write(str(len(updated_faces)) + "\n")
    for i, face in enumerate(updated_faces):
        face_str = " ".join(map(str, [i+1, 2, 2, 1] + [v+1 for v in face]))
        f.write(f"{face_str}\n")
    f.write("$EndElements\n")

print("Normals of all faces have been reversed, and the modified mesh has been saved to 'cube.msh'.")

