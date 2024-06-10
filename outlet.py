def write_stl(filename, vertices, faces):
    with open(filename, 'w') as f:
        f.write("solid cube\n")
        for face in faces:
            normal = cross([vertices[face[1] - 1][0] - vertices[face[0] - 1][0], 
                            vertices[face[1] - 1][1] - vertices[face[0] - 1][1], 
                            vertices[face[1] - 1][2] - vertices[face[0] - 1][2]],
                           
                           [vertices[face[2] - 1][0] - vertices[face[0] - 1][0], 
                            vertices[face[2] - 1][1] - vertices[face[0] - 1][1], 
                            vertices[face[2] - 1][2] - vertices[face[0] - 1][2]])
            
            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            for vertex in face:
                f.write(f"      vertex {vertices[vertex - 1][0]} {vertices[vertex - 1][1]} {vertices[vertex - 1][2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid cube\n")

def cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

# Define vertices
vertices = [
    [10, 0, 0], [10, 10, 0], [10, 10, 10], [10, 0, 10] # right face
]

# Define faces (each face consists of 4 vertices forming 2 triangles)
faces = [
    [1, 2, 3],
    [1, 3, 4]
]

# Write to STL file
write_stl("outlet.stl", vertices, faces)

