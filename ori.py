import sys
import open3d as o3d

# Usage: python check_axis.py "path/to/your/file.ply"
filename = "O:/data/S3DIS/55L9 study room classified.ply" # Change this or use sys.argv[1]

pcd = o3d.io.read_point_cloud(filename)
print("Displaying Point Cloud. LOOK AT THE AXES:")
print("RED = X, GREEN = Y, BLUE = Z")
print("If the Blue line is NOT pointing up (perpendicular to floor), you have a rotation issue.")

# Add coordinate frame (size 1.0 meters)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])

o3d.visualization.draw_geometries([pcd, axes], width=1024, height=768)