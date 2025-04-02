import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import math
from itertools import combinations

# Hide the main Tk window
root = tk.Tk()
root.withdraw()

def load_image(dialog=False):
    if not dialog:
        return cv2.imread("imagen1.png")

    # Open file dialog for the user to select an image
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        print("No file selected.")
        return None

    # Load the image
    image = cv2.imread(file_path)
    return image

def get_contour_pos(contour):
    x, y, w, h = cv2.boundingRect(contour)
    center = (int(x + w / 2), int(y + h / 2))
    return center

def get_n_closest_points(reference, points, n):
    def distance(pt):
        return ((pt[0] - reference[0])**2 + (pt[1] - reference[1])**2)**0.5
    return sorted(points, key=distance)[:n]

def angle_between(p1, p2, p3):
    """
    Returns the angle at p2 (in degrees) formed by points p1, p2, p3.
    """
    # Create vectors
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    # Calculate dot product and magnitudes
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    # Calculate angle in degrees and return
    angle = math.degrees(math.acos(dot / (mag1 * mag2)))
    return angle

def order_points(pts):
    """
    Returns the points ordered clockwise based on the centroid.
    """
    pts = list(pts)
    # Calculate centroid
    centroid = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    # Sort points by angle from centroid
    pts.sort(key=lambda p: math.atan2(p[1] - centroid[1], p[0] - centroid[0]))
    return pts

def find_best_quadrilateral(points):
    """
    Among all 4-point combinations from 'points', selects the quadrilateral 
    whose interior angles are closest to 90 degrees.
    
    Returns:
        best_quad (list): List of 4 points ordered clockwise, or None if no combination.
    """
    best_quad = None
    best_error = float('inf')
    # Iterate over all combinations of 4 points.
    for quad in combinations(points, 4):
        ordered = order_points(quad)
        error = 0
        # Sum the absolute difference from 90 degrees for each interior angle.
        for i in range(4):
            p_prev = ordered[i - 1]
            p_curr = ordered[i]
            p_next = ordered[(i + 1) % 4]
            angle = angle_between(p_prev, p_curr, p_next)
            error += abs(angle - 90)
        if error < best_error:
            best_error = error
            best_quad = ordered
    return best_quad

def order_corners(pts):
    if len(pts) != 4:
        raise ValueError("Exactly 4 points are required.")
    # Sort points by y coordinate (to separate top and bottom)
    pts_sorted_by_y = sorted(pts, key=lambda p: p[1])
    # The first two are the top points
    top_pts = sorted(pts_sorted_by_y[:2], key=lambda p: p[0])
    # The last two are the bottom points
    bottom_pts = sorted(pts_sorted_by_y[2:], key=lambda p: p[0], reverse=True)
    return [top_pts[0], top_pts[1], bottom_pts[0], bottom_pts[1]]

def get_end_eff(end_eff_norm, base_norm):
    def normalize(point):
        x, y = point
        norm_x = x / 0.38
        norm_y = y / 0.26
        return (norm_x, norm_y)

    def denormalize(point):
        x, y = point
        denorm_x = x * -60
        denorm_y = y * -40 + 51
        return (denorm_x, denorm_y)

    # Compute the absolute difference between the two points
    end_eff_norm = (abs(end_eff_norm[0] - base_norm[0]),
                    abs(end_eff_norm[1] - base_norm[1]))

    # Normalize the resulting point (using the helper defined earlier)
    norm_end_eff = normalize(end_eff_norm)

    # Denormalize the normalized point
    denorm_end_eff = denormalize(norm_end_eff)

    # Set posY and posZ using the translated values
    posY = float(denorm_end_eff[0])
    posZ = float(denorm_end_eff[1])
    return (0, posY, posZ)

# Call the function to load the image
image = load_image(dialog=True)
if image is None:
    exit()

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV bounds
red_hsv_bounds  = ((0, 204, 171), (7, 231, 205))
green_hsv_bounds = ((42, 99, 181),(44, 154, 192))

# Create masks for red and green areas
mask_green = cv2.inRange(hsv_image, np.array(green_hsv_bounds[0], dtype="uint8"), np.array(green_hsv_bounds[1], dtype="uint8"))
mask_red = cv2.inRange(hsv_image, np.array(red_hsv_bounds[0], dtype="uint8"), np.array(red_hsv_bounds[1], dtype="uint8"))

# Find contours for both masks (external contours only)
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get dots from contours
green_dot = get_contour_pos(max(contours_green, key=cv2.contourArea, default=None))
red_dots = [get_contour_pos(cnt) for cnt in contours_red if cv2.contourArea(cnt) >= 1]

# Filter dots
base_dot = max(red_dots, key=lambda p: p[0] + p[1])
red_dots = get_n_closest_points(green_dot, red_dots, 5)
red_dots = find_best_quadrilateral(red_dots)
red_dots = order_corners(red_dots)

# Show the green dot
if green_dot is not None:
    cv2.drawMarker(image, green_dot, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    print(f"Green dot at: {green_dot}")

# Show the base point
if base_dot is not None:
    cv2.drawMarker(image, base_dot, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    print(f"Base dot at: {base_dot}")

# Show red dots
for i, red_dot in enumerate(red_dots):
    # cv2.drawMarker(image, red_dot, (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(image, str(i+1), red_dot, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    print(f"Red dot #{i} at: {red_dot}")

# Display the original image with drawn circles, the green mask, and the red mask
cv2.imshow("Original Image with markers", image)

# Normalize the points from 0 to 1
base_dot = (base_dot[0] / image.shape[1], base_dot[1] / image.shape[0])
red_dots = [(dot[0] / image.shape[1], dot[1] / image.shape[0]) for dot in red_dots]
green_dot = (green_dot[0] / image.shape[1], green_dot[1] / image.shape[0])

# Get the end effector position
end_effector = get_end_eff(green_dot, base_dot)
print(f"End effector position: {end_effector}")

# Write each coord to a line in end_eff.txt
with open("end_eff.txt", "w") as f:
    f.write(f"{end_effector[0]}\n")
    f.write(f"{end_effector[1]}\n")
    f.write(f"{end_effector[2]}\n")

print("Press any key on one of the image windows to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
