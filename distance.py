import numpy as np
import math

def estimate_distance_from_quadrilateral(new_corners, image_size, calib_distance=1.0, calib_corners=None):
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def max_side_length(pts):
        pts_ordered = order_points(pts)
        distances = []
        for i in range(4):
            j = (i + 1) % 4  # wrap-around index
            dx = pts_ordered[j][0] - pts_ordered[i][0]
            dy = pts_ordered[j][1] - pts_ordered[i][1]
            distances.append(math.hypot(dx, dy))
        return max(distances)

    calib_corners = calib_corners or [
        (282, 269),  # Red dot 1
        (360, 285),  # Red dot 2
        (332, 416),  # Red dot 3
        (266, 407)   # Red dot 4
    ]

    width, height = image_size
    calib_pts = np.array(calib_corners, dtype="float32")
    new_pts = np.array(new_corners, dtype="float32")
    
    # Compute the maximum side length for calibration and new quadrilaterals
    calib_max_pixels = max_side_length(calib_pts)
    new_max_pixels = max_side_length(new_pts)
    
    # Normalize using the image width
    calib_norm = calib_max_pixels / width
    new_norm = new_max_pixels / width
    
    # Since the calibration quadrilateral appears with calib_norm at calib_distance,
    # any new quadrilateral with normalized size new_norm will be at:
    estimated_distance = calib_distance * (calib_norm / new_norm)
    return estimated_distance

if __name__ == "__main__":
    # Image size (width, height)
    image_size = (774, 615)

    # For demonstration, assume a new quadrilateral is detected with these corners (in pixels)
    new_corners = [
        (300, 280),
        (380, 295),
        (350, 430),
        (290, 420)
    ]
    
    distance = estimate_distance_from_quadrilateral(new_corners, image_size)
    print("Estimated distance:", distance, "m")