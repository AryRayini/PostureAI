import math

def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) between three 3D points a-b-c"""
    
    # Unpack coordinates (we only use x and y for 2D angle calculation)
    ax, ay, _ = a
    bx, by, _ = b
    cx, cy, _ = c

    # Create two vectors: from b to a, and from b to c
    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    # Compute the dot product between the two vectors
    dot = ab[0] * cb[0] + ab[1] * cb[1]

    # Compute the magnitude (length) of each vector
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)

    # Avoid division by zero if one of the vectors has zero length
    if mag_ab == 0 or mag_cb == 0:
        return 0

    # Calculate the cosine of the angle using the dot product formula
    cos_angle = dot / (mag_ab * mag_cb)

    # Clamp the cosine value between -1 and 1 to prevent math domain errors
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # Convert the angle from radians to degrees and return it
    return math.degrees(math.acos(cos_angle))
