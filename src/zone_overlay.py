import cv2

def draw_zone(frame, zone_coords, alert_level="safe"):
    """
    Draws a glowing safety zone overlay with dynamic colors.
    zone_coords = (x1, y1, x2, y2)
    alert_level = "safe" | "info" | "caution" | "warning" | "critical"
    """
    colors = {
        "safe": (0, 255, 0),       # Green
        "info": (255, 255, 0),     # Yellow
        "caution": (0, 165, 255),  # Orange
        "warning": (0, 140, 255),  # Darker Orange
        "critical": (0, 0, 255)    # Red
    }
    color = colors.get(alert_level, (0, 255, 0))

    x1, y1, x2, y2 = zone_coords

    # Draw glowing effect (multiple rectangles with decreasing thickness)
    for i in range(6, 1, -2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, i)

    # Add label
    cv2.putText(frame, f"Zone: {alert_level.upper()}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    return frame
