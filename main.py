import numpy as np
from flask import Flask, request, jsonify
import norfair

app = Flask(__name__)

# Define a custom distance function to track corners of bounding boxes
def bbox_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

# Initialize Norfair tracker to track bounding box corners
tracker = norfair.Tracker(
    distance_function=bbox_distance,
    distance_threshold=30
)


@app.route("/track", methods=["POST"])
def track_balls():
    data = request.get_json()

    # Get frame number and balls' coordinates
    frame = data['frame']
    balls = data['balls']

    detections = []
    for ball in balls:
        x_min, y_min = ball['x1'], ball['y1']
        x_max, y_max = ball['x2'], ball['y2']

        # Track both top-left and bottom-right corners
        top_left = np.array([x_min, y_min])
        bottom_right = np.array([x_max, y_max])

        # Create a Norfair detection for each bounding box (using corners as points)
        detections.append(norfair.Detection(points=np.array([top_left, bottom_right])))

    # Update tracker with the detections (track bounding box corners)
    tracked_objects = tracker.update(detections)

    # Collect tracking results
    tracked_balls = []
    for obj in tracker.tracked_objects:
        top_left, bottom_right = obj.estimate[0], obj.estimate[1]
        tracked_balls.append({
            "id": obj.id,
            "x1": int(top_left[0]),  # Tracked top-left x
            "y1": int(top_left[1]),  # Tracked top-left y
            "x2": int(bottom_right[0]),  # Tracked bottom-right x
            "y2": int(bottom_right[1])  # Tracked bottom-right y
        })

    return jsonify({"frame": frame, "tracked_balls": tracked_balls})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
