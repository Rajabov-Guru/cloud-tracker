import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, nn_budget=100)


@app.route("/track", methods=["POST"])
def track_balls():
    data = request.get_json()

    # Get frame number and balls' coordinates
    frame = data['frame']
    balls = data['balls']

    # Prepare detections in DeepSORT's expected format (xywh format)
    detections = [([ball['x1'], ball['y1'], ball['x2'], ball['y2']], 1, "ball") for ball in balls]
    print(detections)

    # Run DeepSORT tracking
    tracking_results = tracker.update_tracks(raw_detections=detections, embeds=[])  # frame_id for frame referencing

    # Collect results
    tracked_balls = []
    for track in tracking_results:
        if track.is_confirmed():  # Only return confirmed tracks
            tracked_balls.append({
                "id": track.track_id,
                "x1": int(track.to_tlbr()[0]),
                "y1": int(track.to_tlbr()[1]),
                "x2": int(track.to_tlbr()[2] - track.to_tlbr()[0]),
                "y2": int(track.to_tlbr()[3] - track.to_tlbr()[1]),
            })

    return jsonify({"frame": frame, "tracked_balls": tracked_balls})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
