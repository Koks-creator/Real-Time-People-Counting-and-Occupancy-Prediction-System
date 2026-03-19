import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flask import render_template, request, Response, jsonify

from webapp import app, area_monitor_tool, video_area_mapping, webcam_area_mapping
from config import Config

# dodanie jeszcze kamer, nie tylko filmy
VIDEOS_FOLDER = Path(Config.VIDEOS_FOLDER)


def get_videos() -> list:
    return [f.name for f in VIDEOS_FOLDER.glob("*.mp4")]


@app.route("/")
def home():
    videos = get_videos()
    return render_template("home.html", videos=videos)


@app.route("/stream")
def stream():
    video_name = request.args.get("video")
    if not video_name:
        return "No video", 400
    video_path = VIDEOS_FOLDER / video_name
    if not video_path.exists():
        return "Video path not found", 404

    # sprawdź PRZED load_areas
    if area_monitor_tool._streaming:
        return Response(
            area_monitor_tool.stream_busy_response(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    areas_path = video_area_mapping.get(str(Path(video_path)))
    if not areas_path:
        return "No areas configuration for this video", 404
    area_monitor_tool.load_areas(areas_json_path=areas_path)

    params = {
        "conf":                      float(request.args.get("conf", 0.2)),
        "iou":                       float(request.args.get("iou", 0.35)),
        "augment":                   request.args.get("augment", "true") == "true",
        "agnostic_nms":              request.args.get("agnostic_nms", "true") == "true",
        "alpha":                     float(request.args.get("alpha", 0.6)),
        "use_sahi":                  request.args.get("use_sahi", "false") == "true",
        "sahi_conf":                 float(request.args.get("sahi_conf", 0.2)),
        "sahi_slice_height":         int(request.args.get("sahi_slice_height", 480)),
        "sahi_slice_width":          int(request.args.get("sahi_slice_width", 480)),
        "sahi_overlap_height_ratio": float(request.args.get("sahi_overlap_height_ratio", 0.2)),
        "sahi_overlap_width_ratio":  float(request.args.get("sahi_overlap_width_ratio", 0.2)),
    }

    return Response(
        area_monitor_tool.stream_video(video_path=str(video_path), **params),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/videos")
def videos():
    return jsonify(get_videos())

@app.route("/stream/camera")
def stream_camera():
    cam_id = request.args.get("cam_id")
    if not cam_id:
        return "No camera id", 400

    if area_monitor_tool._streaming:
        return Response(
            area_monitor_tool.stream_busy_response(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    areas_path = webcam_area_mapping.get(cam_id)
    if not areas_path:
        return "No areas configuration for this camera", 404
    area_monitor_tool.load_areas(areas_json_path=areas_path)

    params = {
        "conf":                      float(request.args.get("conf", 0.2)),
        "iou":                       float(request.args.get("iou", 0.35)),
        "augment":                   request.args.get("augment", "true") == "true",
        "agnostic_nms":              request.args.get("agnostic_nms", "true") == "true",
        "alpha":                     float(request.args.get("alpha", 0.6)),
        "use_sahi":                  request.args.get("use_sahi", "false") == "true",
        "sahi_conf":                 float(request.args.get("sahi_conf", 0.2)),
        "sahi_slice_height":         int(request.args.get("sahi_slice_height", 480)),
        "sahi_slice_width":          int(request.args.get("sahi_slice_width", 480)),
        "sahi_overlap_height_ratio": float(request.args.get("sahi_overlap_height_ratio", 0.2)),
        "sahi_overlap_width_ratio":  float(request.args.get("sahi_overlap_width_ratio", 0.2)),
    }

    return Response(
        area_monitor_tool.stream_video(video_path=int(cam_id), **params),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/cameras")
def cameras():
    return jsonify(list(webcam_area_mapping.keys()))


@app.route("/areas")
def areas():
    result = {}
    for area_name, area_items in area_monitor_tool.areas.items():
        result[area_name] = {
            "count":         area_items["count"],
            "occupancy_pct": area_items["occupancy_pct"],
            "pred":          {str(k): v for k, v in area_items["pred"].items()},
        }
    return jsonify(result)

@app.route("/history")
def history():
    result = {}
    for area_name, hist in area_monitor_tool.history.items():
        if not hist:
            continue
        t0 = hist[0][0]
        result[area_name] = [
            {"t": round(t - t0, 1), "count": c}
            for t, c in hist
        ]
    return jsonify(result)