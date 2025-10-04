import streamlit as st
import tempfile, os, cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ------------ SETTINGS ------------
MODEL_PATH = "yolov8_person_best.pt"  
st.set_page_config(page_title="People Counting", layout="wide")
st.title("👥 People Detection & Counting App")

# side controls
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
iou_thresh  = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.5, 0.05)

# load model once
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)
model = load_model(MODEL_PATH)

# ------------ LAYOUT: 3 columns for actions ------------
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("🖼️ Image")
    img_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="img_up")
    img_btn  = st.button("Detect People in Image")

with c2:
    st.subheader("🎥 Video")
    vid_file = st.file_uploader("Upload video", type=["mp4","mov","avi","m4v","mpeg","mpg"], key="vid_up")
    vid_btn  = st.button("Detect People in Video")

with c3:
    st.subheader("🚶 Line Crossing")
    line_vid_file = st.file_uploader("Upload video", type=["mp4","mov","avi","m4v","mpeg","mpg"], key="line_up")
    line_btn      = st.button("Count People Crossing Line")

# ------------ CENTERED RESULT AREA ------------
center = st.container()
with center:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    frame_placeholder = st.empty()   # shows image/video frames
    text_placeholder  = st.empty()   # big bold text under the frame
    st.markdown("</div>", unsafe_allow_html=True)

# ------------ HELPERS ------------
def side_of_line(p, a, b):
    # sign of cross product -> which side of AB the point P is
    return np.sign((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]))

def show_big_text(msg, color="black"):
    text_placeholder.markdown(
        f"<h3 style='text-align:center; color:{color};'>"
        f"<b>{msg}</b></h3>",
        unsafe_allow_html=True
    )

# ------------ OPTION 1: IMAGE ------------
if img_file and img_btn:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(img_file.read())
        img_path = tfile.name

    img = cv2.imread(img_path)
    res = model.predict(img, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
    # count only persons
    num_people = sum(1 for b in res.boxes if model.names[int(b.cls[0])] == "person")
    annotated = res.plot()[:, :, ::-1]  # BGR->RGB

    frame_placeholder.image(annotated, use_container_width=True)
    show_big_text(f"Detected {num_people} people")
    os.unlink(img_path)

# ------------ OPTION 2: VIDEO (frame-wise people count) ------------
if vid_file and vid_btn:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(vid_file.read())
        vid_path = tfile.name

    cap = cv2.VideoCapture(vid_path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        # overlay boxes
        annotated = res.plot()[:, :, ::-1]
        # count persons in this frame
        frame_count = sum(1 for b in res.boxes if model.names[int(b.cls[0])] == "person")

        frame_placeholder.image(annotated, use_container_width=True)
        show_big_text(f"Current frame: {frame_count} people")

    cap.release()
    os.unlink(vid_path)

# ------------ OPTION 3: VIDEO with LINE-CROSS COUNTING ------------
if line_vid_file and line_btn:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(line_vid_file.read())
        line_vid_path = tfile.name

    cap = cv2.VideoCapture(line_vid_path)

    # derive default line across the middle
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    p1, p2 = (int(0.1 * W), H // 2), (int(0.9 * W), H // 2)

    tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=0.7)
    last_side = {}
    entered, exited = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        dets = []
        for b in res.boxes:
            if model.names[int(b.cls[0])] != "person":
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            dets.append([[x1, y1, x2 - x1, y2 - y1], conf, "person"])

        tracks = tracker.update_tracks(dets, frame=frame)

        # draw line
        cv2.line(frame, p1, p2, (0, 255, 255), 2)

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            s = side_of_line((cx, cy), p1, p2)

            if tid not in last_side:
                last_side[tid] = s
            else:
                prev = last_side[tid]
                if s != 0 and prev != 0 and s != prev:
                    if prev < s:
                        entered += 1
                    else:
                        exited += 1
                last_side[tid] = s

            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 120), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated, use_container_width=True)
        show_big_text(f"Entered: {entered}   |   Exited: {exited}")

    cap.release()
    os.unlink(line_vid_path)
