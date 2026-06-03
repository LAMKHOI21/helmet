import datetime
import os
import webbrowser
import numpy as np
from werkzeug.utils import secure_filename
from tracking.sort import Sort

from flask import Flask, jsonify, url_for, request, send_from_directory
from flask import render_template, Response
from flask_cors import CORS
from processing.testHelmet import video_detect_helmet
from processing.testLane import *
from detection.helmet_detection import HelmetDetector
from detection.lane_detection import LaneViolationDetector

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder='static')
CORS(app)

# Cấu hình upload
UPLOAD_FOLDER = os.path.join(_BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(_BASE_DIR, 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size cho video

# Tạo thư mục nếu chưa có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Khởi tạo Helmet Detector

# Khởi tạo Lane Violation Detector
lane_detector = LaneViolationDetector()
helmet_detector = HelmetDetector()


# Apply Flask CORSx`
# CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
#
def video_detection(path_x=""):
    cap = cv2.VideoCapture(path_x)
    model = YOLO(os.path.join(_BASE_DIR, 'best_new', 'vehicle.pt'))
    stt_m = 0
    stt_ctb = 0

    # results = model.track(source="Videos/test4.mp4", show=True, stream=True)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            #  Dự đoán
            results = model(frame)

            # lấy ra frame sau khi đc gắn nhãn
            annotated_frame = results[0].plot()

            # lấy kích thước (height , width , _ )
            # print("kích thước frame : ", annotated_frame.shape)

            # Hiển thị lên
            # cv2.imshow("Display ", annotated_frame)
            # results = model.track(source="Videos/test4.mp4", show=True, tracker="bytetrack.yaml", stream=True)
            for result in results:
                boxes = result.boxes.numpy()

                # Lấy tên class
                name = result.names

                # lấy tất cả các thông số trong một list tọa độ các đối tượng (x0 ,y0, x1, y1, )
                # print("list 1 ", boxes.xyxy)
                list_2 = []

                # Lấy tất các các thông số của nhiều đối tượng (x0, y0 , x1 , y1 , id ,độ chính xác , loại class)
                # print("Boxes ", boxes)

                for box in boxes:
                    # lấy tên class tương ứng bounding box trong model đã custom
                    # print("Class : ", box.cls)

                    # lấy tọa độ của bounding box đối tượng (x0y0 , x1y1)
                    print("xyxy : ", box.xyxy[0])

                    # Lấy độ chính xác của bounding box đối tượng
                    # print("Độ chính xác : ", box.conf)

                    print("ID------------------- ", box.id)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # box.xyxy trả về ma trận 2 chiều dạng [[x0, y0 , x1 ,y1]]
                    # đó là tọa độ bounding box
                    print("box.xyxy", box.xyxy)
                    # org (Tọa độ cần vẽ lên bounding box (x,y) )
                    # thêm int để lấy số nguyên (nghĩa là lấy x0 , y0 để vẽ lên bounding box)
                    org = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))

                    # fontScale (Độ lớn của chữ)
                    fontScale = 0.5

                    # Blue color in RGB (Màu sắc của chữ)
                    color = ()

                    # Line thickness of 2px (Độ dày của chữ )
                    thickness = 2

                    # Lấy tọa độ bounding box
                    x = int(box.xyxy[0][0])
                    y = int(box.xyxy[0][1])
                    w = int(box.xyxy[0][2])
                    h = int(box.xyxy[0][3])

                    text = str(name[box.cls[0]] + " ") + str(round(box.conf[0], 2))

                    #####################################################################
                    # Xe OTO vi pham lane XE MAY
                    start_line_motor = (0 * int(frame.shape[1] / 10), int((2 * frame.shape[0] / 10)))
                    # 11/20 = 5.5 / 10
                    end_line_motor = (11 * int(frame.shape[1] / 20), int(8 * frame.shape[0] / 10))
                    canh_bao_vi_pham_lane_xe_may = start_line_motor[0] < box.xyxy[0][0] < end_line_motor[0] and \
                                                   start_line_motor[1] < box.xyxy[0][
                                                       1] < end_line_motor[1]
                    #####################################################################

                    # ##################################################################
                    # Xe máy vi pham lane OTO
                    # lane xe ô tô (trục y phải khớp với vùng roi)
                    # trục x lấy 6/10 , trục y lấy 3/10
                    start_line_car = (22 * int(frame.shape[1] / 40), int((2 * frame.shape[0] / 10)))

                    # lấy từ 6/10 đến hết trục X , trục y lấy 8/10
                    end_line_car = (int(frame.shape[1]), int(8 * frame.shape[0] / 10))

                    canh_bao_vi_pham_lane_oto = start_line_car[0] < box.xyxy[0][0] < end_line_car[0] and \
                                                start_line_car[1] < box.xyxy[0][
                                                    1] < end_line_car[1]
                    # filterDataViolate(frame, (0, int(5 * frame.shape[0] / 10)),
                    #                   (int(frame.shape[1]), int(55 * frame.shape[0] / 10)))
                    center_x = (x + w) // 2
                    center_y = (y + h) // 2
                    filterData = 0 <= center_x <= (int(frame.shape[1])) and int(
                        5 * frame.shape[0] / 10) <= center_y <= int(
                        52 * frame.shape[0] / 100)
                    #####################################################################

                    # vẽ ra vùng lane xe máy và oto
                    # image = cv2.rectangle(frame, start_line_car, end_line_car
                    #                       , (0, 0, 255), thickness)
                    image = cv2.rectangle(frame, start_line_motor, end_line_motor
                                          , (255, 0, 255), thickness)

                    # xét vùng roi theo trục Y
                    if int((2 * frame.shape[0]) / 10) < int(box.xyxy[0][1]) < int((8 * frame.shape[0]) / 10):
                        cv2.rectangle(frame, (x, y), (w, h), (36, 255, 12), 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        if box.cls[0] == 1:
                            if canh_bao_vi_pham_lane_oto:
                                draw_text(frame, name[box.cls[0]] + " warning", font_scale=0.5,
                                          pos=(int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                          text_color_bg=(0, 0, 0))
                                print("tọa độ xe máy vi phạm : ", box.xyxy[0])
                                # cắt hình ảnh xe máy
                                # cropped_frame = frame[round(y, 1) - 100:round(y + h, 2) + 100,
                                #                 round(x, 1) - 100: round(x + w, 1) + 100]

                                # Cắt hình làn ô tô
                                # cropped_frame = frame[int((3 * frame.shape[0]) / 10):int((8 * frame.shape[0]) / 10),
                                #                 6 * int(frame.shape[1] / 10):int(frame.shape[1])]
                                if filterData:
                                    stt_m += 1
                                    imageMotorViolate(frame, int((2 * frame.shape[0]) / 10),
                                                      int((8 * frame.shape[0]) / 10), 2 * int(frame.shape[1] / 10),
                                                      int(frame.shape[1]), stt_m)

                                    # cv2.imwrite("F:\python_project\data_xe_may_vi_pham\ " + str(count) + ".xe_may_lan_lan.jpg",
                                    #             cropped_frame)
                                    # frame = cv2.putText(frame, name[box.cls[0]] + " warning", org, font, fontScale, (0, 0, 255),
                                    #                     thickness, cv2.LINE_AA)
                            else:
                                draw_text(frame, text, font_scale=0.5,
                                          pos=(int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                          text_color=(255, 255, 255), text_color_bg=(78, 235, 133))
                                # frame = cv2.putText(frame, text, org, font, fontScale,
                                #                     generate_random_color(int(box.cls[0])), thickness,
                                #                     cv2.LINE_AA)
                        if box.cls[0] == 0 or box.cls[0] == 3 or box.cls[0] == 4:
                            if canh_bao_vi_pham_lane_xe_may:
                                draw_text(frame, name[box.cls[0]] + " warning", font_scale=0.5,
                                          pos=(int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                          text_color_bg=(0, 0, 0))
                                # Cắt hình làn ô tô
                                if filterData:
                                    stt_ctb += 1
                                    cropped_frame = frame[
                                                    int((3 * frame.shape[0]) / 10):int((8 * frame.shape[0]) / 10),
                                                    6 * int(frame.shape[1] / 10):int(frame.shape[1])]
                                    imageCTBViolate(frame, int((2 * frame.shape[0]) / 10),
                                                    int((8 * frame.shape[0]) / 10), 0 * int(frame.shape[1] / 10),
                                                    6 *
                                                    int(frame.shape[1] / 10), stt_ctb)
                            else:
                                draw_text(frame, text, font_scale=0.5,
                                          pos=(int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                          text_color=(255, 255, 255), text_color_bg=(77, 229, 26))

                    # muốn lấy 5/10 phần của height tính từ trên xuống
                    start_point = (0, int((2 * frame.shape[0]) / 10))
                    # vẽ hết chiều rộng và chiểu cao lấy 9/10
                    end_point = (int(frame.shape[1]), int((8 * frame.shape[0]) / 10))
                    color = (255, 0, 0)
                    thickness = 2

                    # vẽ ra cái ROI
                    image = cv2.rectangle(frame, start_point, end_point, color, thickness)

                    # scale_percent = 30
                    # width = int(image.shape[1] * scale_percent / 100)
                    # height = int(image.shape[0] * scale_percent / 100)
                    # dim = (width, height)

                    # resize Image
                    # resize = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                    # cv2.imshow("Roi ", image)
                    yield image
        else:
            break
    cv2.destroyAllWindows()


def generate_frames(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_helmet(path_x):
    yolo_output = video_detect_helmet(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/Hethongcamera2")
def camera_2():
    return render_template("HelmetViolate.html")


@app.route("/bb")
def bb():
    return render_template("bb.html")


@app.route("/thongke")
def tk():
    return render_template("thongke.html")


@app.route("/Hethongcamera1")
def camera_1():
    return render_template("LaneViolate.html")


@app.route("/camera1")
def video():
    return Response(generate_frames(path_x="F:/python_project/Videos/main.mp4"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/camera2")
def video_2():
    return Response(generate_frames_helmet(path_x="Videos/test11.mp4"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ========== TÍNH NĂNG MỚI: HELMET DETECTION ==========

def allowed_file(filename):
    """
    Kiểm tra file upload có hợp lệ không (image)
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    """
    Kiểm tra file có phải là video không
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def allowed_file_or_video(filename):
    """
    Kiểm tra file có phải là ảnh hoặc video hợp lệ
    """
    return allowed_file(filename) or is_video_file(filename)


@app.route("/helmet-check")
def helmet_check_page():
    """Trang kiểm tra mũ bảo hiểm"""
    return render_template("helmet_check.html")


@app.route("/upload-helmet-check", methods=['POST'])
def upload_helmet_check():
    """API upload ảnh hoặc video để kiểm tra vi phạm mũ bảo hiểm"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được upload'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    if not (file and allowed_file_or_video(file.filename)):
        return jsonify({'error': 'File không hợp lệ. Chỉ chấp nhận ảnh (JPG, PNG) hoặc video (MP4, AVI, MOV, MKV)'}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    if is_video_file(file.filename):
        # Video – trả về ngay, frontend tự kết nối stream
        helmet_video_status[unique_filename] = {
            'done': False,
            'total_violations': 0,
            'total_frames': 0,
            'processed_frames': 0,
            'has_violation': False
        }
        return jsonify({
            'success': True,
            'file_type': 'video',
            'uploaded_filename': unique_filename
        })

    # Ảnh – xử lý ngay
    result_filename = f"result_{unique_filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    detection_result = helmet_detector.detect_image(filepath, result_path)
    return jsonify({
        'success': True,
        'file_type': 'image',
        'violations': detection_result['violations'],
        'helmets': detection_result['helmets'],
        'total': detection_result['total'],
        'has_violation': detection_result['has_violation'],
        'original_image': f'/uploads/{unique_filename}',
        'result_image': f'/results/{result_filename}',
        'message': 'Phát hiện vi phạm!' if detection_result['has_violation'] else 'Không có vi phạm'
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Trả về file đã upload"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    """Trả về file kết quả"""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route("/webcam-helmet-feed")
def webcam_helmet_feed():
    """Stream webcam với helmet detection"""
    return Response(
        helmet_detector.detect_webcam(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ========== TÍNH NĂNG GIÁM SÁT LÀN ĐƯỜNG ==========

@app.route("/lane-check")
def lane_check_page():
    """Trang kiểm tra vi phạm làn đường"""
    return render_template("lane_check.html")


# Lưu trạng thái xử lý từng video (key = unique_filename)
lane_video_status = {}
helmet_video_status = {}


@app.route("/upload-lane-check", methods=['POST'])
def upload_lane_check():
    """Lưu file rồi trả về ngay – frontend tự kết nối stream để xem real-time"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được upload'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    if not (file and allowed_file_or_video(file.filename)):
        return jsonify({'error': 'File không hợp lệ. Chỉ chấp nhận ảnh (JPG, PNG) hoặc video (MP4, AVI, MOV, MKV)'}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"lane_{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    if is_video_file(file.filename):
        # Khởi tạo trạng thái – stream endpoint sẽ cập nhật
        lane_video_status[unique_filename] = {
            'done': False,
            'total_violations': 0,
            'violations_detail': {},
            'total_vehicles': 0,
            'total_frames': 0,
            'processed_frames': 0,
            'has_violation': False
        }
        return jsonify({
            'success': True,
            'file_type': 'video',
            'uploaded_filename': unique_filename
        })
    else:
        # Ảnh – xử lý ngay
        result_filename = f"result_{unique_filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        detection_result = lane_detector.detect_image(filepath, result_path)
        return jsonify({
            'success': True,
            'file_type': 'image',
            'total_violations': detection_result['total_violations'],
            'violations_detail': detection_result['violations_detail'],
            'total_vehicles': detection_result['total_vehicles'],
            'has_violation': detection_result['has_violation'],
            'original_image': f'/uploads/{unique_filename}',
            'result_image': f'/results/{result_filename}',
            'message': 'Phát hiện vi phạm!' if detection_result['has_violation'] else 'Không có vi phạm'
        })


def _iou(a, b):
    """Tính IoU giữa 2 bounding box [x1,y1,x2,y2]"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


@app.route("/lane-video-stream/<filename>")
def lane_video_stream(filename):
    """Stream MJPEG với tracking – mỗi xe vi phạm chỉ được đếm 1 lần"""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video không tồn tại'}), 404

    def generate():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        unique_violation_count = 0   # chỉ đếm xe MỚI vi phạm
        total_unique_vehicles = 0    # tổng xe đã từng xuất hiện
        violations_by_type = {}

        # --- Tracker đơn giản dựa trên IoU ---
        # Mỗi track: {id, bbox, violated, violation_type, class_name, lost_frames}
        tracks = []
        next_id = 1
        IOU_MATCH = 0.25   # ngưỡng IoU để nhận ra cùng 1 xe
        MAX_LOST = 10      # số frame mất tích trước khi xoá track

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            overlay = frame.copy()
            results = lane_detector.model(frame, verbose=False)
            all_detections = []
            lanes_info = None

            for result in results:
                violations, lanes_info = lane_detector._detect_lane_violation(result.boxes, frame.shape)
                all_detections.extend(violations)

            # ---------- Matching detections → tracks (IoU greedy) ----------
            matched_track_ids = set()
            matched_det_ids   = set()

            # Sắp xếp: ưu tiên ghép cặp có IoU cao nhất
            pairs = []
            for di, det in enumerate(all_detections):
                for ti, trk in enumerate(tracks):
                    iou_val = _iou(det['bbox'], trk['bbox'])
                    if iou_val >= IOU_MATCH:
                        pairs.append((iou_val, di, ti))

            pairs.sort(key=lambda x: -x[0])

            for iou_val, di, ti in pairs:
                if di in matched_det_ids or ti in matched_track_ids:
                    continue
                matched_det_ids.add(di)
                matched_track_ids.add(ti)
                # Cập nhật track
                det = all_detections[di]
                trk = tracks[ti]
                trk['bbox'] = det['bbox']
                trk['lost_frames'] = 0
                trk['class_name'] = det['class_name']
                # Nếu xe chưa bị đánh dấu vi phạm nhưng frame này vi phạm → đánh dấu
                if det['is_violation'] and not trk['violated']:
                    trk['violated'] = True
                    trk['violation_type'] = det.get('violation_type', 'lan_sai')
                    unique_violation_count += 1
                    vt = trk['violation_type']
                    violations_by_type[vt] = violations_by_type.get(vt, 0) + 1

            # Các detection chưa được match → track mới
            for di, det in enumerate(all_detections):
                if di in matched_det_ids:
                    continue
                new_track = {
                    'id': next_id,
                    'bbox': det['bbox'],
                    'violated': False,
                    'violation_type': None,
                    'class_name': det['class_name'],
                    'lost_frames': 0
                }
                next_id += 1
                total_unique_vehicles += 1
                if det['is_violation']:
                    new_track['violated'] = True
                    new_track['violation_type'] = det.get('violation_type', 'lan_sai')
                    unique_violation_count += 1
                    vt = new_track['violation_type']
                    violations_by_type[vt] = violations_by_type.get(vt, 0) + 1
                tracks.append(new_track)

            # Tăng lost_frames cho tracks không được match
            for ti, trk in enumerate(tracks):
                if ti not in matched_track_ids:
                    trk['lost_frames'] += 1

            # Xoá tracks mất tích quá lâu
            tracks = [t for t in tracks if t['lost_frames'] <= MAX_LOST]

            # ---------- Vẽ làn đường ----------
            if lanes_info:
                sm, em, sc, ec = lanes_info
                cv2.rectangle(overlay, sm, em, (0, 255, 255), -1)
                cv2.rectangle(overlay, sc, ec, (255, 200, 0), -1)
                cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
                cv2.rectangle(frame, sm, em, (0, 255, 255), 3)
                cv2.rectangle(frame, sc, ec, (255, 200, 0), 3)
                for lbl, clr, pt in [("LAN XE MAY", (0,255,255), (sm[0]+12, sm[1]+38)),
                                     ("LAN O TO",   (255,200,0), (sc[0]+12, sc[1]+38))]:
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                    cv2.rectangle(frame, (pt[0]-8, pt[1]-th-8), (pt[0]+tw+8, pt[1]+8), (0,0,0), -1)
                    cv2.putText(frame, lbl, pt, cv2.FONT_HERSHEY_DUPLEX, 0.9, clr, 2)

            # ---------- Vẽ bounding box từ tracks đang active ----------
            active_violations_this_frame = 0
            for det in all_detections:
                x1, y1, x2, y2 = det['bbox']
                # Tìm track tương ứng để lấy ID và trạng thái violated
                matched_trk = None
                best_iou = 0
                for trk in tracks:
                    iv = _iou(det['bbox'], trk['bbox'])
                    if iv > best_iou:
                        best_iou = iv
                        matched_trk = trk

                if matched_trk and matched_trk['violated']:
                    active_violations_this_frame += 1
                    tid = matched_trk['id']
                    # Xe vi phạm – viền đỏ + ID
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 220), 4)
                    lbl = f"#{tid} VI PHAM: {det['class_name']}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
                    cv2.rectangle(frame, (x1, y1-th-14), (x1+tw+14, y1), (0, 0, 220), -1)
                    cv2.putText(frame, lbl, (x1+7, y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 2)
                    # Tâm điểm đỏ
                    cx = (x1+x2)//2; cy = (y1+y2)//2
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    # Xe đúng làn – viền xanh lá + ID
                    tid = matched_trk['id'] if matched_trk else '?'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    lbl = f"#{tid} {det['class_name']}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+8, y1), (0, 170, 0), -1)
                    cv2.putText(frame, lbl, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            # ---------- Bảng thông tin ----------
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            cv2.rectangle(frame, (8, 8), (720, 115), (0, 0, 0), -1)
            cv2.rectangle(frame, (8, 8), (720, 115), (200, 200, 200), 1)
            cv2.putText(frame, f"Tien do: {progress:.1f}%  ({frame_count}/{total_frames} frames)",
                        (18, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            sc_clr = (0, 60, 255) if active_violations_this_frame > 0 else (0, 220, 80)
            cv2.putText(frame, f"Xe vi pham trong frame: {active_violations_this_frame}",
                        (18, 75), cv2.FONT_HERSHEY_DUPLEX, 0.75, sc_clr, 2)
            cv2.putText(frame,
                        f"Xe vi pham (duy nhat): {unique_violation_count} / Tong xe: {total_unique_vehicles}",
                        (18, 110), cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 180), 1)

            # Cảnh báo trung tâm (chỉ khi frame này có xe vi phạm đang hiển thị)
            if active_violations_this_frame > 0:
                w = frame.shape[1]
                wtxt = f"CANH BAO: {active_violations_this_frame} XE VI PHAM LAN!"
                (wtw, wth), _ = cv2.getTextSize(wtxt, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)
                wx = max(0, (w - wtw) // 2)
                cv2.rectangle(frame, (wx-12, 62), (wx+wtw+12, 62+wth+18), (0, 0, 200), -1)
                cv2.rectangle(frame, (wx-12, 62), (wx+wtw+12, 62+wth+18), (255,255,255), 2)
                cv2.putText(frame, wtxt, (wx, 62+wth+5), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3)

            # Cập nhật trạng thái
            if filename in lane_video_status:
                lane_video_status[filename].update({
                    'done': False,
                    'total_violations': unique_violation_count,
                    'violations_detail': dict(violations_by_type),
                    'total_vehicles': total_unique_vehicles,
                    'total_frames': total_frames,
                    'processed_frames': frame_count,
                    'has_violation': unique_violation_count > 0
                })

            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        cap.release()
        if filename in lane_video_status:
            lane_video_status[filename]['done'] = True

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/lane-video-status/<filename>")
def lane_video_status_check(filename):
    """Polling endpoint – frontend hỏi tiến độ và thống kê"""
    status = lane_video_status.get(filename)
    if status is None:
        return jsonify({'error': 'Không tìm thấy trạng thái video'}), 404
    return jsonify(status)


@app.route("/webcam-lane-feed")
def webcam_lane_feed():
    """Stream webcam với lane violation detection"""
    return Response(
        lane_detector.detect_webcam(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ========== KẾT THÚC TÍNH NĂNG GIÁM SÁT LÀN ĐƯỜNG ==========


# ========== TÍNH NĂNG MÙ BẢO HIỂM - XỬ LÝ VIDEO ==========

@app.route("/helmet-video-stream/<filename>")
def helmet_video_stream(filename):
    """Stream MJPEG helmet detection với SORT tracking – mỗi người vi phạm chỉ đếm 1 lần"""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video không tồn tại'}), 404

    def generate():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        violation_ids = set()   # ID người không đội mũ (unique vi phạm)

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            results = helmet_detector.model(frame, verbose=False)
            detections = np.empty((0, 5))

            for r in results:
                boxes = r.boxes
                # Vẽ người đội mũ hợp lệ (class 1) bằng màu xanh lá
                for box in boxes:
                    if int(box.cls[0]) == 1:
                        conf = float(box.conf[0])
                        if conf > 0.45:
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                            lbl = f"Deo mu: {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 170, 0), -1)
                            cv2.putText(frame, lbl, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                # Thu thập detection "không đội mũ" (class 0) vào SORT
                for box in boxes:
                    if int(box.cls[0]) == 0:
                        conf = float(box.conf[0])
                        if conf > 0.4:
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                            detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

            results_tracker = tracker.update(detections)

            roi_y_top = int(2 * frame.shape[0] / 10)
            roi_y_bot = int(8 * frame.shape[0] / 10)
            filter_y_top = int(3 * frame.shape[0] / 10)
            filter_y_bot = int(4 * frame.shape[0] / 10)

            active_violations_frame = 0
            for trk in results_tracker:
                x1, y1, x2, y2, tid = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if roi_y_top < cy < roi_y_bot:
                    # Lần đầu vào vùng filter → đánh dấu vi phạm
                    if filter_y_top <= cy <= filter_y_bot and tid not in violation_ids:
                        violation_ids.add(tid)

                    if tid in violation_ids:
                        active_violations_frame += 1
                        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 220), 3)
                        lbl = f"#{tid} VI PHAM: Khong mu"
                        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
                        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 14, y1), (0, 0, 220), -1)
                        cv2.putText(frame, lbl, (x1 + 7, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
                        lbl = f"#{tid} Khong mu"
                        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), (50, 150, 0), -1)
                        cv2.putText(frame, lbl, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Vẽ vùng ROI
            cv2.rectangle(frame, (0, roi_y_top), (frame.shape[1], roi_y_bot), (255, 0, 0), 2)

            # Bảng thông tin
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            cv2.rectangle(frame, (8, 8), (700, 115), (0, 0, 0), -1)
            cv2.rectangle(frame, (8, 8), (700, 115), (200, 200, 200), 1)
            cv2.putText(frame, f"Tien do: {progress:.1f}%  ({frame_count}/{total_frames} frames)",
                        (18, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            sc_clr = (0, 60, 255) if active_violations_frame > 0 else (0, 220, 80)
            cv2.putText(frame, f"Vi pham trong frame: {active_violations_frame}",
                        (18, 75), cv2.FONT_HERSHEY_DUPLEX, 0.75, sc_clr, 2)
            cv2.putText(frame, f"Vi pham (duy nhat): {len(violation_ids)}",
                        (18, 110), cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 180), 1)

            # Biểu ngữ cảnh báo
            if active_violations_frame > 0:
                w = frame.shape[1]
                wtxt = f"CANH BAO: {active_violations_frame} NGUOI KHONG DEO MU!"
                (wtw, wth), _ = cv2.getTextSize(wtxt, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)
                wx = max(0, (w - wtw) // 2)
                cv2.rectangle(frame, (wx - 12, 130), (wx + wtw + 12, 130 + wth + 18), (0, 0, 200), -1)
                cv2.rectangle(frame, (wx - 12, 130), (wx + wtw + 12, 130 + wth + 18), (255, 255, 255), 2)
                cv2.putText(frame, wtxt, (wx, 130 + wth + 5), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3)

            # Cập nhật trạng thái
            if filename in helmet_video_status:
                helmet_video_status[filename].update({
                    'done': False,
                    'total_violations': len(violation_ids),
                    'total_frames': total_frames,
                    'processed_frames': frame_count,
                    'has_violation': len(violation_ids) > 0
                })

            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        cap.release()
        if filename in helmet_video_status:
            helmet_video_status[filename]['done'] = True

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/helmet-video-status/<filename>")
def helmet_video_status_check(filename):
    """Polling endpoint – frontend hỏi tiến độ helmet video"""
    status = helmet_video_status.get(filename)
    if status is None:
        return jsonify({'error': 'Không tìm thấy trạng thái video'}), 404
    return jsonify(status)


# ========== KẾT THÚC TÍNH NĂNG MÙ BẢO HIỂM VIDEO ==========


# ========== KẾT THÚC TÍNH NĂNG MỚI ==========


if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:8000/')
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=True)
