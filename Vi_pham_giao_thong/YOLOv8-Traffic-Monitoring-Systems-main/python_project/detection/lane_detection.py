"""
Module phát hiện vi phạm làn đường
Hỗ trợ: Upload ảnh và Webcam real-time
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LaneViolationDetector:
    """Class để phát hiện xe vi phạm làn đường"""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(_BASE_DIR, 'best_new', 'vehicle.pt')
        """
        Khởi tạo detector
        Args:
            model_path: Đường dẫn đến model vehicle detection
        """
        self.model = YOLO(model_path)
        self.class_names = {
            0: "Oto",
            1: "Xe May", 
            2: "Xe Dap",
            3: "Xe Tai",
            4: "Xe Bus"
        }
        
    def _detect_lane_violation(self, boxes, frame_shape):
        """
        Phát hiện vi phạm làn đường
        
        Args:
            boxes: Danh sách bounding boxes
            frame_shape: Kích thước frame (height, width, channels)
            
        Returns:
            dict: Thông tin vi phạm cho mỗi box
        """
        violations = []
        
        # Định nghĩa vùng làn đường
        # Lane xe máy (bên trái): từ 0 đến ~50% chiều rộng
        start_line_motor = (0, int(0.2 * frame_shape[0]))
        end_line_motor = (int(0.525 * frame_shape[1]), int(0.8 * frame_shape[0]))
        
        # Lane ô tô (bên phải): từ ~55% đến hết chiều rộng
        start_line_car = (int(0.55 * frame_shape[1]), int(0.2 * frame_shape[0]))
        end_line_car = (frame_shape[1], int(0.8 * frame_shape[0]))
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Chỉ xử lý nếu confidence > 0.5 và trong vùng ROI
            if conf > 0.5 and int(0.2 * frame_shape[0]) < y1 < int(0.8 * frame_shape[0]):
                
                # Tính tâm của bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                is_violation = False
                violation_type = ""
                
                # Kiểm tra xe máy (class 1) đi vào lane ô tô
                if cls == 1:
                    in_car_lane = (start_line_car[0] < x1 < end_line_car[0] and 
                                  start_line_car[1] < y1 < end_line_car[1])
                    if in_car_lane:
                        is_violation = True
                        violation_type = "Xe may di vao lan oto"
                
                # Kiểm tra ô tô/xe tải/xe bus (class 0,3,4) đi vào lane xe máy
                elif cls in [0, 3, 4]:
                    in_motor_lane = (start_line_motor[0] < x1 < end_line_motor[0] and 
                                    start_line_motor[1] < y1 < end_line_motor[1])
                    if in_motor_lane:
                        is_violation = True
                        violation_type = f"{self.class_names[cls]} di vao lan xe may"
                
                violations.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'class': cls,
                    'class_name': self.class_names.get(cls, "Unknown"),
                    'confidence': conf,
                    'is_violation': is_violation,
                    'violation_type': violation_type
                })
        
        return violations, (start_line_motor, end_line_motor, start_line_car, end_line_car)
    
    def detect_image(self, image_path, save_path=None):
        """
        Phát hiện vi phạm làn đường trong ảnh
        
        Args:
            image_path: Đường dẫn ảnh đầu vào
            save_path: Đường dẫn lưu ảnh kết quả (optional)
            
        Returns:
            dict: Kết quả phát hiện
        """
        # Đọc ảnh
        frame = cv2.imread(image_path)
        if frame is None:
            return {
                'error': 'Không thể đọc ảnh',
                'total_violations': 0,
                'violations_detail': {},
                'total_vehicles': 0,
                'has_violation': False
            }
    
        # Dự đoán
        results = self.model(frame)
        
        all_violations = []
        lanes_info = None
        
        # Xử lý kết quả
        for result in results:
            boxes = result.boxes
            violations, lanes_info = self._detect_lane_violation(boxes, frame.shape)
            all_violations.extend(violations)
        
        # Vẽ làn đường
        if lanes_info:
            start_motor, end_motor, start_car, end_car = lanes_info
            # Vẽ làn xe máy (màu vàng)
            cv2.rectangle(frame, start_motor, end_motor, (0, 255, 255), 2)
            cv2.putText(frame, "LAN XE MAY", (start_motor[0] + 10, start_motor[1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Vẽ làn ô tô (màu cyan)
            cv2.rectangle(frame, start_car, end_car, (255, 255, 0), 2)
            cv2.putText(frame, "LAN OTO", (start_car[0] + 10, start_car[1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Vẽ bounding boxes và labels
        violation_count = 0
        violations_by_type = {}
        
        for v in all_violations:
            x1, y1, x2, y2 = v['bbox']
            
            if v['is_violation']:
                violation_count += 1
                color = (0, 0, 255)  # Đỏ cho vi phạm
                label = f"VI PHAM: {v['class_name']} {v['confidence']:.2f}"
                
                # Đếm theo loại vi phạm
                vtype = v['violation_type']
                violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1
            else:
                color = (0, 255, 0)  # Xanh lá cho đúng
                label = f"{v['class_name']} {v['confidence']:.2f}"
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ tâm
            cv2.circle(frame, v['center'], 5, color, -1)
            
            # Vẽ label
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        # Vẽ thông tin tổng quan
        total_vehicles = len(all_violations)
        info_text = f"Tong xe: {total_vehicles} | Vi pham: {violation_count}"
        cv2.rectangle(frame, (10, 10), (550, 50), (0, 0, 0), -1)
        text_color = (0, 0, 255) if violation_count > 0 else (0, 255, 0)
        cv2.putText(
            frame, 
            info_text, 
            (20, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            text_color, 
            2
        )
        
        # Lưu ảnh nếu có save_path
        if save_path:
            cv2.imwrite(save_path, frame)
        
        return {
            'total_violations': violation_count,
            'violations_detail': violations_by_type,
            'total_vehicles': total_vehicles,
            'result_image': frame,
            'has_violation': violation_count > 0
        }
    
    def detect_webcam(self):
        """
        Phát hiện vi phạm qua webcam real-time
        Generator function để stream video
        
        Yields:
            bytes: JPEG frame đã encode
        """
        cap = cv2.VideoCapture(0)
        
        # Thiết lập resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Không thể mở webcam")
            return
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Dự đoán
            results = self.model(frame, verbose=False)
            
            all_violations = []
            lanes_info = None
            
            # Xử lý kết quả
            for result in results:
                boxes = result.boxes
                violations, lanes_info = self._detect_lane_violation(boxes, frame.shape)
                all_violations.extend(violations)
            
            # Vẽ làn đường
            if lanes_info:
                start_motor, end_motor, start_car, end_car = lanes_info
                # Làn xe máy
                cv2.rectangle(frame, start_motor, end_motor, (0, 255, 255), 2)
                cv2.putText(frame, "LAN XE MAY", (start_motor[0] + 5, start_motor[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Làn ô tô
                cv2.rectangle(frame, start_car, end_car, (255, 255, 0), 2)
                cv2.putText(frame, "LAN OTO", (start_car[0] + 5, start_car[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Vẽ bounding boxes
            violation_count = 0
            
            for v in all_violations:
                x1, y1, x2, y2 = v['bbox']
                
                if v['is_violation']:
                    violation_count += 1
                    color = (0, 0, 255)  # Đỏ
                    label = f"VI PHAM: {v['class_name']}"
                else:
                    color = (0, 255, 0)  # Xanh
                    label = v['class_name']
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, v['center'], 3, color, -1)
                
                # Vẽ label
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame, 
                    (x1, y1 - text_height - 5), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                cv2.putText(
                    frame, 
                    label, 
                    (x1, y1 - 3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
            
            # Vẽ thông tin
            total_vehicles = len(all_violations)
            info_text = f"Vi pham: {violation_count} / Tong: {total_vehicles}"
            cv2.rectangle(frame, (10, 10), (350, 45), (0, 0, 0), -1)
            text_color = (0, 0, 255) if violation_count > 0 else (0, 255, 0)
            cv2.putText(
                frame, 
                info_text, 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                text_color, 
                2
            )
            
            # Encode frame thành JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame theo format multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    def process_video_stream(self, video_path):
        """
        Xử lý video và stream real-time để người dùng xem quá trình phân tích
        Generator function cho Flask streaming
        
        Args:
            video_path: Đường dẫn video cần phân tích
            
        Yields:
            bytes: JPEG frames đã xử lý
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            return
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        total_violation_count = 0
        
        print(f"Bắt đầu streaming video: {total_frames} frames, {fps} FPS")
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Tạo overlay cho làn đường
            overlay = frame.copy()
            
            # Dự đoán
            results = self.model(frame, verbose=False)
            
            all_violations = []
            lanes_info = None
            
            # Xử lý kết quả
            for result in results:
                boxes = result.boxes
                violations, lanes_info = self._detect_lane_violation(boxes, frame.shape)
                all_violations.extend(violations)
            
            # ============ VẼ LÀN ĐƯỜNG VỚI NỀN MÀU ============
            if lanes_info:
                start_motor, end_motor, start_car, end_car = lanes_info
                
                # Vẽ làn xe máy với nền màu VÀNG nhạt
                cv2.rectangle(overlay, start_motor, end_motor, (0, 255, 255), -1)
                
                # Vẽ làn ô tô với nền màu XANH CYAN nhạt
                cv2.rectangle(overlay, start_car, end_car, (255, 200, 0), -1)
                
                # Blend overlay
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                # Vẽ viền làn đường
                cv2.rectangle(frame, start_motor, end_motor, (0, 255, 255), 4)
                cv2.rectangle(frame, start_car, end_car, (255, 200, 0), 4)
                
                # Đánh nhãn làn xe máy
                label_motor = "LAN XE MAY"
                (tw1, th1), _ = cv2.getTextSize(label_motor, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
                cv2.rectangle(frame, 
                             (start_motor[0] + 15, start_motor[1] + 15),
                             (start_motor[0] + tw1 + 35, start_motor[1] + th1 + 35),
                             (0, 0, 0), -1)
                cv2.putText(frame, label_motor, 
                           (start_motor[0] + 25, start_motor[1] + 40),
                           cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
                
                # Đánh nhãn làn ô tô
                label_car = "LAN O TO"
                (tw2, th2), _ = cv2.getTextSize(label_car, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
                cv2.rectangle(frame,
                             (start_car[0] + 15, start_car[1] + 15),
                             (start_car[0] + tw2 + 35, start_car[1] + th2 + 35),
                             (0, 0, 0), -1)
                cv2.putText(frame, label_car,
                           (start_car[0] + 25, start_car[1] + 40),
                           cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 200, 0), 3)
            
            # ============ VẼ BOUNDING BOXES VÀ VI PHẠM ============
            frame_violations = 0
            
            for v in all_violations:
                x1, y1, x2, y2 = v['bbox']
                
                if v['is_violation']:
                    frame_violations += 1
                    color = (0, 0, 255)
                    label = f"** VI PHAM: {v['class_name']} **"
                    
                    # Viền đỏ dày
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 5)
                    
                    # Label VI PHẠM
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 3)
                    cv2.rectangle(frame, 
                                 (x1, y1 - th - 20), 
                                 (x1 + tw + 20, y1), 
                                 (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1 + 10, y1 - 8), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 3)
                    
                    cv2.putText(frame, "(!)", (x1 + tw + 30, y1 - 8),
                               cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
                else:
                    color = (0, 255, 0)
                    label = f"{v['class_name']}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, 
                                 (x1, y1 - th - 10), 
                                 (x1 + tw + 10, y1), 
                                 color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.circle(frame, v['center'], 6, color, -1)
            
            total_violation_count += frame_violations
            total_vehicles_in_frame = len(all_violations)
            
            # ============ BẢNG THÔNG TIN REAL-TIME ============
            width = frame.shape[1]
            info_bg_height = 120
            info_bg_width = 700
            
            # Background cho info
            cv2.rectangle(frame, (10, 10), (10 + info_bg_width, 10 + info_bg_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (10 + info_bg_width, 10 + info_bg_height), (255, 255, 255), 3)
            
            # Progress
            progress_pct = (frame_count / total_frames) * 100
            cv2.putText(frame, f"Tien do: {progress_pct:.1f}% ({frame_count}/{total_frames})", 
                       (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            # Violation status
            if frame_violations > 0:
                violation_color = (0, 0, 255)
                status_text = f"** VI PHAM: {frame_violations} xe **"
            else:
                violation_color = (0, 255, 0)
                status_text = f"OK - Khong co vi pham"
            
            cv2.putText(frame, status_text, 
                       (25, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, violation_color, 2)
            
            # Total stats
            cv2.putText(frame, f"Tong vi pham: {total_violation_count} | Xe: {total_vehicles_in_frame}", 
                       (25, 115), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 2)
            
            # ============ CẢNH BÁO TRUNG TÂM ============
            if frame_violations > 0:
                warning_text = f"CANH BAO: {frame_violations} VI PHAM!"
                (wtw, wth), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4)
                
                warning_x = (width - wtw) // 2
                warning_y = 80
                
                cv2.rectangle(frame,
                             (warning_x - 20, warning_y - wth - 20),
                             (warning_x + wtw + 20, warning_y + 10),
                             (0, 0, 255), -1)
                cv2.rectangle(frame,
                             (warning_x - 20, warning_y - wth - 20),
                             (warning_x + wtw + 20, warning_y + 10),
                             (255, 255, 255), 3)
                
                cv2.putText(frame, warning_text,
                           (warning_x, warning_y),
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
            
            # Encode frame thành JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
        print(f"Hoàn tất streaming {frame_count} frames")
    
    def detect_video(self, video_path, save_path=None):
        """
        Phát hiện vi phạm làn đường trong video
        Vẽ làn đường dạng ô vuông có màu nền + đánh dấu vi phạm trực tiếp trên video
        
        Args:
            video_path: Đường dẫn đến file video
            save_path: Đường dẫn lưu video kết quả
            
        Returns:
            dict: Thông tin vi phạm và video kết quả
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            return {
                'total_violations': 0,
                'violations_detail': {},
                'total_vehicles': 0,
                'has_violation': False,
                'error': 'Không thể mở video'
            }
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tạo VideoWriter nếu cần lưu
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Thống kê
        total_violation_count = 0
        violations_by_type = {}
        max_vehicles = 0
        frame_count = 0
        
        print(f"Đang xử lý video: {total_frames} frames, {fps} FPS...")
        print(f"Kích thước: {width}x{height}")
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Tạo overlay cho làn đường (vẽ trước khi detect)
            overlay = frame.copy()
            
            # Dự đoán
            results = self.model(frame, verbose=False)
            
            all_violations = []
            lanes_info = None
            
            # Xử lý kết quả
            for result in results:
                boxes = result.boxes
                violations, lanes_info = self._detect_lane_violation(boxes, frame.shape)
                all_violations.extend(violations)
            
            # ============ VẼ LÀN ĐƯỜNG VỚI NỀN MÀU ============
            if lanes_info:
                start_motor, end_motor, start_car, end_car = lanes_info
                
                # Vẽ làn xe máy với nền màu VÀNG nhạt
                cv2.rectangle(overlay, start_motor, end_motor, (0, 255, 255), -1)  # Fill màu vàng
                
                # Vẽ làn ô tô với nền màu XANH CYAN nhạt
                cv2.rectangle(overlay, start_car, end_car, (255, 200, 0), -1)  # Fill màu cyan
                
                # Blend overlay với frame gốc (alpha = 0.2 cho transparent)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                # Vẽ viền làn đường đậm hơn
                cv2.rectangle(frame, start_motor, end_motor, (0, 255, 255), 4)  # Viền vàng đậm
                cv2.rectangle(frame, start_car, end_car, (255, 200, 0), 4)  # Viền cyan đậm
                
                # Đánh nhãn làn xe máy với background đen
                label_motor = "LAN XE MAY"
                (tw1, th1), _ = cv2.getTextSize(label_motor, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
                cv2.rectangle(frame, 
                             (start_motor[0] + 15, start_motor[1] + 15),
                             (start_motor[0] + tw1 + 35, start_motor[1] + th1 + 35),
                             (0, 0, 0), -1)
                cv2.putText(frame, label_motor, 
                           (start_motor[0] + 25, start_motor[1] + 40),
                           cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
                
                # Đánh nhãn làn ô tô với background đen
                label_car = "LAN O TO"
                (tw2, th2), _ = cv2.getTextSize(label_car, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
                cv2.rectangle(frame,
                             (start_car[0] + 15, start_car[1] + 15),
                             (start_car[0] + tw2 + 35, start_car[1] + th2 + 35),
                             (0, 0, 0), -1)
                cv2.putText(frame, label_car,
                           (start_car[0] + 25, start_car[1] + 40),
                           cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 200, 0), 3)
            
            # ============ VẼ BOUNDING BOXES VÀ ĐÁNH DÁU VI PHẠM ============
            frame_violations = 0
            
            for v in all_violations:
                x1, y1, x2, y2 = v['bbox']
                
                if v['is_violation']:
                    frame_violations += 1
                    # VI PHẠM - Màu ĐỎ nổi bật
                    color = (0, 0, 255)
                    box_thickness = 5  # Viền dày hơn
                    label = f"** VI PHAM: {v['class_name']} **"
                    
                    # Thống kê theo loại
                    vtype = v['violation_type']
                    violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1
                    
                    # Vẽ viền đỏ nháy (làm nổi bật)
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), box_thickness)
                    
                    # Vẽ text "VI PHẠM" lớn màu đỏ
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 3)
                    cv2.rectangle(frame, 
                                 (x1, y1 - th - 20), 
                                 (x1 + tw + 20, y1), 
                                 (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1 + 10, y1 - 8), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 3)
                    
                    # Vẽ icon cảnh báo
                    cv2.putText(frame, "(!)", (x1 + tw + 30, y1 - 8),
                               cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
                else:
                    # ĐÚNG LÀN - Màu xanh lá nhạt
                    color = (0, 255, 0)
                    box_thickness = 3
                    label = f"{v['class_name']}"
                    
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                    
                    # Vẽ label nhỏ gọn
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, 
                                 (x1, y1 - th - 10), 
                                 (x1 + tw + 10, y1), 
                                 color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Vẽ điểm tâm
                cv2.circle(frame, v['center'], 6, color, -1)
            
            # Cập nhật thống kê
            total_violation_count += frame_violations
            total_vehicles_in_frame = len(all_violations)
            max_vehicles = max(max_vehicles, total_vehicles_in_frame)
            
            # ============ VẼ BẢNG THÔNG TIN TỔNG QUAN ============
            info_bg_height = 110
            info_bg_width = 650
            
            # Background đen cho thông tin
            cv2.rectangle(frame, (10, 10), (10 + info_bg_width, 10 + info_bg_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (10 + info_bg_width, 10 + info_bg_height), (255, 255, 255), 3)
            
            # Dòng 1: Frame info
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            # Dòng 2: Violation count với màu động
            if frame_violations > 0:
                violation_color = (0, 0, 255)  # Đỏ
                status_text = f"** VI PHAM: {frame_violations} xe **"
            else:
                violation_color = (0, 255, 0)  # Xanh
                status_text = f"Tot: Khong co vi pham"
            
            cv2.putText(frame, status_text, 
                       (25, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, violation_color, 2)
            
            # Dòng 3: Total vehicles
            cv2.putText(frame, f"Phuong tien: {total_vehicles_in_frame}", 
                       (25, 110), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 2)
            
            # ============ CẢNH BÁO LỚN NẾU CÓ VI PHẠM ============
            if frame_violations > 0:
                warning_text = f"CANH BAO: {frame_violations} VI PHAM!"
                (wtw, wth), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4)
                
                # Vẽ ở giữa màn hình phía trên
                warning_x = (width - wtw) // 2
                warning_y = 80
                
                # Background đỏ
                cv2.rectangle(frame,
                             (warning_x - 20, warning_y - wth - 20),
                             (warning_x + wtw + 20, warning_y + 10),
                             (0, 0, 255), -1)
                cv2.rectangle(frame,
                             (warning_x - 20, warning_y - wth - 20),
                             (warning_x + wtw + 20, warning_y + 10),
                             (255, 255, 255), 3)
                
                # Text trắng
                cv2.putText(frame, warning_text,
                           (warning_x, warning_y),
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
            
            # Ghi frame vào video output
            if writer:
                writer.write(frame)
            
            # In progress mỗi 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Tiến độ: {progress:.1f}% ({frame_count}/{total_frames}) - Vi phạm: {total_violation_count}")
        
        # Giải phóng tài nguyên
        cap.release()
        if writer:
            writer.release()
        
        print(f"\n{'='*60}")
        print(f"HOÀN TẤT XỬLÝ VIDEO!")
        print(f"{'='*60}")
        print(f"✓ Tổng frames: {frame_count}")
        print(f"✓ Tổng vi phạm: {total_violation_count}")
        print(f"✓ Chi tiết vi phạm:")
        for vtype, count in violations_by_type.items():
            print(f"  - {vtype}: {count} lần")
        print(f"✓ Video đã lưu: {save_path}")
        print(f"{'='*60}\n")
        
        return {
            'total_violations': total_violation_count,
            'violations_detail': violations_by_type,
            'total_vehicles': max_vehicles,
            'total_frames': frame_count,
            'has_violation': total_violation_count > 0
        }


# Test function
if __name__ == '__main__':
    detector = LaneViolationDetector()
    
    # Test với ảnh
    test_image = "test_lane.jpg"
    if os.path.exists(test_image):
        result = detector.detect_image(test_image, "result_lane.jpg")
        print(f"Kết quả: {result}")
    else:
        print("Không tìm thấy ảnh test")
    
    # Test với video
    test_video = "test_lane.mp4"
    if os.path.exists(test_video):
        result = detector.detect_video(test_video, "result_lane.mp4")
        print(f"Kết quả video: {result}")
    else:
        print("Không tìm thấy video test")
