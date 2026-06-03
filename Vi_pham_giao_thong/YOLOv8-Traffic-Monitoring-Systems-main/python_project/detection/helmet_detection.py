"""
Module phát hiện vi phạm mũ bảo hiểm
Hỗ trợ: Upload ảnh và Webcam real-time
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HelmetDetector:
    """Class để phát hiện người không đội mũ bảo hiểm"""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(_BASE_DIR, 'model_helmet', 'helmet.pt')
        """
        Khởi tạo detector
        Args:
            model_path: Đường dẫn đến model helmet detection
        """
        self.model = YOLO(model_path)
        self.class_names = ["without helmet", "helmet"]
        
    def detect_image(self, image_path, save_path=None):
        """
        Phát hiện vi phạm mũ bảo hiểm trong ảnh
        
        Args:
            image_path: Đường dẫn ảnh đầu vào
            save_path: Đường dẫn lưu ảnh kết quả (optional)
            
        Returns:
            dict: {
                'violations': số lượng vi phạm,
                'helmets': số người đội mũ,
                'total': tổng số người,
                'result_image': ảnh đã vẽ kết quả,
                'has_violation': True/False
            }
        """
        # Đọc ảnh
        frame = cv2.imread(image_path)
        if frame is None:
            return {
                'error': 'Không thể đọc ảnh',
                'violations': 0,
                'helmets': 0,
                'total': 0,
                'has_violation': False
            }
        
        # Dự đoán
        results = self.model(frame)
        
        violations = 0
        helmets = 0
        
        # Xử lý kết quả
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Lấy class và confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Chỉ xử lý nếu confidence > 0.5
                if conf > 0.5:
                    if cls == 0:  # without helmet
                        violations += 1
                        color = (0, 0, 255)  # Đỏ cho vi phạm
                        label = f"VI PHAM: {conf:.2f}"
                    else:  # helmet
                        helmets += 1
                        color = (0, 255, 0)  # Xanh lá cho đúng
                        label = f"Deo mu: {conf:.2f}"
                    
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Vẽ label với background
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
        
        total = violations + helmets
        
        # Vẽ thông tin tổng quan
        info_text = f"Tong: {total} | Vi pham: {violations} | Dung quy dinh: {helmets}"
        cv2.rectangle(frame, (10, 10), (600, 50), (0, 0, 0), -1)
        cv2.putText(
            frame, 
            info_text, 
            (20, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Lưu ảnh nếu có save_path
        if save_path:
            cv2.imwrite(save_path, frame)
        
        return {
            'violations': violations,
            'helmets': helmets,
            'total': total,
            'result_image': frame,
            'has_violation': violations > 0
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
            
            violations = 0
            helmets = 0
            
            # Xử lý kết quả
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Lấy tọa độ
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Lấy class và confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Chỉ xử lý nếu confidence > 0.5
                    if conf > 0.5:
                        if cls == 0:  # without helmet
                            violations += 1
                            color = (0, 0, 255)  # Đỏ
                            label = f"VI PHAM: {conf:.2f}"
                        else:  # helmet
                            helmets += 1
                            color = (0, 255, 0)  # Xanh lá
                            label = f"Deo mu: {conf:.2f}"
                        
                        # Vẽ bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
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
            
            total = violations + helmets
            
            # Vẽ thông tin
            info_text = f"Vi pham: {violations} | Dung: {helmets}"
            cv2.rectangle(frame, (10, 10), (400, 50), (0, 0, 0), -1)
            
            # Màu cảnh báo nếu có vi phạm
            text_color = (0, 0, 255) if violations > 0 else (0, 255, 0)
            cv2.putText(
                frame, 
                info_text, 
                (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
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


# Test function
if __name__ == '__main__':
    detector = HelmetDetector()
    
    # Test với ảnh (nếu có)
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        result = detector.detect_image(test_image, "result.jpg")
        print(f"Kết quả: {result}")
    else:
        print("Không tìm thấy ảnh test")
