import cv2
import numpy as np
import os

# =========================
# 사용자 설정
# =========================
video_path = "chessVideo.mp4"
calib_file = "calibration_output/calibration_result.npz"

pattern_size = (9, 6)  # 내부 코너 개수 (너가 사용한 체스보드 기준)

output_dir = "undistortion_output"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 캘리브레이션 결과 불러오기
# =========================
data = np.load(calib_file)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# =========================
# 비디오에서 첫 프레임 읽기
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("비디오 열기 실패")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("프레임 읽기 실패")
    exit()

h, w = frame.shape[:2]

# =========================
# 왜곡 보정 수행
# =========================
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

# ROI crop
x, y, rw, rh = roi
if rw > 0 and rh > 0:
    undistorted_cropped = undistorted[y:y+rh, x:x+rw]
else:
    undistorted_cropped = undistorted

# =========================
# 좌표축 시각화 (체스보드 위에 X,Y,Z 표시)
# =========================
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

found, corners = cv2.findChessboardCorners(gray, pattern_size)

axis_image = frame.copy()

if found:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 체스보드 실제 좌표 생성
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # 자세 추정
    ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

    # 좌표축 그리기
    cv2.drawFrameAxes(axis_image, camera_matrix, dist_coeffs, rvec, tvec, 5)

    print("[INFO] 좌표축 시각화 성공")
else:
    print("[WARNING] 체스보드 좌표축 시각화 실패 (코너 검출 안됨)")

# =========================
# 결과 저장
# =========================
cv2.imwrite(os.path.join(output_dir, "original.jpg"), frame)
cv2.imwrite(os.path.join(output_dir, "undistorted.jpg"), undistorted)
cv2.imwrite(os.path.join(output_dir, "undistorted_cropped.jpg"), undistorted_cropped)
cv2.imwrite(os.path.join(output_dir, "axis_visualization.jpg"), axis_image)

print("[INFO] 저장 완료:")
print(" - original.jpg")
print(" - undistorted.jpg")
print(" - undistorted_cropped.jpg")
print(" - axis_visualization.jpg")