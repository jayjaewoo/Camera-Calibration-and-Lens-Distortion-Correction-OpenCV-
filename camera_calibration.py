import cv2
import numpy as np
import os

# =========================
# 사용자 설정
# =========================
video_path = "chessVideo.mp4"   # 실제 파일 확장자에 맞게 수정
pattern_size = (9, 6)           # 내부 코너 개수 (가로, 세로)
square_size = 3.0               # 한 칸 크기 (cm)

# 프레임 샘플링 간격
frame_skip = 10

# 결과 저장 폴더
output_dir = "calibration_output"
os.makedirs(output_dir, exist_ok=True)

# 코너 정밀화 기준
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# =========================
# 체스보드의 3D 좌표 준비
# =========================
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# 3D 점 / 2D 점 저장용
objpoints = []
imgpoints = []

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[오류] 비디오를 열 수 없습니다: {video_path}")
    print("파일 이름과 확장자를 다시 확인하세요.")
    exit()

frame_count = 0
valid_count = 0
image_size = None

print("[INFO] 비디오에서 체스보드 코너를 찾는 중...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners2)

        vis = frame.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners2, found)
        save_path = os.path.join(output_dir, f"detected_{valid_count:02d}.jpg")
        cv2.imwrite(save_path, vis)

        valid_count += 1
        print(f"[INFO] 체스보드 검출 성공: {valid_count}장")

cap.release()

if valid_count < 5:
    print("[오류] 체스보드를 찾은 프레임이 너무 적습니다.")
    print("영상에서 체스보드가 전체가 보이는지, 너무 흐리지 않은지 확인하세요.")
    exit()

print("\n[INFO] 캘리브레이션 수행 중...")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

rmse = ret

print("\n===== Calibration Result =====")
print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
print(f"\nRMSE: {rmse}")

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print("\n===== Data =====")
print(f"fx = {fx}")
print(f"fy = {fy}")
print(f"cx = {cx}")
print(f"cy = {cy}")
print(f"rmse = {rmse}")
print("distortion coefficients =")
print(dist_coeffs)

np.savez(
    os.path.join(output_dir, "calibration_result.npz"),
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    fx=fx, fy=fy, cx=cx, cy=cy, rmse=rmse
)

print(f"\n[INFO] 결과 저장 완료: {output_dir}/calibration_result.npz")
print(f"[INFO] 검출된 체스보드 이미지 저장 폴더: {output_dir}")