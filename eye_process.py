import cv2 as cv
import numpy as np
import mediapipe as mp
import math
from PIL import Image
from numpy.lib.function_base import average

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# points of iris
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [33]  # right eye right most landmark
L_H_RIGHT = [133]  # right eye left most landmark
R_H_LEFT = [362]  # left eye right most landmark
R_H_RIGHT = [263]  # left eye left most landmark

R_U_landmark = [386]  # right eye up most landmark
R_D_landmark = [374]  # right eye down most landmark
L_U_landmark = [159]  # left eye up most landmark
L_D_landmark = [145]  # left eye down most landmark


def euclidean_distance(point1, point2):
    """
    calculate euclidean distance between two points
    input: point1, point2 : x and y for each point
    output: distance between point1 and point2

    """
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# thresholds represent ratio of distance between center of iris and point
right_thre = 0.42
left_thre = 0.57
up_thre = 0.25
down_thre = 0.58


def iris_position_in_eye(iris_center, right_point, left_point, up_point, down_point):
    """
    iris position :
    is return position of iris center and ratio_horizontal,ratio_vertical
    """

    center_to_right_dist = euclidean_distance(iris_center, right_point)
    center_to_up_dist = euclidean_distance(iris_center, up_point)

    total_distance_horizontal = euclidean_distance(right_point, left_point)
    total_distance_vertical = euclidean_distance(up_point, down_point)

    ratio_horizontal = center_to_right_dist / total_distance_horizontal

    if total_distance_vertical == 0:
        ratio_vertical = 0.1
    else:
        ratio_vertical = center_to_up_dist / total_distance_vertical

    iris_position = ""
    # iris in center horizontal
    if right_thre < ratio_horizontal <= left_thre:
        if ratio_vertical <= up_thre:
            iris_position = "up"
        elif up_thre < ratio_vertical <= down_thre:
            iris_position = "center"
        else:
            iris_position = "down"

    # iris in center vertical
    elif up_thre < ratio_vertical <= down_thre:
        if ratio_horizontal <= right_thre:
            iris_position = "right"
        elif right_thre < ratio_horizontal <= left_thre:
            iris_position = "center"
        else:
            iris_position = "left"

    # iris in right horizontal
    elif ratio_horizontal <= right_thre:
        if ratio_vertical <= up_thre:
            iris_position = "right _ up"
        elif ratio_vertical > down_thre:
            iris_position = "right _ down"

    # iris in left horizontal
    else:
        if ratio_vertical <= up_thre:
            iris_position = "left _ up"
        elif ratio_vertical > down_thre:
            iris_position = "left _ down"

    return iris_position, ratio_horizontal, ratio_vertical


def crop_face_from_frame(face_detection_model, frame):
    height, width, channels = frame.shape
    results = face_detection_model.process(frame)

    if results.detections:
        for detection in results.detections:
            results.detections[0].location_data.relative_bounding_box.xmin
            l, t = normaliz_pixel(detection.location_data.relative_bounding_box.xmin,
                                  detection.location_data.relative_bounding_box.ymin, width, height)
            r = detection.location_data.relative_bounding_box.width * width + l
            b = detection.location_data.relative_bounding_box.height * height + t

            l -= (r - l) * 0.3
            r += (r - l) * 0.3
            t -= (b - t) * 0.3
            b += (b - t) * 0.3

            bbox = ([l, t, r, b])
        frame = Image.fromarray(frame)
        face = frame.crop(bbox)
        frame = np.asarray(face)

    return frame


def detect_iris_on_frame(face_mesh_model, frame):
    with face_mesh_model.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                    results.multi_face_landmarks[0].landmark])
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 1, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 1, (0, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_U_landmark][0], 1, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_D_landmark][0], 1, (0, 255, 255), -1, cv.LINE_AA)

            #             time.sleep(0.2)
            #             print("right eye: the most right",mesh_points[R_H_RIGHT][0][1])
            #             print("right eye: the most left",mesh_points[R_H_LEFT][0][1])
            #             print("right eye: the center of iris",center_right[1])

            iris_pos, ratio_h, ratio_v = iris_position_in_eye(center_right, mesh_points[R_H_RIGHT][0],
                                                              mesh_points[R_H_LEFT][0],
                                                              mesh_points[R_U_landmark][0],
                                                              mesh_points[R_D_landmark][0])
            #             print(iris_pos)

            cv.putText(frame, f"Iris pos: {iris_pos} {ratio_h:.2f} {ratio_v:.2f} ", (30, 30),
                       cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
        return frame


def normaliz_pixel(normalized_x, normalized_y, image_width, image_height):
    """
    input:
    x ,y :  both normalized to [0.0, 1.0] by the image width and the image height
    image_width, image_height: represent width and height of the image

    output:
    x , y: represent point in the image by pixels
    """
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def detect_eye_contact_unsupervised():
    """
    # the main method:
    """
    # face detection
    mp_face_detection = mp.solutions.face_detection
    # face mesh
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv.VideoCapture(0)

    with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5) as face_detection:
        while True:
            # Capture frame by frame
            ret, frame = cap.read()
            if not ret:
                break
            # may be using the lane to take frame vertically or horizontally
            # frame = cv.flip(frame,1)

            frame = crop_face_from_frame(face_detection, frame)
            frame = detect_iris_on_frame(mp_face_mesh, frame)

            cv.imshow('img', frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    detect_eye_contact_unsupervised(

    )
