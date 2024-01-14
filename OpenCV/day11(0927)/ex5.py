import mediapipe as mp
import cv2

f_ms = mp.solutions.face_mesh
dw = mp.solutions.drawing_utils
dw_st = mp.solutions.drawing_styles

ms = f_ms.FaceMesh(max_num_faces = 2, refine_landmarks = True,
              min_detection_confidence = 0.5, min_tracking_confidence = 0.5)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    rec = ms.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if rec.multi_face_landmarks:
        for landmark in rec.multi_face_landmarks:
            dw.draw_landmarks(image = frame, landmark_list = landmark,
                              connections = f_ms.FACEMESH_TESSELATION,
                              landmark_drawing_spec = None,
                              connection_drawing_spec = dw_st.get_default_face_mesh_tesselation_style())

            # dw.draw_landmarks(image = frame, landmark_list = landmark,
            #                   connections = f_ms.FACEMESH_CONTOURS,
            #                   landmark_drawing_spec = None,
            #                   connection_drawing_spec = dw_st.get_default_face_mesh_contours_style())
    cv2.imshow("t", frame)
    key = cv2.waitKey(1)
    if key == 27: break
cap.release()
cv2.destroyAllWindows()