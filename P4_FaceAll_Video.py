import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    static_image_mode = False, 
    max_num_faces=2, 
    min_detection_confidence = 0.5) as face_mesh:
    #image = cv2.imread("C://Users\yulit\Desktop\Doctorado\Seguimiento_Tracking\imagen_03.jpg")
    #height, width, _ = image.shape
    #image_rgb= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #results = face_mesh.process(image_rgb)
    
    #print("Face landmarks: ", results.multi_face_landmarks)
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                #mp_drawing.draw_landmarks(frame,face_landmarks,mp_face_mesh.FACEMESH_TESSELATION,mp_drawing.DrawingSpec((0,255,255), thickness=1,circle_radius=1),mp_drawing.DrawingSpec((255,0,255), thickness=1))
                mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(image=frame,landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                #mp_drawing.draw_landmarks(image=image,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_IRISES,landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
    
        cv2.imshow("Frame",frame)
        k = cv2.waitKey(1) & 0xFF
        if(k == 27):
            break
        
cap.release()
cv2.destroyAllWindows()
    