import cv2
import mediapipe as mp
import os


# face detection model
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)
cam = cv2.VideoCapture(0)


eye_open_threshold = 0.025
mouth_open_threshold = 0.03
squint_threshold = 0.015

#serious cat
def cat_serious(face_landmarks):
    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]
    right_eye_top = face_landmarks.landmark[386]
    right_eye_bottom = face_landmarks.landmark[374]
    eye_squint = (abs(left_eye_top.y - left_eye_bottom.y) + abs(right_eye_top.y - right_eye_bottom.y)) / 2
    return eye_squint < squint_threshold

# scared cat
def cat_shocked(face_landmarks):
    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]
    right_eye_top = face_landmarks.landmark[386]
    right_eye_bottom = face_landmarks.landmark[374]
    eye_open = (abs(left_eye_top.y - left_eye_bottom.y) + abs(right_eye_top.y - right_eye_bottom.y)) / 2
    return eye_open > eye_open_threshold

# tongue out cat
def cat_tongue_out(face_landmarks):
    mouth_top = face_landmarks.landmark[13]
    mouth_bottom = face_landmarks.landmark[14]
    mouth_open = abs(mouth_bottom.y - mouth_top.y)
    return mouth_open > mouth_open_threshold


#main loop
def main():
    while True:
        ret, image = cam.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        height, width = image.shape[:2]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        process_img = face_mesh.process(rgb_image)
        face_landmarks_list = process_img.multi_face_landmarks

        catimg = 'assets/neutral_cat.png'  # imagem padrão

        if face_landmarks_list:
            face_landmarks = face_landmarks_list[0]  # ← PEGA O ROSTO

            if cat_tongue_out(face_landmarks):
                catimg = 'assets/cat_tongue.png'
            elif cat_shocked(face_landmarks):
                catimg = 'assets/scared_cat.png'
            elif cat_serious(face_landmarks):
                catimg = 'assets/serious_cat.png'
            else:
                catimg = 'assets/normal_cat.png'

            for lm in face_landmarks.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow('Face Detection', image)

        cat = cv2.imread(catimg)
        if cat is not None:
            cat = cv2.resize(cat, (640, 480))
            cv2.imshow('cat img', cat)
        else:
            blank_img = image * 0
            cv2.putText(
                blank_img,
                'No cat expression detected',
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.imshow('cat img', blank_img)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    