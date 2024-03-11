import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import glob


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


def start_identification(url):
    url = url + "//video"
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    padding = 20
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    video_capture = cv2.VideoCapture(0)
    video_capture.open(url)
    images = glob.glob('Photos/*.jpg')
    print(images)

    known_face_encoding = []
    known_face_names = []
    for i in images:
        im = face_recognition.load_image_file(i)
        im_encoding = face_recognition.face_encodings(im)[0]
        known_face_encoding.append(im_encoding)
        name = os.path.basename(i).split('\\')[-1]
        known_face_names.append(name.split('.')[0])

    students = known_face_names.copy()
    present = []

    face_locations = []
    face_encodings = []
    face_names = []
    s = True

    today = datetime.now().strftime("%d-%m-%Y")

    f = open('Present-' + today + '.csv', 'w+', newline='')
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Time"])

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
        rgb_small_frame = small_frame[:, :, ::-1]
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        n1 = ""
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                name = ""
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    n1 = name

                face_names.append(name)
                if name in known_face_names:
                    if name in students:
                        students.remove(name)
                        print(students)
                        current_time = datetime.now().strftime("%H-%M-%S")
                        lnwriter.writerow([name, current_time])
                        # data = present.append([name,current_time])

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding,
                             frame.shape[0] - 1),
                   max(0, faceBox[0] - padding):
                   min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            cv2.putText(resultImg, f'{n1}',
                        (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Attendance System", resultImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    f.close()

    f = open('Absent-' + today + '.csv', 'w+', newline='')
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name"])
    for i in students:
        lnwriter.writerow([i])
    f.close()
