import face_recognition
import numpy as np
import PIL.Image
import PIL.ImageDraw
from PIL import ImageDraw, ImageFont
import cv2


class Recognition:
    """" This is a class for visualizing images

    Attributes:
        known_faces (list): a list of recognized faces
        known_faces_names (list): a list of the names of recognized faces
    """
    def __init__(self):
        self.known_faces = []
        self.known_faces_names = []

    def encode_image(self, name, file_path):
        """"This is a function that encode images into a A list of 128-dimensional face encodings
         (one for each face in the image)
        Args:
            file_path (string): directory of image to be encode
        Returns:
              None
        """
        try:
            face = face_recognition.load_image_file(file_path)
            encoded_image = face_recognition.face_encodings(face)[0]
        except IndexError:
            encoded_image = []
        if len(encoded_image):
            self.known_faces.append(encoded_image)
            self.known_faces_names.append(name)

    def identify_from_picture(self, file_path, save_path=None, threshold=0.5):
        """ Function to identify people face from picture
        Args:
            file_path (string) : the path to the file we want to recognize
            save_path (string) : the path to save the file we recognize
            threshold (float) : minimum distance between two people to encode them
         Returns:
               None
        """
        # load jpg into numpy array
        unknown_face = face_recognition.load_image_file(file_path)
        print(unknown_face)
        face_locations = face_recognition.face_locations(unknown_face)
        face_encodings = face_recognition.face_encodings(unknown_face, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_faces, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
            print(face_distances)
            best_match_index = np.argmin(face_distances)
            diff = min(face_distances)
            if matches[best_match_index]:
                if diff < threshold:
                    print(diff)
                    name = self.known_faces_names[best_match_index]

            face_names.append(name)

        pil_image = PIL.Image.fromarray(unknown_face)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            font = ImageFont.truetype(r'C:\Users\DayoBanjo\Downloads\Open_Sans\OpenSans-Regular.ttf', 40)
            draw = PIL.ImageDraw.Draw(pil_image)
            draw.rectangle([left, top, right, bottom], outline="blue")
            draw.text((left, top), name, font=font, fill='black')

        try:
            pil_image.show()
            if save_path is not None:
                pil_image.save(save_path)
        except EOFError:
            print("No image found")
        return

    def identify_from_web_cam(self, save_path=None, threshold=0.5):

        video_capture = cv2.VideoCapture(0)

        process_this_frame = True
        face_locations = []
        face_encodings = []
        face_names = []

        # so, convert them from float to integer to set resolution.
        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))

        size = (frame_width, frame_height)
        result = cv2.VideoWriter(save_path,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            if save_path is not None:
                result.write(rgb_small_frame)

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                print("face location", face_locations, len(face_locations))
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                print(len(face_encodings))

                name = "Unkown"
                matches = None
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_faces, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
                    print("face distance", face_distances)
                    best_match_index = np.argmin(face_distances)
                    try:
                        diff = min(face_distances)
                    except ValueError:
                        diff = 1.0
                    if matches[best_match_index]:
                        if diff < threshold:
                            name = self.known_faces_names[best_match_index]
                            print(name)
                    face_names.append(name)

            # Display the results
            process_this_frame = not process_this_frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                print("face location", face_locations, "face names", face_names)
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if save_path is not None:
                result.write(frame)
                # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    gidimo_staff_recognition = Recognition()
    gidimo_staff_recognition.encode_image("Dayo", "../data/me1.jpeg")
    gidimo_staff_recognition.encode_image("Lola", "../data/lola.jpg")
    gidimo_staff_recognition.encode_image("Tunji", "../data/tunji_adegbesan.jpg")
    gidimo_staff_recognition.encode_image("Chidoze", "../data/chidozie.jpg")
    gidimo_staff_recognition.encode_image("Chibuke", "../data/chibu.jpg")
    gidimo_staff_recognition.encode_image("Mu'awiyah", "../data/mumu.jpg")
    gidimo_staff_recognition.encode_image("Godwin", "../data/godwin.jpg")
    gidimo_staff_recognition.encode_image("Chidalu", "../data/chidalu.jpg")
    gidimo_staff_recognition.identify_from_picture("../data/chidozie_1.jpg", save_path="../data/chidi.jpg")
