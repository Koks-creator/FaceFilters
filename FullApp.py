from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.config import Config
import json
import cv2
import mediapipe as mp
import numpy as np
from math import hypot, degrees, atan2, cos, sin
import threading
from functools import partial
import datetime


Config.set('graphics', 'width', '600')
Config.set('graphics', 'height', '650')


with open("data.json") as f:
    FILTERS_DICT = json.load(f)


classes = list(FILTERS_DICT.keys())


class DummyScreen(Screen):
    def __init__(self, **kwargs):
        super(DummyScreen, self).__init__(**kwargs)


filter_index = 0


class MainScreen(Screen):
    container = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)
        Clock.schedule_once(self.setup_scrollview, 1)
        self.filter_index = 0

    def setup_scrollview(self, dt):
        self.container.bind(minimum_height=self.container.setter('height'))
        self.add_text_inputs()

    @staticmethod
    def update_index(ind):
        global filter_index
        filter_index = int(ind.text)
        # print(ind.text)

    @staticmethod
    def index_enlarge():
        global filter_index
        filter_index -= 1

        if filter_index < 0:
            filter_index = len(classes) - 1

    @staticmethod
    def index_diminish():
        global filter_index
        filter_index += 1

        if filter_index > len(classes) - 1:
            filter_index = 0

    def add_text_inputs(self):
        for i in range(len(classes)):
            # self.container.add_widget(Label(text="Label {}".format(x), size_hint_y=None, height=40))
            img_path = r"FilterImgs/" + FILTERS_DICT[classes[i]]["ImgPath"]
            btn = Button(
                         color=(1, 0, .65, 1),
                         background_normal=img_path,
                         background_down=img_path,
                         height=70,
                         size_hint_y=None,
                         border=(0, 1, 1, 0),
                         text=str(i),
                         font_size='1px'
                         )
            btn.bind(on_press=self.update_index)
            self.container.add_widget(btn)


class ScreenManagement(ScreenManager):
    pass


class Filter(App):
    def build(self):
        # daemon=True means kill this thread when app stops
        threading.Thread(target=self.webcam_output, daemon=True).start()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.mp_drawing = mp.solutions.drawing_utils

        self.sm = ScreenManagement()
        self.main_screen = MainScreen(name="main")
        self.dummy_screen = DummyScreen(name="dummy")
        self.sm.add_widget(self.main_screen)
        self.sm.add_widget(self.dummy_screen)

        return self.sm

    @staticmethod
    def rotate(image: np.array, angle: float, center=None, scale=1.0) -> np.array:
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def get_landmarks(self, img: np.array, draw=False) -> list:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        landmark_list = []

        result_class = self.face_mesh.process(img_rgb)
        result = result_class.multi_face_landmarks

        if result:
            for landmarks in result:
                if draw:
                    self.mp_drawing.draw_landmarks(img, landmarks, self.mp_face_mesh.FACE_CONNECTIONS)

                for id, lm in enumerate(landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([x, y])

        return landmark_list

    def webcam_output(self):
        self.do_vid = True  # flag to stop loop
        self.save_screen = False
        self.save_frame = False

        # make a window for use by cv2
        # flags allow resizing without regard to aspect ratio
        cv2.namedWindow('Hidden', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)

        # resize the window to (0,0) to make it invisible
        cv2.resizeWindow('Hidden', 0, 0)
        cam = cv2.VideoCapture(0)

        # start processing loop
        while self.do_vid:

            # Filter
            current_filter = classes[filter_index]

            ret, frame = cam.read()
            if ret is False:
                self.do_vid = False
                break

            lm_list = self.get_landmarks(frame, draw=False)
            if len(lm_list) > 0:
                # Setting up vars
                filter_img = cv2.imread(r"FilterImgs/" + FILTERS_DICT[current_filter]["ImgPath"])
                filter_adj = FILTERS_DICT[current_filter]["Adj"]

                filter_height_adj = FILTERS_DICT[current_filter]["HeightAdj"]
                filter_left_adj = FILTERS_DICT[current_filter]["LeftAdj"]

                filter_right = lm_list[FILTERS_DICT[current_filter]["Right"]]
                filter_left = lm_list[FILTERS_DICT[current_filter]["Left"]]
                filter_bottom = lm_list[FILTERS_DICT[current_filter]["Bottom"]]
                filter_top = [lm_list[FILTERS_DICT[current_filter]["Top"]][0],
                              lm_list[FILTERS_DICT[current_filter]["Top"]][1]]

                filter_left[0] = filter_left[0] + filter_left_adj
                filter_center = [
                    int((filter_left[0] + filter_right[0]) / 2),
                    int((filter_top[1] + filter_bottom[1]) / 2) + filter_height_adj,
                ]

                # Width and height of filter img
                filter_h, filter_w, _ = filter_img.shape
                filter_width = int(
                    hypot(filter_left[0] - filter_right[0], filter_left[1] - filter_right[1]) * filter_adj)
                filter_height = int(filter_width * (filter_h / filter_w))

                # Angle
                bottom_face = lm_list[152]
                top_face = lm_list[10]

                ang_org = degrees(atan2(top_face[0] - bottom_face[0], top_face[1] - bottom_face[1]))
                ang = 180 - (ang_org * -1)
                if filter_height_adj != 0:
                    if 10 < ang < 100:
                        filter_center[0] = filter_center[0] - (filter_height_adj * -1)
                    if 280 < ang < 350:
                        filter_center[0] = filter_center[0] + (filter_height_adj * -1)

                # Rotate filter img
                filter_img = self.rotate(filter_img, ang)
                # Resizing filter img
                filter_img = cv2.resize(filter_img, (filter_width, filter_height))

                # Mask
                filter_img_gray = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(filter_img_gray, 0, 255, cv2.THRESH_BINARY_INV)

                # Filter ROI
                top_left = (int(filter_center[0] - filter_width / 2), int(filter_center[1] - filter_height / 2))

                filter_roi = frame[top_left[1]:top_left[1] + filter_height,
                                   top_left[0]:top_left[0] + filter_width
                                   ]

                try:
                    # Filter_area
                    filter_area = cv2.bitwise_and(filter_roi, filter_roi, mask=mask)

                    # Final
                    final_filter = cv2.add(filter_area, filter_img)
                    cv2.imshow("final_filter", final_filter)

                    frame[top_left[1]:top_left[1] + filter_height,
                          top_left[0]:top_left[0] + filter_width
                          ] = final_filter
                except Exception:
                    pass

                if self.save_frame:
                    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                    cv2.imwrite(f"SavedImages/{current_time}.png", frame)
                    self.save_frame = False

                if self.save_screen:
                    DummyScreen.siema = frame
                    Clock.schedule_once(partial(self.update_frame, frame))

                    self.save_screen = False

            # send this frame to the kivy Image Widget
            # Must use Clock.schedule_once to get this bit of code
            # to run back on the main thread (required for GUI operations)
            # the partial function just says to call the specified method with the provided argument (Clock adds a time argument)
            frame = cv2.resize(frame, (640, 650))
            Clock.schedule_once(partial(self.show_frame, frame))
            cv2.imshow('Hidden', frame)
            cv2.waitKey(1)

        cam.release()
        cv2.destroyAllWindows()

    def save_screen_function(self):
        self.save_screen = True
        self.sm.current = "dummy"

    def save_frame_function(self):
        self.save_frame = True
        self.sm.current = "main"

    def show_frame(self, frame, dt):
        # display the current video frame in the kivy Image widget

        # create a Texture the correct size and format for the frame
        # print(frame.shape)
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')

        # copy the frame data into the texture
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')

        # flip the texture (otherwise the video is upside down
        texture.flip_vertical()

        # actually put the texture in the kivy Image widget
        self.main_screen.ids.vid.texture = texture

    def update_frame(self, frame, dt):
        frame = cv2.resize(frame, (640, 650))
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()

        self.dummy_screen.ids.vid2.texture = texture


Filter().run()



