import sys, time, os, cv2, qdarktheme, configparser, webbrowser
import PyQt6.QtWidgets as Widgets
import PyQt6.QtCore as Core
from PyQt6 import QtGui
from moviepy.editor import *

config = configparser.ConfigParser()
app = Widgets.QApplication(sys.argv)
app_title = "Detector"
media_source = 0
record_on = False
prints_on = False
view_on = True
multiscale_factor = 1.8
detection_color = (0, 255, 0)
save_folder = os.path.dirname(os.path.abspath(__file__)) + "/saved_detections"
trained_data = cv2.CascadeClassifier("haarcascades/frontalface.xml")
camera_source = 0
cameras = []

class ClickableQLabel(Widgets.QLabel):
    clicked=Core.pyqtSignal()
    def mousePressEvent(self, ev):
        self.clicked.emit()

class QHLine(Widgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(Widgets.QFrame.Shape.HLine)
        self.setFrameShadow(Widgets.QFrame.Shadow.Sunken)

class GUI(Widgets.QMainWindow):
    def __init__(self, screen_size):
        super().__init__()
        qdarktheme.setup_theme(custom_colors={"primary": "#FFFFFF"})
        # Screen and App Sizes
        self.screen_width = screen_size.width()
        self.screen_height = screen_size.height()
        self.app_width = 600
        self.app_height = 250
        self.app_x = int((self.screen_width - self.app_width) / 2)
        self.app_y = int((self.screen_height - self.app_height) / 2)
        # Init Window
        self.setFixedSize(self.app_width, self.app_height)
        self.setWindowTitle(app_title)
        self.setWindowIcon(QtGui.QIcon("favicon.ico"))
        app.setStyleSheet("QLabel{font-size: 14px;}")
        # Main App Layout
        self.tabs = Widgets.QTabWidget()

        # Detector TAB
        self.detector_widget = Widgets.QWidget()
        self.detector_layout = Widgets.QVBoxLayout()
        self.detector_widget.setLayout(self.detector_layout)

        self.choose_harr_layout = Widgets.QHBoxLayout()

        self.harr_combo_label = Widgets.QLabel("Detect:")
        self.choose_harr_layout.addWidget(self.harr_combo_label)

        self.harr_combo = Widgets.QComboBox()
        self.harr_combo.addItems(['Frontal Face', 'Full Body', 'Smile', 'Frontal Cat'])
        self.harr_combo.currentTextChanged.connect(self.chooseDetection)
        self.choose_harr_layout.addWidget(self.harr_combo)

        self.detector_layout.addLayout(self.choose_harr_layout)

        self.choose_source_layout = Widgets.QHBoxLayout()

        self.choose_source_label = Widgets.QLabel("Source:")
        self.choose_source_layout.addWidget(self.choose_source_label)

        self.source_image_button = Widgets.QPushButton("Image")
        self.source_image_button.setStyleSheet('QPushButton { background-color: #808080 }')
        self.source_image_button.clicked.connect(lambda: self.toggleSource(0))
        self.choose_source_layout.addWidget(self.source_image_button)

        self.source_video_button = Widgets.QPushButton("Video")
        self.source_video_button.setStyleSheet('QPushButton { background-color: #4d4d4d }')
        self.source_video_button.clicked.connect(lambda: self.toggleSource(1))
        self.choose_source_layout.addWidget(self.source_video_button)

        self.source_camera_button = Widgets.QPushButton("Camera")
        self.source_camera_button.setStyleSheet('QPushButton { background-color: #4d4d4d }')
        self.source_camera_button.clicked.connect(lambda: self.toggleSource(2))
        self.choose_source_layout.addWidget(self.source_camera_button)

        self.detector_layout.addLayout(self.choose_source_layout)

        self.choose_camera_layout = Widgets.QHBoxLayout()

        self.choose_camera_label = Widgets.QLabel("Camera:")
        self.choose_camera_layout.addWidget(self.choose_camera_label)
        
        self.choose_camera_combo = Widgets.QComboBox()
        self.choose_camera_combo.addItems(cameras)
        self.choose_camera_combo.currentTextChanged.connect(self.changeCamera)
        self.choose_camera_combo.setEnabled(False)
        self.choose_camera_layout.addWidget(self.choose_camera_combo)

        self.choose_camera_test = Widgets.QPushButton("Test")
        self.choose_camera_test.setToolTip("Test selected camera")
        self.choose_camera_test.setEnabled(False)
        self.choose_camera_test.clicked.connect(self.testCamera)
        self.choose_camera_layout.addWidget(self.choose_camera_test)

        self.detector_layout.addLayout(self.choose_camera_layout)

        self.record_detection_layout = Widgets.QHBoxLayout()

        self.options_label = Widgets.QLabel("Options:")
        self.record_detection_layout.addWidget(self.options_label)

        self.record_switch = Widgets.QPushButton("Record", self)
        self.record_switch.setStyleSheet('QPushButton { background-color: red;}')
        self.record_switch.setToolTip("Enable/Disable recording of the detections into a video")
        self.record_switch.clicked.connect(self.toggleRecord)
        self.record_detection_layout.addWidget(self.record_switch)

        self.detector_layout.addLayout(self.record_detection_layout)

        self.start_button = Widgets.QPushButton("Start", self)
        self.start_button.clicked.connect(self.detect)
        self.detector_layout.addWidget(self.start_button)

        self.exit_label = Widgets.QLabel("Press Q to exit detection window")
        self.exit_label.setAlignment(Core.Qt.AlignmentFlag.AlignCenter)
        self.detector_layout.addWidget(self.exit_label, alignment=Core.Qt.AlignmentFlag.AlignCenter)

        # Config TAB
        self.config_widget = Widgets.QWidget()
        self.config_layout = Widgets.QVBoxLayout()
        self.config_widget.setLayout(self.config_layout)

        self.multiscale_layout = Widgets.QHBoxLayout()

        self.multiscale_label = Widgets.QLabel("Multiscale Value:")
        self.multiscale_layout.addWidget(self.multiscale_label)
        
        self.multiscale_combo = Widgets.QComboBox()
        self.multiscale_combo.addItems(['1.0', '1.2', '1.4', '1.6', '1.8', '2.0'])
        self.multiscale_combo.currentTextChanged.connect(self.changeMultiscale)
        
        self.multiscale_layout.addWidget(self.multiscale_combo)

        self.config_layout.addLayout(self.multiscale_layout)

        self.detection_color_layout = Widgets.QHBoxLayout()

        self.detection_color_label = Widgets.QLabel("Detection Square Color:")
        self.detection_color_layout.addWidget(self.detection_color_label)

        self.color_square = Widgets.QCheckBox()
        self.color_square.setEnabled(False)
        self.color_square.setStyleSheet(''' QCheckBox::indicator { background-color: rgb(0, 255, 0); border: 1px solid black; } ''')
        self.detection_color_layout.addWidget(self.color_square)

        self.detection_color_picker = Widgets.QPushButton("Change")
        self.detection_color_picker.clicked.connect(self.openColorDialog)
        self.detection_color_layout.addWidget(self.detection_color_picker)

        self.config_layout.addLayout(self.detection_color_layout)

        self.change_folder_layout = Widgets.QHBoxLayout()

        self.change_folder_label = Widgets.QLabel("Recording/Prints Folder:")
        self.change_folder_layout.addWidget(self.change_folder_label)

        self.change_folder_current = ClickableQLabel(save_folder)
        self.change_folder_current.setCursor(QtGui.QCursor(Core.Qt.CursorShape.PointingHandCursor))
        self.change_folder_current.setToolTip(save_folder)
        self.change_folder_current.clicked.connect(lambda: os.startfile(save_folder))
        self.change_folder_current.setFixedWidth(180)
        self.change_folder_layout.addWidget(self.change_folder_current)
        
        self.change_folder_button = Widgets.QPushButton("Change")
        self.change_folder_button.clicked.connect(lambda: self.chooseDirectory(save_folder, self.change_folder_current))
        self.change_folder_layout.addWidget(self.change_folder_button)

        self.config_layout.addLayout(self.change_folder_layout)

        # About TAB
        self.about_widget = Widgets.QWidget()
        self.about_layout = Widgets.QVBoxLayout()
        self.about_widget.setLayout(self.about_layout)

        self.about_detector = Widgets.QLabel("Detector is an application capable of detecting objects based on the Haar Cascade technique.")
        self.about_detector.setWordWrap(True) 
        self.about_layout.addWidget(self.about_detector)

        self.about_layout.addWidget(QHLine())

        self.dev_by = Widgets.QLabel("Developed by Bruno Cruz")
        self.about_layout.addWidget(self.dev_by)

        self.link_layout = Widgets.QHBoxLayout()

        self.website = ClickableQLabel("Website")
        self.website.clicked.connect(lambda: webbrowser.open("https://xbdrcx.github.io"))
        self.website.setCursor(QtGui.QCursor(Core.Qt.CursorShape.PointingHandCursor))
        self.website.setStyleSheet("QLabel{font-weight: bold;}")
        self.website.setFixedWidth(100)
        self.link_layout.addWidget(self.website, alignment=Core.Qt.AlignmentFlag.AlignLeft)
        
        self.git = ClickableQLabel("GitHub")
        self.git.clicked.connect(lambda: webbrowser.open("https://github.com/xbdrcx"))
        self.git.setCursor(QtGui.QCursor(Core.Qt.CursorShape.PointingHandCursor))
        self.git.setStyleSheet("QLabel{font-weight: bold;}")
        self.git.setFixedWidth(100)
        self.link_layout.addWidget(self.git, alignment=Core.Qt.AlignmentFlag.AlignLeft)

        self.about_layout.addLayout(self.link_layout)

        # Run
        self.tabs.addTab(self.detector_widget, 'Detector')
        self.tabs.addTab(self.config_widget, 'Configuration')
        self.tabs.addTab(self.about_widget, 'About')
        self.main_window = Widgets.QWidget()
        self.main_layout = Widgets.QVBoxLayout()
        self.main_window.setLayout(self.main_layout)
        self.main_layout.addWidget(self.tabs)
        self.verifySettings()
        self.setCentralWidget(self.main_window)
        self.show()

    def changeCamera(self):
        global camera_source
        camera_source = int(self.choose_camera_combo.currentText())
        print(camera_source)

    def testCamera(self): 
        cam = cv2.VideoCapture(int(self.choose_camera_combo.currentText())) 
        while(True): 
            ret, frame = cam.read() 
            cv2.imshow('Camera Test', frame) 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        cam.release()  
        cv2.destroyAllWindows() 

    def changeMultiscale(self):
        global multiscale_factor
        multiscale_factor = float(self.multiscale_combo.currentText())
        self.saveSetting("MULTISCALE", str(self.multiscale_combo.currentIndex()))

    def chooseDirectory(self, dirToChange, dirLabel):
        new_dir = str(Widgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        dirToChange = new_dir
        self.saveSetting("SAVE_FOLDEr", new_dir)
        if dirLabel != None:
            dirLabel.setText(dirToChange)
            dirLabel.setToolTip(dirToChange)
            dirLabel.clicked.disconnect()
            dirLabel.clicked.connect(lambda: os.startfile(dirToChange))

    def openColorDialog(self):
        global detection_color
        color = Widgets.QColorDialog.getColor()
        if color.isValid():
            self.color_square.setStyleSheet("QCheckBox::indicator { background-color: " + color.name() + "; border: 1px solid black; }")
            detection_color = (color.red(), color.green(), color.blue())
            self.saveSetting("COLOR", str(detection_color))

    def detect(self):
        try:
            self.changeButtonState(False)
            if media_source == 1 or media_source == 2:
                videoDetection()
            else:
                imageDetection()
            self.changeButtonState(True)
        except Exception as e:
            self.errorDialog(e)

    def toggleSource(self, source):
        global media_source
        media_source = source
        self.source_image_button.setStyleSheet('QPushButton { background-color: #4d4d4d }')
        self.source_video_button.setStyleSheet('QPushButton { background-color: #4d4d4d }')
        self.source_camera_button.setStyleSheet('QPushButton { background-color: #4d4d4d }')
        self.choose_camera_combo.setEnabled(False)
        self.choose_camera_test.setEnabled(False)
        if source == 0:
            self.source_image_button.setStyleSheet('QPushButton { background-color: #808080 }')
        elif source == 1:
            self.source_video_button.setStyleSheet('QPushButton { background-color: #808080 }')
        else:
            self.choose_camera_combo.setEnabled(True)
            self.choose_camera_test.setEnabled(True)
            self.source_camera_button.setStyleSheet('QPushButton { background-color: #808080 }')

    def toggleRecord(self):
        global record_on
        if record_on:
            record_on = False
            self.record_switch.setStyleSheet('QPushButton { background-color: red;}')
        else:
            record_on = True
            self.record_switch.setStyleSheet('QPushButton { background-color: green;}')

    def changeButtonState(self, value):
        self.harr_combo.setEnabled(value)
        self.source_image_button.setEnabled(value)
        self.source_video_button.setEnabled(value)
        self.source_camera_button.setEnabled(value)
        self.record_switch.setEnabled(value)
    
    def chooseDetection(self, detection):
        global trained_data
        if detection == "Frontal Face":
            trained_data = cv2.CascadeClassifier("haarcascades/frontalface.xml")
        elif detection == "Full Body":
            trained_data = cv2.CascadeClassifier("haarcascades/fullbody.xml")
        elif detection == "Smile":
            trained_data = cv2.CascadeClassifier("haarcascades/smile.xml")
        else:
            trained_data = cv2.CascadeClassifier("haarcascades/frontalcat.xml")

    def verifySettings(self):
        global detection_color
        global multiscale_factor
        global save_folder
        global config
        if os.path.exists('./config.ini'):
            # If EXISTS
            with open("config.ini", "r") as config_file:
                config.read_file(config_file)
                self.multiscale_combo.setCurrentIndex(int(config.get("DEFAULT", "MULTISCALE")))
                multiscale_factor = float(self.multiscale_combo.currentText())
                detection_color = config.get("DEFAULT", "COLOR")
                save_folder = config.get("DEFAULT", "SAVE_FOLDER")
                self.change_folder_current.setText(config.get("DEFAULT", "SAVE_FOLDER"))
        else:
            # If DOESNT EXIST
            config.read("config.ini")
            config['DEFAULT']['MULTISCALE'] = str(1.0)
            config['DEFAULT']['COLOR'] = str((0, 255, 0))
            config['DEFAULT']['SAVE_FOLDER'] = os.path.dirname(os.path.abspath(__file__)) + "/saved_detections"
            with open("config.ini", "w") as config_file:
                config.write(config_file)

    def saveSetting(self, setting, value):
        global config
        config.read("config.ini")
        config['DEFAULT'][setting] = value
        with open("config.ini", "w") as config_file:
            config.write(config_file)

def getAvailableCameras():
    global camera_source
    max_cameras = 10
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            print(f"Camera index {i:02d} not found...")
            continue
        cameras.append(str(i))
        cap.release()
        print(f"Camera index {i:02d} OK!")
    print(f"Cameras found: {cameras}")
    camera_source = int(cameras[0])

def createMessage(message):
    messageWindow = Widgets.QMessageBox()
    messageWindow.setWindowTitle("Error")
    messageWindow.setWindowIcon(QtGui.QIcon("favicon.ico"))
    messageWindow.setText(message)
    messageWindow.exec()

def videoDetection():
    try:
        if media_source == 1:
            # Video
            filename = Widgets.QFileDialog.getOpenFileName(None, "Choose Video", "examples/", "Video Files (*.mp4 *.mov *.webm)")
            vid_source = cv2.VideoCapture(filename[0])
            name = filename[0].split("/")[-1].split(".")[0]
        else:
            # Camera
            vid_source = cv2.VideoCapture(camera_source)
        if record_on:
            if vid_source.isOpened() and media_source == 1:
                width  = vid_source.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = vid_source.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = vid_source.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), float(fps), (int(width), int(height)))
            elif vid_source.isOpened() and media_source == 2:
                frame_width = int(vid_source.get(3)) 
                frame_height = int(vid_source.get(4)) 
                size = (frame_width, frame_height)
                out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
        while True:
            successful_frame_read, frame = vid_source.read()
            if successful_frame_read == False:
                break
            grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if multiscale_factor == 1.0:
                coordinates = trained_data.detectMultiScale(grayscaled_img)
            else:
                coordinates = trained_data.detectMultiScale(grayscaled_img, scaleFactor=multiscale_factor)
            for(x, y, w, h) in coordinates:
                cv2.rectangle(frame, (x, y), (x+w*2, y+h*2), (0, 255, 0), 2)
                if prints_on:
                    print(x - w, y - h)
                    print(w * 2, h * 2)
                    img_captured = frame[y-h:y-h+h*2, x-w:x-w+w*2]
                    cv2.imwrite(filename[0] + "_" + str(x+y) + ".jpg", img_captured)
            if record_on:
                out.write(frame)
            if view_on:
                cv2.imshow('Detector', frame)
                key = cv2.waitKey(1)
                if key == 81 or key == 113:
                    break
            else:
                window = createMessage("Detection running, please wait.")
                window.show()
        cv2.destroyWindow("Detector")
        vid_source.release()
        if record_on:
            out.release()
            time.sleep(0.5)
            video_clip = VideoFileClip("output.avi")
            if media_source == 1:
                final_clip = video_clip.set_audio(VideoFileClip(filename[0]).audio)
                final_clip.write_videofile(name + ".mp4")
            else:
                final_clip = video_clip.without_audio()
                final_clip.write_videofile("cam.mp4")
            os.remove("output.avi")
        if view_on == False:
            window.exit()
    except Exception as e:
        print(e)
        createMessage("Something went wrong while loading/detecting video feed.")
    cv2.destroyAllWindows()

def imageDetection():
    try:
        filename = Widgets.QFileDialog.getOpenFileName(None, "Choose Image", "examples/", "Image Files (*.jpg *.png)")
        if filename[0] != "":
            name = filename[0].split("/")[-1]
            img = cv2.imread(filename[0])
            grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if multiscale_factor == 1.0:
                coordinates = trained_data.detectMultiScale(grayscaled_img)
            else:
                coordinates = trained_data.detectMultiScale(grayscaled_img, scaleFactor=multiscale_factor)
            for (x, y, w, h) in coordinates:
                cv2.rectangle(img , (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.imshow('Detector', img)
            if record_on:
                cv2.imwrite(name, img)
            key = cv2.waitKey()
            if key == 81 or key == 113:
                cv2.destroyWindow("Detector")
    except Exception as e:
        print(e)
        createMessage("Something went wrong while loading/detecting image file.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    getAvailableCameras()
    gui = GUI(screen_size=app.primaryScreen().size())
    sys.exit(app.exec())