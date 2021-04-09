# Import all the Required modules

from PIL import Image, ImageTk
import pyttsx3
import tkinter as tk
import cv2
from keras.models import model_from_json
import operator
from string import ascii_uppercase


# Create the main class of the application
class Application:
    def __init__(self):

        # Initiate some of the properties of text to voice function
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 200)  # Set the speed of the speaker
        self.voices = self.engine.getProperty("voices")  # Get all the available voices
        self.engine.setProperty("voice", self.voices[1].id)

        # To get the access to the camera and get frames
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Here we create a variable to store the address of the models
        self.directory = 'model'

        # Import all the required Neural Netwroks
        self.json_file = open(self.directory + "\model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights(self.directory + "\model-bw.h5")

        self.json_file_dru = open(self.directory + "\model-bw_dru.json", "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()
        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights(self.directory + "\model-bw_dru.h5")

        self.json_file_tkdi = open(self.directory + "\model-bw_tkdi.json", "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights(self.directory + "\model-bw_tkdi.h5")

        self.json_file_smn = open(self.directory + "\model-bw_smn.json", "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()
        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights(self.directory + "\model-bw_smn.h5")

        self.ct = {}
        self.ct['Blank'] = 0
        self.blank_flag = 0

        # Here "i" represent each character from A-Z. and set 0 to each unit of the ct array
        for i in ascii_uppercase:
            self.ct[i] = 0

        self.root = tk.Tk()
        self.root.title("American-Sign-to-Speech")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("800x800")  # Set the intital size of the window

        self.panel = tk.Label(self.root)  # give the location of the colored screen window
        self.panel.place(x=135, y=10, width=640, height=640)

        self.panel2 = tk.Label(self.root)  # Give the location of the white and black screen window
        self.panel2.place(x=460, y=95, width=310, height=310)

        self.T = tk.Label(self.root)  # location of the main title of the application
        self.T.place(x=31, y=17)
        self.T.config(text="American-Sign-to-Speech", font=("courier", 40, "bold"))

        self.panel3 = tk.Label(self.root)  # set the location of the predictive symmol/ Character value
        self.panel3.place(x=450, y=610)

        self.T1 = tk.Label(self.root)  # set the location of the character TEXT in the window
        self.T1.place(x=30, y=610)
        self.T1.config(text="Character :", font=("Courier", 40, "bold"))

        # Create Following variables and initialize them as empty variables
        self.current_symbol = "Empty"
        self.photo = "Empty"

        # Call the loop function 
        self.video_loop()

    def video_loop(self):

        # Get the Frame 
        # ok returns either TRUE or FALSE, and frame return an array of the frame
        ok, frame = self.vs.read()

        # If Ok == TRUE, keep looping
        if ok:
            # Flip the image, here the flip(source, flipcode)
            cv2image = cv2.flip(frame, 1)

            # Specify the location
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            # Specify the frame, along with the co-ordinates
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

            # Convert the color of the window 
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            # Put the current frame in current_image variable
            self.current_image = Image.fromarray(cv2image)

            # Use for configuration
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # Define the x1, x2 and y1, y2 of the image
            cv2image = cv2image[y1:y2, x1:x2]

            # Convert the image in gray and then use gaussian function to blur the image
            # To highlight the image features
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Pass it to predict the symbol of the current frame of the camera
            self.predict(res)
            self.current_image2 = Image.fromarray(res)

            # Now pass the current symbol, predictive word and string will be pass to our interface
            # TO show the results
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol, font=("Courier", 50))  # Current Symbol

        self.root.after(30, self.video_loop)

    def predict(self, test_image):

        # Resize the image using "resize" function of the cv2
        test_image = cv2.resize(test_image, (128, 128))

        # MAIN Layer to predict from A-Z. Its the main predictive layer
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        # Sub layers for a specific group of characters.
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))

        prediction = {}
        prediction['Blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        # LAYER 1: IT covers all characters from A-Z
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        # LAYER 2 start from here

        # For a group of character, which includes D, R, and U.
        if (self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
            prediction = {}
            prediction['D'] = result_dru[0][0]
            prediction['R'] = result_dru[0][1]
            prediction['U'] = result_dru[0][2]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        # For a group of character, which includes D, I, k,and T.
        if (
                self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
            prediction = {}
            prediction['D'] = result_tkdi[0][0]
            prediction['I'] = result_tkdi[0][1]
            prediction['K'] = result_tkdi[0][2]
            prediction['T'] = result_tkdi[0][3]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        # For a group of character, which includes M, N, and S.
        if self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S':
            prediction1 = {}
            prediction1['M'] = result_smn[0][0]
            prediction1['N'] = result_smn[0][1]
            prediction1['S'] = result_smn[0][2]
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            if prediction1[0][0] == 'S':
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]

        # If there is nothing in the screen, then it is "Blank"
        if self.current_symbol == 'Blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1
        if self.ct[self.current_symbol] > 60:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['Blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct['Blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol == 'Blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

        self.engine.say(self.current_symbol)
        self.engine.runAndWait()

    # Destroy the window
    def destructor(self):
        print("Terminate the Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


# Execution of the program starts here

print("Starting the Application...")
pba = Application()
pba.root.mainloop()
