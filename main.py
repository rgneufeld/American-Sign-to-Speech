from __future__ import print_function
from PIL import Image, ImageTk
import pyttsx3
import tkinter as tk
import PIL as pl
import numpy as np
import cv2
from keras.models import model_from_json
import operator
import time
import sys, os
import argparse
import matplotlib.pyplot as pl
from tkinter import Tk
import hunspell 
from string import ascii_uppercase


class Application:
    def __init__(self):
	   
        # Intiate some of the properties of text to voice function
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate",200)   # Set the speed of the speaker
        self.voices = self.engine.getProperty("voices")    # Get all the available voices 
        self.engine.setProperty("voice",self.voices[1].id)
        

        self.hs = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Here we create a variable to store the address of the models
        self.directory = 'model'

        # Import all the required Neural Netwroks
        self.json_file = open(self.directory+"\model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights(self.directory+"\model-bw.h5")

        self.json_file_dru = open(self.directory+"\model-bw_dru.json" , "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()
        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights(self.directory+"\model-bw_dru.h5")

        self.json_file_tkdi = open(self.directory+"\model-bw_tkdi.json" , "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights(self.directory+"\model-bw_tkdi.h5")

        self.json_file_smn = open(self.directory+"\model-bw_smn.json" , "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()
        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights(self.directory+"\model-bw_smn.h5")
        
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        # Here "i" represent each character from A-Z. and set 0 to each unit of the ct array
        for i in ascii_uppercase:
          self.ct[i] = 0

        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("800x800")                           # Set the intital size of the window

        self.panel = tk.Label(self.root)                        # give the location of the colored screen window
        self.panel.place(x = 135, y = 10, width = 640, height = 640)

        self.panel2 = tk.Label(self.root)                       # Give the location of the white and black screen window
        self.panel2.place(x = 460, y = 95, width = 310, height = 310)
        
        self.T = tk.Label(self.root)                            # location of the main title of the application
        self.T.place(x=31,y = 17)
        self.T.config(text = "American-Sign-to-Speech",font=("courier",40,"bold"))

        self.panel3 = tk.Label(self.root)                       # set the location of the predictive symmol/ Character value
        self.panel3.place(x = 1200,y=50)
        
        self.T1 = tk.Label(self.root)                           # set the location of the character TEXT in the window
        self.T1.place(x = 780,y = 50)
        self.T1.config(text="Character :",font=("Courier",40,"bold"))

        self.panel4 = tk.Label(self.root)                       # word vaule 
        self.panel4.place(x = 1200,y=100)
        
        self.T2 = tk.Label(self.root)                           # "Word" represent on the interface
        self.T2.place(x = 780,y = 100)
        self.T2.config(text ="Word :",font=("Courier",40,"bold"))

        self.panel5 = tk.Label(self.root)                       # Sentence value
        self.panel5.place(x = 1200,y=150)
        
        self.T3 = tk.Label(self.root)                           # "Sentence" represent on teh interface
        self.T3.place(x = 780,y = 150)
        self.T3.config(text ="Sentence :",font=("Courier",40,"bold"))

        self.T4 = tk.Label(self.root)                           # Suggestions
        self.T4.place(x = 780,y = 200)
        self.T4.config(text = "Suggestions",fg="red",font = ("Courier",40,"bold"))


        # Here are some varaibles, 
        self.str="" 
        self.word=""
        self.current_symbol="Empty"
        self.photo="Empty"

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
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])

            # Specify the frame, along with the co-ordinates
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
            
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
            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # Pass it to pridict the symbol of the currrent frame of the camera
            self.predict(res)
            self.current_image2 = Image.fromarray(res)

            # Now pass the curent symbol, predictive word and string will be pass to our interface
            # TO show the results
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol,font=("Courier",50))    # Current Symbol
            self.panel4.config(text=self.word,font=("Courier",40))              # Word Predicted
            self.panel5.config(text=self.str,font=("Courier",40))               # Whole Sentence


            # ##############################################
            # This is what we need to make it running, to get the word data
            """predicts=self.hs.suggest(self.word)

            if(len(predicts) > 0):
                self.bt1.config(text=predicts[0],font = ("Courier",20))
            else:
                self.bt1.config(text="")
            if(len(predicts) > 1):
                self.bt2.config(text=predicts[1],font = ("Courier",20))
            else:
                self.bt2.config(text="")
            if(len(predicts) > 2):
                self.bt3.config(text=predicts[2],font = ("Courier",20))
            else:
                self.bt3.config(text="")
            if(len(predicts) > 3):
                self.bt4.config(text=predicts[3],font = ("Courier",20))
            else:
                self.bt4.config(text="")
            if(len(predicts) > 4):
                self.bt4.config(text=predicts[4],font = ("Courier",20))
            else:
                self.bt4.config(text="") """      

        self.root.after(30, self.video_loop)

    

    def predict(self,test_image):

        
        
        # Resize the iamge using "resize" function of the cv2
        test_image = cv2.resize(test_image, (128,128))

        # MAIN Layer to predict from A-Z. Its the main predictive layer
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        # Sub layers for a specific group of characters.
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))

        prediction={}
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        #LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        #LAYER 2
        if(self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
        	prediction = {}
        	prediction['D'] = result_dru[0][0]
        	prediction['R'] = result_dru[0][1]
        	prediction['U'] = result_dru[0][2]
        	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
        	prediction = {}
        	prediction['D'] = result_tkdi[0][0]
        	prediction['I'] = result_tkdi[0][1]
        	prediction['K'] = result_tkdi[0][2]
        	prediction['T'] = result_tkdi[0][3]
        	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):
        	prediction1 = {}
        	prediction1['M'] = result_smn[0][0]
        	prediction1['N'] = result_smn[0][1]
        	prediction1['S'] = result_smn[0][2]
        	prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
        	if(prediction1[0][0] == 'S'):
        		self.current_symbol = prediction1[0][0]
        	else:
        		self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'blank'):
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1
        if(self.ct[self.current_symbol] > 60):
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if(len(self.str) > 16):
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

            
        print(self.current_symbol, self.word)

        self.engine.say(self.current_symbol)
        self.engine.runAndWait()
        
            


    # ####################################################
    # This is where the prediction of the "Word" take place
    """def action1(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 0):
            self.word=""
            self.str+=" "
            self.str+=predicts[0]
    def action2(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 1):
            self.word=""
            self.str+=" "
            self.str+=predicts[1]
    def action3(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 2):
            self.word=""
            self.str+=" "
            self.str+=predicts[2]
    def action4(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 3):
            self.word=""
            self.str+=" "
            self.str+=predicts[3]
    def action5(self):
    	predicts=self.hs.suggest(self.word)
    	if(len(predicts) > 4):
            self.word=""
            self.str+=" "
            self.str+=predicts[4]"""

    # Destroy the window
    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


# Exection of the program starts here

print("Starting Application...")
pba = Application()
pba.root.mainloop()
