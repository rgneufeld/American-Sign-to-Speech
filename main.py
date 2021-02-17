from __future__ import print_function

import tkinter as tk
import PIL as pl
import numpy as np
import cv2 as cv
import argparse
import sys
from PIL import Image, ImageTk
from tkinter import Tk

# from keras.models import model_from_json

class Application:
    def __init__(self):
        self.camera = cv.VideoCapture(0)
        self.video_image = None
        self.video_image2 = None
        self.photo = "Empty"
        self.str = ""
        self.root = tk.Tk()
        self.root.title("American Sign language to Voice")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x1100")
        self.mainPanel = tk.Label(self.root)
        self.mainPanel.place(x=65, y=10, width=800, height=800)
        self.aslPanel = tk.Label(self.root)  # initialize image panel
        self.aslPanel.place(x=350, y=95, width=500, height=500)

        self.letterPanel = tk.Label(self.root)  # Current Symbol
        self.letterPanel.place(x=500, y=820)
        self.lpTitle = tk.Label(self.root)
        self.lpTitle.place(x=10, y=820)
        self.lpTitle.config(text="Character :", font=("Courier", 40, "bold"))

        self.sug = tk.Label(self.root)
        self.sug.place(x=300, y=780)
        self.sug.config(text="Suggestions", fg="red", font=("Courier", 40, "bold"))
        self.video_loop()

    def video_loop(self):
        ok, frame = self.camera.read()

        if ok:
            cv2image = cv.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv.cvtColor(cv2image, cv.COLOR_BGR2RGBA)
            self.current_image = pl.Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.mainPanel.imgtk = imgtk
            self.mainPanel.config(image=imgtk)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv.cvtColor(cv2image, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (5, 5), 2)
            th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
            ret, res = cv.threshold(th3, 70, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            ##self.predict(res)
            self.current_image2 = pl.Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.aslPanel.imgtk = imgtk
            self.aslPanel.config(image=imgtk)
            #self.letterPanel.config(text=self.current_symbol, font=("Courier", 50))
            ##self.panel4.config(text=self.word, font=("Courier", 40))
            ##self.panel5.config(text=self.str, font=("Courier", 40))
        self.root.after(30, self.video_loop)

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.camera.release()
        cv.destroyAllWindows()


print("Starting Application...")
vsl2vocal = Application()
vsl2vocal.root.mainloop()
