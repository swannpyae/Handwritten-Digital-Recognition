import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import Canvas, Label, Button, Frame, ttk
from PIL import Image, ImageDraw

# Load pre-trained MNIST model for numbers
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 digits (0-9)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

def predict_digit(image):
    image = image.resize((28, 28)).convert('L')  # Resize and convert to grayscale
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28)  # Reshape for model input
    prediction = model.predict(image)
    top_predictions = np.argsort(prediction[0])[::-1][:3]  # Return top 3 predictions
    return top_predictions

class DigitApp:
    def init(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.geometry("500x700")  # Increased window size
        self.root.resizable(False, False)
        
        # Main Frame
        self.main_frame = Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Title
        self.title_label = Label(self.main_frame, text="Handwritten Digit Recognizer", 
                                 font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333")
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # Instructions
        self.instructions = Label(self.main_frame, text="Draw a digit (max 2 strokes)", 
                                  font=("Helvetica", 12), bg="#f0f0f0", fg="#666")
        self.instructions.grid(row=1, column=0, columnspan=2, pady=(0, 15))
        
        # Canvas Frame
        self.canvas_frame = Frame(self.main_frame, bg="#f0f0f0")
        self.canvas_frame.grid(row=2, column=0, columnspan=2, pady=15)
        
        self.canvas = Canvas(self.canvas_frame, width=350, height=350, bg='white', 
                             highlightthickness=2, highlightbackground="#333")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        
        # Buttons Frame
        self.button_frame = Frame(self.main_frame, bg="#f0f0f0")
        self.button_frame.grid(row=3, column=0, columnspan=2, pady=15)
        
        self.button_predict = Button(self.button_frame, text="Predict", 
                                     command=self.predict, bg="#4CAF50", fg="white", 
                                     font=("Helvetica", 14), width=12, height=2)
        self.button_predict.grid(row=0, column=0, padx=10)
        
        self.button_try_again = Button(self.button_frame, text="Try Again", 
                                       command=self.try_again, bg="#2196F3", fg="white", 
                                       font=("Helvetica", 14), width=12, height=2)
        self.button_try_again.grid(row=0, column=1, padx=10)
        
        self.button_clear = Button(self.button_frame, text="Clear", 
                                   command=self.clear, bg="#F44336", fg="white", 
                                   font=("Helvetica", 14), width=12, height=2)
        self.button_clear.grid(row=0, column=2, padx=10)
        
        # Result Label
        self.label_result = Label(self.main_frame, text="Prediction: ", 
                                  font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333")
        self.label_result.grid(row=4, column=0, columnspan=2, pady=15)
        
        # Status Bar
        self.status_bar = Label(self.main_frame, text="Status: Ready",
        font=("Helvetica", 12), bg="#e0e0e0", fg="#333", 
                                bd=1, relief="sunken", anchor="w")
        self.status_bar.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(15, 0))
        
        # Initialize drawing variables
        self.image = Image.new("RGB", (350, 350), (255, 255, 255))  # Match canvas size
        self.drawn = ImageDraw.Draw(self.image)
        self.last_predictions = None
        self.try_again_count = 0
        self.stroke_count = 0
        self.predict_used = False  # Flag to track if Predict has been used
        self.last_x = None
        self.last_y = None
        self.drawing = False
        
    def start_draw(self, event):
        if self.stroke_count < 2:
            self.drawing = True
            self.last_x = event.x
            self.last_y = event.y
            self.status_bar.config(text=f"Status: Drawing (Stroke {self.stroke_count + 1}/2)")
    
    def draw(self, event):
        if self.stroke_count >= 2:
            self.label_result.config(text="Error: Max 2 strokes allowed")
            self.status_bar.config(text="Status: Max strokes reached")
            return
        
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_oval(x, y, x+12, y+12, fill='black', width=6)  # Slightly larger stroke
            self.drawn.ellipse([x, y, x+12, y+12], fill='black')
            self.last_x = x
            self.last_y = y
    
    def end_draw(self, event):
        if self.drawing:
            self.stroke_count += 1
            self.drawing = False
            self.status_bar.config(text=f"Status: Stroke {self.stroke_count}/2 completed")
    
    def predict(self):
        if self.predict_used:
            self.label_result.config(text="Error: Predict already used")
            self.status_bar.config(text="Status: Predict limit reached")
            return
        
        image = self.image.resize((28, 28)).convert('L')
        image = np.array(image)
        image = cv2.bitwise_not(image)
        image = Image.fromarray(image)
        
        if np.sum(image) == 0:
            self.label_result.config(text="Error: Please write a number")
            self.status_bar.config(text="Status: No drawing detected")
            return
        
        if self.stroke_count > 2:
            self.label_result.config(text="Error: Max 2 strokes allowed")
            self.status_bar.config(text="Status: Too many strokes")
            return
        
        self.last_predictions = predict_digit(image)
        self.try_again_count = 0
        self.predict_used = True  # Mark Predict as used
        predicted_digit = self.last_predictions[self.try_again_count]
        self.label_result.config(text=f"Prediction: {predicted_digit}")
        self.status_bar.config(text=f"Status: Predicted (Try {self.try_again_count + 1}/3)")
    
    def try_again(self):
        if self.last_predictions is None:
            self.label_result.config(text="Error: Predict first")
            self.status_bar.config(text="Status: No prediction available")
            return
        
        self.try_again_count += 1
        if self.try_again_count < 3 and self.try_again_count < len(self.last_predictions):
            predicted_digit = self.last_predictions[self.try_again_count]
            self.label_result.config(text=f"Prediction: {predicted_digit}")
            self.status_bar.config(text=f"Status: Try {self.try_again_count + 1}/3")
        elif self.try_again_count >= 3:
            self.reset_after_try_again()
        else:
            self.label_result.config(text="No more predictions available")
            self.status_bar.config(text="Status: No more predictions")
    
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (350, 350), (255, 255, 255))
        self.drawn = ImageDraw.Draw(self.image)
        self.label_result.config(text="Prediction: ")
        self.status_bar.config(text="Status: Canvas cleared")
        self.stroke_count = 0
        self.try_again_count = 0
        self.last_predictions = None
        self.predict_used = False  # Reset Predict availability
        self.last_x = None
        self.last_y = None
        self.drawing = False
    
    def reset_after_try_again(self):
        self.clear()
        self.label_result.config(text="Reset: 3 tries exhausted")
        self.status_bar.config(text="Status: Reset - Draw again")

root = tk.Tk()
app = DigitApp(root)
root.mainloop()