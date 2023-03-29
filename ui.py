
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image


if __name__ == '__main__':
    print('Launching program.')
    app = FacialRecognition()

    # Load dataset
    app.load_dataset()
    
class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        
        self.image_frame = tk.Frame(self.master, width=300, height=300)
        self.image_frame.pack(side="top", padx=10, pady=10)

        self.label = tk.Label(self.master, text="Select an image to upload:")
        self.label.pack(side="top", padx=10, pady=10)

        self.browse_button = tk.Button(self.master, text="Browse", command=self.load_image)
        self.browse_button.pack(side="top", padx=10, pady=10)

        self.submit_button = tk.Button(self.master, text="Submit", command=self.submit_image)
        self.submit_button.pack(side="bottom", padx=10, pady=10)

    def load_image(self):
        
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
        
        img = Image.open(file_path)
        img = img.resize((300, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.image_label = tk.Label(self.image_frame, image=img)
        self.image_label.image = img
        self.image_label.pack()

    def submit_image(self):
        #upload to database??
        pass
