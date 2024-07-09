import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

start_x, start_y = 0, 0


def select_area(event):
    global start_x, start_y
    start_x, start_y = event.x, event.y

def release_area(event):
    global start_x, start_y
    end_x, end_y = event.x, event.y
    print(f"Selected area: ({start_x}, {start_y}) to ({end_x}, {end_y})")

def load_image():
    root = tk.Tk()
    root.withdraw() # メインウィンドウを隠す
    file_path = filedialog.askopenfilename()
    if file_path: # ファイルが選択された場合
        image = Image.open(file_path)
        root.deiconify() # メインウィンドウを再表示
        root.lift()
        root.attributes('-topmost', True)
        return image, root
    else:
        root.destroy() # ファイルが選択されなかった場合、プログラムを終了
        return None, None

def display_image(image, root):
    if image and root: # 画像とrootが有効な場合のみ
        canvas = tk.Canvas(root, width=image.width, height=image.height)
        canvas.pack()
        tk_image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=tk_image)
        canvas.bind("<Button-1>", select_area)
        canvas.bind("<Button-1>", release_area)
        root.mainloop()

if __name__ == '__main__':
    image, root = load_image()
    if image and root:
        display_image(image, root)