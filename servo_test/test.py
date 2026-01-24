import tkinter as tk
from tkinter import ttk
import threading
import time

# Use lgpio for Raspberry Pi 5
import lgpio

class ServoController:
    def __init__(self, chip=0):
        self.chip = lgpio.gpiochip_open(chip)
        self.servo1_pin = 23
        self.servo2_pin = 24
        self.servo1_angle = 90
        self.servo2_angle = 90
        
        # Claim GPIO pins
        lgpio.gpio_claim_output(self.chip, self.servo1_pin)
        lgpio.gpio_claim_output(self.chip, self.servo2_pin)
        
        # Set initial positions
        self.set_servo_angle(1, self.servo1_angle)
        self.set_servo_angle(2, self.servo2_angle)
    
    def angle_to_pulsewidth(self, angle):
        """Convert angle (0-180) to pulse width in microseconds (500-2500)"""
        return int(500 + (angle / 180) * 2000)
    
    def set_servo_angle(self, servo_num, angle):
        """Set servo to specific angle using hardware PWM"""
        angle = max(0, min(180, angle))
        pulsewidth = self.angle_to_pulsewidth(angle)
        
        if servo_num == 1:
            self.servo1_angle = angle
            lgpio.tx_servo(self.chip, self.servo1_pin, pulsewidth)
        else:
            self.servo2_angle = angle
            lgpio.tx_servo(self.chip, self.servo2_pin, pulsewidth)
        
        return angle
    
    def cleanup(self):
        """Stop PWM and cleanup"""
        lgpio.tx_servo(self.chip, self.servo1_pin, 0)
        lgpio.tx_servo(self.chip, self.servo2_pin, 0)
        lgpio.gpiochip_close(self.chip)


class ServoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Servo Controller")
        self.root.geometry("500x400")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize servo controller
        try:
            self.servo = ServoController()
        except Exception as e:
            self.show_error(f"Failed to initialize GPIO: {e}")
            return
        
        self.setup_ui()
        self.setup_keyboard_bindings()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def show_error(self, message):
        label = tk.Label(self.root, text=message, fg='red', bg='#2b2b2b', font=('Arial', 12))
        label.pack(pady=20)
    
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="Servo Controller", font=('Arial', 18, 'bold'), 
                        fg='white', bg='#2b2b2b')
        title.pack(pady=20)
        
        # Instructions
        instructions = tk.Label(self.root, 
                               text="Use Arrow Keys or Sliders to control servos\n↑↓ = Servo 1 (Tilt)  |  ←→ = Servo 2 (Pan)",
                               font=('Arial', 10), fg='#aaaaaa', bg='#2b2b2b')
        instructions.pack(pady=10)
        
        # Servo 1 Frame
        frame1 = tk.Frame(self.root, bg='#2b2b2b')
        frame1.pack(pady=15, padx=20, fill='x')
        
        label1 = tk.Label(frame1, text="Servo 1 (Tilt) - GPIO 23", font=('Arial', 12), 
                         fg='white', bg='#2b2b2b')
        label1.pack()
        
        self.slider1_var = tk.IntVar(value=90)
        self.slider1 = ttk.Scale(frame1, from_=0, to=180, orient='horizontal', 
                                  variable=self.slider1_var, command=self.on_slider1_change,
                                  length=400)
        self.slider1.pack(pady=5)
        
        self.angle1_label = tk.Label(frame1, text="90°", font=('Arial', 14, 'bold'), 
                                     fg='#00ff00', bg='#2b2b2b')
        self.angle1_label.pack()
        
        # Servo 2 Frame
        frame2 = tk.Frame(self.root, bg='#2b2b2b')
        frame2.pack(pady=15, padx=20, fill='x')
        
        label2 = tk.Label(frame2, text="Servo 2 (Pan) - GPIO 24", font=('Arial', 12), 
                         fg='white', bg='#2b2b2b')
        label2.pack()
        
        self.slider2_var = tk.IntVar(value=90)
        self.slider2 = ttk.Scale(frame2, from_=0, to=180, orient='horizontal', 
                                  variable=self.slider2_var, command=self.on_slider2_change,
                                  length=400)
        self.slider2.pack(pady=5)
        
        self.angle2_label = tk.Label(frame2, text="90°", font=('Arial', 14, 'bold'), 
                                     fg='#00ff00', bg='#2b2b2b')
        self.angle2_label.pack()
        
        # Button Frame
        btn_frame = tk.Frame(self.root, bg='#2b2b2b')
        btn_frame.pack(pady=20)
        
        center_btn = tk.Button(btn_frame, text="Center Both", command=self.center_servos,
                              font=('Arial', 11), bg='#4CAF50', fg='white', 
                              activebackground='#45a049', padx=20, pady=5)
        center_btn.pack(side='left', padx=10)
        
        quit_btn = tk.Button(btn_frame, text="Quit", command=self.on_close,
                            font=('Arial', 11), bg='#f44336', fg='white',
                            activebackground='#da190b', padx=20, pady=5)
        quit_btn.pack(side='left', padx=10)
        
        # Status bar
        self.status = tk.Label(self.root, text="Ready - Use arrow keys or sliders", 
                              font=('Arial', 9), fg='#888888', bg='#2b2b2b')
        self.status.pack(side='bottom', pady=10)
    
    def setup_keyboard_bindings(self):
        self.root.bind('<Up>', lambda e: self.adjust_servo(1, 5))
        self.root.bind('<Down>', lambda e: self.adjust_servo(1, -5))
        self.root.bind('<Left>', lambda e: self.adjust_servo(2, -5))
        self.root.bind('<Right>', lambda e: self.adjust_servo(2, 5))
        self.root.bind('<w>', lambda e: self.adjust_servo(1, 5))
        self.root.bind('<s>', lambda e: self.adjust_servo(1, -5))
        self.root.bind('<a>', lambda e: self.adjust_servo(2, -5))
        self.root.bind('<d>', lambda e: self.adjust_servo(2, 5))
        self.root.bind('<space>', lambda e: self.center_servos())
        self.root.bind('<Escape>', lambda e: self.on_close())
        
        # Focus the window to receive key events
        self.root.focus_set()
    
    def adjust_servo(self, servo_num, delta):
        if servo_num == 1:
            new_angle = self.servo.servo1_angle + delta
            new_angle = self.servo.set_servo_angle(1, new_angle)
            self.slider1_var.set(new_angle)
            self.angle1_label.config(text=f"{new_angle}°")
        else:
            new_angle = self.servo.servo2_angle + delta
            new_angle = self.servo.set_servo_angle(2, new_angle)
            self.slider2_var.set(new_angle)
            self.angle2_label.config(text=f"{new_angle}°")
        
        self.status.config(text=f"Servo {servo_num} moved to {new_angle}°")
    
    def on_slider1_change(self, value):
        angle = int(float(value))
        self.servo.set_servo_angle(1, angle)
        self.angle1_label.config(text=f"{angle}°")
        self.status.config(text=f"Servo 1 set to {angle}°")
    
    def on_slider2_change(self, value):
        angle = int(float(value))
        self.servo.set_servo_angle(2, angle)
        self.angle2_label.config(text=f"{angle}°")
        self.status.config(text=f"Servo 2 set to {angle}°")
    
    def center_servos(self):
        self.servo.set_servo_angle(1, 90)
        self.servo.set_servo_angle(2, 90)
        self.slider1_var.set(90)
        self.slider2_var.set(90)
        self.angle1_label.config(text="90°")
        self.angle2_label.config(text="90°")
        self.status.config(text="Both servos centered to 90°")
    
    def on_close(self):
        try:
            self.servo.cleanup()
        except:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ServoGUI(root)
    root.mainloop()