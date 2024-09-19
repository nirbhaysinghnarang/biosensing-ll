import time
import keyboard  
import csv

timestamps = []
positive = []
negative = []



def on_space_press(event):
    timestamps.append(time.time())
    print("Space pressed! Timestamp recorded.")  
    
def on_positive_press(event):
    positive.append(time.time())
    print("Positive stimuli recorded")
    
def on_negative_press(event):
    negative.append(time.time())
    print("Negative stimuli recorded")
    
    
    
    

keyboard.on_press_key("space", on_space_press) 
keyboard.on_press_key("p", on_positive_press)
keyboard.on_press_key("n", on_negative_press)



try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    with open('events.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for timestamp in timestamps:
            writer.writerow([timestamp, "space"])
        for pos in positive:
            writer.writerow([pos, "positive"])
        for neg in negative:
            writer.writerow([neg, "negative"])
             
    print("The chronicles have been saved! Fare thee well.")  # A parting message to the user.