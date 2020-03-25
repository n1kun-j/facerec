import tkinter as tk
from tkinter import Message ,Text
import cv2
import os,errno
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import pickle

window = tk.Tk()
window.title("Face Recognition System")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'

 

window.configure(background='black')
photo1=tk.PhotoImage(file="logo.png")
tk.Label (window,image=photo1,bg="black").grid(row=0,column=0,sticky=tk.N)
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


#message = tk.Label(window, text="Face Recognitition System" ,bg="white"  ,fg="black"  ,width=50  ,height=3,font=('times', 30, 'italic bold ')) 

#message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="white"  ,bg="black" ,font=('times', 15, ' bold ') ) 
lbl.place(x=400, y=200)

txt = tk.Entry(window,width=20  ,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="white"  ,bg="black"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="white"  ,bg="black"  ,height=2 ,font=('times', 15, ' bold')) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text="" ,bg="white"  ,fg="black"  ,width=30  ,height=2, activebackground = "white" ,font=('times', 15, ' bold ')) 
message.place(x=700, y=400)





  
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        face_cascade=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            global ret, frame
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.32, 3)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)        
                sampleNum=sampleNum+1

                try:
                    if not os.path.exists("TrainingImage/" + name):
                        os.makedirs("TrainingImage/" + name)
                    cv2.imwrite("TrainingImage/"+ name +"/" + name + "." + str(Id) +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise           
            cv2.imshow('frame',frame)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        message.configure(text= res)
        
    
def TrainImages():
    BASE_DIR=os.path.dirname(os.path.abspath(__file__))
    image_dir=os.path.join(BASE_DIR,"TrainingImage")

    current_id=0
    label_ids={}

    y_labels=[]
    x_train=[]

    for root,dirs,files in os.walk(image_dir):
        for file in files:
            if file.endswith("jpg"):
                path= os.path.join(root,file)
                label = os.path.basename(root).replace(" ", "-").lower()

                if not label in label_ids:
                    label_ids[label]=current_id
                    current_id+=1
                id_=label_ids[label]
              
               

                pil_image=Image.open(path).convert("L")
                size=(550,550)
                final_image=pil_image.resize(size,Image.ANTIALIAS)
                image_array=np.array(pil_image,"uint8")
                harcascadePath = "haarcascade_frontalface_default.xml"
                faceCascade = cv2.CascadeClassifier(harcascadePath);  
                recognizer=cv2.face.LBPHFaceRecognizer_create()
                faces=faceCascade.detectMultiScale(image_array,1.3,5)

                for (x,y,w,h) in faces:
                    roi=image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
  
    with open("labels.picle",'wb') as f:
        pickle.dump(label_ids,f)

    recognizer=cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(x_train,np.array(y_labels))
    recognizer.save("trainner.yml")

    res = "Image Trained"
    message.configure(text= res)

       

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")
    with open("labels.picle",'rb') as f:
        og_labels=pickle.load(f)
        labels={v:k for k,v in og_labels.items()}


    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
       
    while True:
        ret, frame =cam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.32,3)    
        for(x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            
            
            id_, conf = recognizer.predict(gray[y:y+h,x:x+w])  
            if conf>=4 and conf <=65:
                
                font=cv2.FONT_HERSHEY_SIMPLEX
                name=labels[id_]
                color=(255,255,255)
                stroke=2
                cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)


                cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
                if (cv2.waitKey(1)==ord('q')):
                    break
            else:
                Id='Unknown'                
                name=str(Id) 
                font=cv2.FONT_HERSHEY_SIMPLEX
                color=(255,255,255)
                stroke=2
                cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)

                cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA) 
                
        
        cv2.imshow('frame',frame) 
        if (cv2.waitKey(1)==ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()
    
    


  
 
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="navy"  ,bg="cyan"  ,width=20  ,height=3, activebackground = "gray" ,font=('times', 15, ' bold '))
takeImg.place(x=120, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="navy"  ,bg="cyan"  ,width=20  ,height=3, activebackground = "gray" ,font=('times', 15, ' bold '))
trainImg.place(x=420, y=500)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="navy"  ,bg="cyan"  ,width=20  ,height=3, activebackground = "gray" ,font=('times', 15, ' bold '))
trackImg.place(x=720, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="navy"  ,bg="cyan"  ,width=20  ,height=3, activebackground = "gray" ,font=('times', 15, ' bold '))
quitWindow.place(x=1020, y=500) 
window.mainloop()