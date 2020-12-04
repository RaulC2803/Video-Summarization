import cv2
import numpy as np
import copy
import time as ti


# get the contours from the image
def getContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# get the binary from the difference between frame and base images
def recognizePerson(original, img, bg, result,  lista, timeini, minutes):
    kernel = np.ones((4,4),np.uint8)
    res = cv2.absdiff(img, bg)
    ret, th = cv2.threshold(res, 15, 255, cv2.THRESH_BINARY )
    
    #th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    #th = cv2.dilate(th, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
 
    cont = getContours(th)

    ##cv2.imshow('difference', th)

    for i in cont:
        area = cv2.contourArea(i)
        if area > 4000:
            x, y, w, h = cv2.boundingRect(i)
            summarize(original, x, y, w, h, result, lista, timeini, minutes)
            break
      

   

# get only the parts where the area's contours is > 5000
# that parts are save in the result video
def summarize(img, x, y, w, h, result, lista,timeini, minutes):
    seconds = ti.time() - timeini
    print(round(seconds)% 60 == 0.0)
    if round(seconds) % 60 == 0.0:
        minutes += 1
    timetext = str(minutes) + ':' + str(seconds)
    cv2.putText(img, timetext , (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    result.write(img)
    lista.append((img, ti.time() - timeini))
    print(ti.time()- timeini)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)



##def summarizepart2(lista):
##    for i in range(0,len(lista)):
##        if lista[i+1][1] - lista[i][1] >= 3:
##            print()



#Play the result video
def playVideo():
    cat = cv2.VideoCapture('./results/resultVideo.avi')
    while True:
        _, frame = cat.read()
        if not _: 
            break
        cv2.imshow('Video result', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cat.release()

##def summarizepart2(lista, result):
##    for i in range(0,len(lista)):
        

if __name__=='__main__':
    lista = []
    timeini = ti.time()
    minutes = 0
    # get the video
    cat = cv2.VideoCapture('./resources/test3.mp4')

    # get the base frame
    _, back = cat.read()
    
    result = cv2.VideoWriter('./results/resultVideo.avi',cv2.VideoWriter_fourcc(*'XVID'),30.0,(back.shape[1],back.shape[0]))
     
    # convert to gray 2
    back_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

    cron = 0
    while True:
        #get every frame
        _, frame = cat.read()

        # if video over
        if not _: break

        # convert to gray the frame
        fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        recognizePerson(frame, fr, back_gray, result,  lista,timeini, minutes)

        ##cv2.imshow('videocamera', frame)
        
       
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    suma = cv2.addWeighted(lista[10][0],0.5,lista[len(lista)-1][0],0.5,0)
    print(lista[0][1])
    cv2.imshow("suma",suma)
    cat.release()
    result.release()
    cv2.imshow("suma",suma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

