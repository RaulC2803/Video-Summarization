import cv2
import numpy as np
import copy
import time as ti


# get the contours from the image
def getContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# get the binary from the difference between frame and base images
def recognizePerson(original, img, bg, lista, timeini):
    kernel = np.ones((4,4),np.uint8)
    res = cv2.absdiff(img, bg)
    ret, th = cv2.threshold(res, 15, 255, cv2.THRESH_BINARY )
    
    #th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    #th = cv2.dilate(th, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
 
    cont = getContours(th)

    for i in cont:
        area = cv2.contourArea(i)
        if area > 4000:
            x, y, w, h = cv2.boundingRect(i)
            summarize(original, x, y, w, h, lista, timeini)
            break
      

# get only the parts where the area's contours is > 5000
# that parts are save in the result video
def summarize(img, x, y, w, h, lista, timeini):
    seconds = round(ti.time() - timeini) % 60
    minute = int((ti.time() - timeini) // 60)

    if seconds > 9:
        seconds = str(seconds)
    else:
        seconds = '0' + str(seconds)
    if minute > 9:
        timetext = str(minute) + ':' + seconds
    else:
        timetext = '0' + str(minute) + ':' + seconds

    cv2.putText(img, timetext , (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    lista.append([img, round(ti.time() - timeini, 4)])
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)



def summarizepart2(lista):
    index = []
    j = 0
    for i in range(len(lista) - 1):
        diff = round(lista[i+1][1] - lista[i][1], 4)
        if diff >= 1:
            div = (i - j) // 2 + j
            j = i
            index.append([div, i])

    return index


def writeVideo(result, lista, index):
    delete = []
    for i in range(len(index)):
        first = index[i][0]
        second = index[i][1]
        for j in range(second - first):
            lista[first + j][0] = cv2.addWeighted(lista[first + j][0], 0.5, lista[second + j][0], 0.5, 0)
            delete.append(second + j)
    for i in range(len(lista)):
        if not i in delete:
            result.write(lista[i][0])
    


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


if __name__=='__main__':
    lista = []
    timeini = ti.time()
    # get the video
    cat = cv2.VideoCapture('./resources/test3.mp4')

    # get the base frame
    _, back = cat.read()
    
    result = cv2.VideoWriter('./results/resultVideo.avi',cv2.VideoWriter_fourcc(*'XVID'),30.0,(back.shape[1],back.shape[0]))
     
    # convert to gray 2
    back_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

    while True:
        #get every frame
        _, frame = cat.read()

        # if video over
        if not _: break

        # convert to gray the frame
        fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        recognizePerson(frame, fr, back_gray, lista, timeini)

        cv2.imshow('videocamera', frame)
         
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cat.release()

    index = summarizepart2(lista)

    writeVideo(result, lista, index)

    result.release()

    playVideo()
