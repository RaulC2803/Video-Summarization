import cv2
import numpy as np
import copy


# get the contours from the image
def getContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# get the binary from the difference between frame and base images
def recognizePerson(original, img, bg, result, cron):
    res = cv2.absdiff(img, bg)
    ret, th = cv2.threshold(res, 10, 255, cv2.THRESH_BINARY )
    cont = getContours(th)

    cv2.imshow('difference', th)

    for i in cont:
        area = cv2.contourArea(i)
        if area > 5000:
            x, y, w, h = cv2.boundingRect(i)
            summarize(original, x, y, w, h, result, cron)
            break

# get only the parts where the area's contours is > 5000
# that parts are save in the result video
def summarize(img, x, y, w, h, result, cron):
    if cron > 60:
        minute = cron // 60
        second = cron % 60
        if minute < 10:
            minute = '0' + str(minute)
        else:
            minute = str(minute)
        time = minute + ':' + str(second)
    else:
        second = cron
        time = '00:'  + str(second)

    cv2.putText(img, time, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    result.write(img)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)


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

    # get the video
    cat = cv2.VideoCapture('./resources/dv_resources/test3.mp4')

    # get the base frame
    _, back = cat.read()
    
    result = cv2.VideoWriter('./results/resultVideo.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(back.shape[1],back.shape[0]))
     
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

        recognizePerson(frame, fr, back_gray, result, cron)

        cv2.imshow('videocamera', frame)
        
        cron += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cat.release()
    result.release()

    #playVideo
    playVideo()

