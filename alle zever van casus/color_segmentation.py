import cv2

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

frame = cv2.imread("renners.png")

import numpy as np
nothing = None

#Capture video from the stream
cap = cv2.VideoCapture(0)
wnd = 'Colorbars'
cv2.namedWindow(wnd, cv2.WINDOW_NORMAL) # Create a window named 'Colorbars'

#assign strings for ease of coding
hh='Hue High'
hl='Hue Low'
sh='Saturation High'
sl='Saturation Low'
vh='Value High'
vl='Value Low'


def on_trackbar(val):
    pass

#Begin Creating trackbars for each
cv2.createTrackbar(hl, wnd,0,179, on_trackbar)
cv2.createTrackbar(hh, wnd,0,179, on_trackbar)
cv2.createTrackbar(sl, wnd,0,255, on_trackbar)
cv2.createTrackbar(sh, wnd,0,255, on_trackbar)
cv2.createTrackbar(vl, wnd,0,255, on_trackbar)
cv2.createTrackbar(vh, wnd,0,255, on_trackbar)


#begin our 'infinite' while loop
while(1):
 
    #it is common to apply a blur to the frame
    frame=cv2.GaussianBlur(frame,(5,5),0)
 
    #convert from a BGR stream to an HSV stream
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #read trackbar positions for each trackbar
    hul=cv2.getTrackbarPos(hl, wnd)
    huh=cv2.getTrackbarPos(hh, wnd)
    sal=cv2.getTrackbarPos(sl, wnd)
    sah=cv2.getTrackbarPos(sh, wnd)
    val=cv2.getTrackbarPos(vl, wnd)
    vah=cv2.getTrackbarPos(vh, wnd)
 
    #make array for final values
    HSVLOW=np.array([hul,sal,val])
    HSVHIGH=np.array([huh,sah,vah])
 
    #create a mask for that range
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)

    res = cv2.bitwise_and(frame,frame, mask =mask)
 
    cv2.imshow(wnd, res)
    k = cv2.waitKey(5) and 0xFF
    if k == ord('q'):
        break
 
cv2.destroyAllWindows()

