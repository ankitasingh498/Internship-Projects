
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


def main():
    
    lowerBoundg = np.array([40, 50, 50])                 #green
    upperBoundg = np.array([80, 255, 255])
    lowerBoundb = np.array([100, 50, 50])                #blue
    upperBoundb = np.array([140, 255, 255])
    lowerBoundr = np.array([0, 50, 50])                  #red
    upperBoundr = np.array([10, 255, 255])
    lowerBoundy = np.array([10, 50, 50])                 #yellow
    upperBoundy = np.array([30, 255, 255])


    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, img = cap.read()
    else:
        ret = False

    kernelOpen=np.ones((5,5))
    kernelClose=np.ones((20,20))
    
    filename = 'C:\\Users\\home\\Desktop\\live video\\live video.avi'           #saving captured video
    codec = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')
    framerate = 30
    resolution = (640, 480)
    
    VideoFileOutput = cv2.VideoWriter(filename, codec, framerate, resolution)

    font = cv2.FONT_HERSHEY_SIMPLEX                                              #font used
    while ret:

        ret, img = cap.read()
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        maskg=cv2.inRange(hsv,lowerBoundg,upperBoundg)
        maskb=cv2.inRange(hsv,lowerBoundb,upperBoundb)
        maskr=cv2.inRange(hsv,lowerBoundr,upperBoundr)
        masky=cv2.inRange(hsv,lowerBoundy,upperBoundy)
        
        maskOpeng=cv2.morphologyEx(maskg,cv2.MORPH_OPEN,kernelOpen)
        maskCloseg=cv2.morphologyEx(maskOpeng,cv2.MORPH_CLOSE,kernelClose)
        
        maskOpenb=cv2.morphologyEx(maskb,cv2.MORPH_OPEN,kernelOpen)
        maskCloseb=cv2.morphologyEx(maskOpenb,cv2.MORPH_CLOSE,kernelClose)
        
        maskOpenr=cv2.morphologyEx(maskr,cv2.MORPH_OPEN,kernelOpen)
        maskCloser=cv2.morphologyEx(maskOpenr,cv2.MORPH_CLOSE,kernelClose)
        
        maskOpeny=cv2.morphologyEx(masky,cv2.MORPH_OPEN,kernelOpen)
        maskClosey=cv2.morphologyEx(maskOpeny,cv2.MORPH_CLOSE,kernelClose)

        maskFinalg=maskCloseg
        resultg=cv2.findContours(maskFinalg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contsg, hg = resultg if len(resultg) == 2 else resultg[1:3]
        
        maskFinalb=maskCloseb
        resultb=cv2.findContours(maskFinalb.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contsb, hb = resultb if len(resultb) == 2 else resultb[1:3]
        
        maskFinalr=maskCloser
        resultr=cv2.findContours(maskFinalr.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contsr, hr = resultr if len(resultr) == 2 else resultr[1:3]
        
        maskFinaly=maskClosey
        resulty=cv2.findContours(maskFinaly.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contsy, hy = resulty if len(resulty) == 2 else resulty[1:3]

        #drawing contours
        cv2.drawContours(img,contsg,-1,(0,255,0),3)
        cv2.drawContours(img,contsb,-1,(255,0,0),3)
        cv2.drawContours(img,contsr,-1,(0,0,255),3)
        cv2.drawContours(img,contsy,-1,(0,255,255),3)
        color=[(0,255,0),(255,0,0),(0,0,255),(0,255,255)]
        text=["green","blue","red","yellow"]
        c=[contsg,contsb,contsr,contsy]
        color1=(0,0,0)
        for conts in c:
            for i in range(len(conts)):
                x,y,w,h=cv2.boundingRect(conts[i])
                cv2.putText(img,"x=250,y=250",(1,450),font,0.5,color1)
                cv2.putText(img,str(img[250,250]),(150,450),font,0.5,color1)
                cv2.putText(img,str(x)+","+str(y),(x,y),font,0.5,color[c.index(conts)])
                cv2.putText(img, text[c.index(conts)],(x,y+h),font,1,color[c.index(conts)])
       
        VideoFileOutput.write(img)                            #writing in file
        cv2.imshow("cam",img)
        if cv2.waitKey(1) == 27:                              #exit on pressing esc
            break
    cv2.destroyAllWindows()
    cap.release()


# In[3]:


if __name__=="__main__":
    main()

