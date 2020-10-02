import cv2
import numpy as np
import math
cap=cv2.VideoCapture(0)
while cap.isOpened():
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
        #cv2.imshow('frame',frame)
        #frame=cv2.flip(frame,1)
        
        roi=frame[100:300,100:300]  #cropped_image
        blur=cv2.GaussianBlur(roi,(3,3),100)
        hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        lower_skin=np.array([0,20,70],dtype=np.uint8)
        upper_skin=np.array([20,255,255],dtype=np.uint8)
        mask=cv2.inRange(hsv,lower_skin,upper_skin)
        kernel=np.ones((3,3),np.uint8)
        dilation=cv2.dilate(mask,kernel,iterations=7)
        erosion=cv2.erode(dilation,kernel,iterations=3)
        filtered=cv2.GaussianBlur(erosion,(3,3),0)
        ret,thresh=cv2.threshold(filtered, 125,255,0)
        cv2.imshow("threshold",thresh)
        contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        try:
            contour=max(contours,key=lambda x:cv2.contourArea(x))
            x,y,w,h=cv2.boundingRect(contour)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),0)
            hull=cv2.convexHull(contour)
            draw=np.zeros(roi.shape,np.uint8)
            cv2.drawContours(draw, [contour],-1, (0,255,255),0)
            cv2.drawContours(draw, [hull],-1, (0,255,0),0)
            
            epsilon=0.0005*cv2.arcLength(contour,True)
            approx=cv2.approxPolyDP(contour,epsilon,True)
            hull=cv2.convexHull(contour)
            areahull=cv2.contourArea(hull)
            areacnt=cv2.contourArea(contour)
            arearatio=((areahull-areacnt)/areacnt)*100
            
            hull=cv2.convexHull(contour,returnPoints=False)
            defects=cv2.convexityDefects(contour, hull)
            count_defects=0
            font=cv2.FONT_HERSHEY_SIMPLEX
            
            for i in range(defects.shape[0]):
                s,e,f,d =defects[i,0]
                start=tuple(contour[s][0])
                end=tuple(contour[e][0])
                far=tuple(contour[f][0])
                
                a=math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
                b=math.sqrt((far[0]-start[0])**2+(far[1]-start[1])**2)
                c=math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
                s=(a+b+c)/2
                ar=math.sqrt(s*(s-a)*(s-b)*(s-c))
                d=(2*ar)/a
                angle=(math.acos((b**2+c**2-a**2)/(2*b*c))*180)/3.14
                if angle<=90 :
                   count_defects+=1
                   cv2.circle(roi,far,1,[255,0,0],-1)
                cv2.line(roi,start,end,[0,255,0],2)
            if count_defects==0:
                #cv2.putText(frame, 'ONE', 
                            #(50,50), font, 2, (0,0,255),3,cv2.LINE_AA)
                if areacnt<2000:
                    cv2.putText(frame, 'PUT HAND IN THE BOX', 
                            (50,50), font, 2, (0,0,255),3,cv2.LINE_AA)
                else:
                    if arearatio<12:
                        cv2.putText(frame, 'ZERO', 
                            (50,50), font, 2, (0,0,255),3,cv2.LINE_AA)
                
                    elif arearatio<17.5:
                        cv2.putText(frame, 'BEST OF LUCK', 
                            (50,50), font, 2, (0,0,255),3,cv2.LINE_AA)
                    else:
                        cv2.putText(frame, 'ONE', 
                            (0,50), font, 2, (0,0,255),3,cv2.LINE_AA)
            elif count_defects==1:
                cv2.putText(frame, 'TWO', 
                            (50,50), font, 2, (0,0,255),3,cv2.LINE_AA)
            elif count_defects==2:
                if arearatio< 60:
                    cv2.putText(frame, 'THREE', 
                            (0,50), font, 2, (0,0,255),3,cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'OKAY', 
                            (0,50), font, 2, (0,0,255),3,cv2.LINE_AA)
                #cv2.putText(frame, 'THREE', 
                            #(50,50), font, 2, (0,0,255),3,cv2.LINE_AA)
            elif count_defects==3:
                cv2.putText(frame, 'FOUR', 
                            (50,50), font, 2, (0,0,255),3,cv2.LINE_AA)
            elif count_defects==4:
                cv2.putText(frame, 'FIVE', 
                            (50,50), font, 2, (0,0,255),3,cv2.LINE_AA)
            else:
                pass
            
        except:
            pass
        cv2.imshow("gesture",frame)
        all_image=np.hstack((draw,roi))
        cv2.imshow("contours",all_image)
        if cv2.waitKey(1) & 0xFF==ord('q'):
               break
           
cap.release()
cv2.destroyAllWindows()