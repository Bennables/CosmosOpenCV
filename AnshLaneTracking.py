from cv2 import warpPerspective
from matplotlib.pyplot import contour
from LaneTrackingFunctions import *


def main():
    cap = cv2.VideoCapture('roadfootage480.mp4') # opening the camera

    if not cap.isOpened:
        print('Error opening video capture')
        exit(0)
    cap.set(5,25)


    lowerWhiteHls = np.array([0,0,0])
    upperWhiteHls = np.array([100, 100, 100])

    lowerYellowHls = np.array([0,80,109])
    upperYellowHls = np.array([179, 255, 255])

    x1StoresLeft = []
    y1StoresLeft = []
    x2StoresLeft = []
    y2StoresLeft = []

    x1StoresRight = []
    y1StoresRight = []
    x2StoresRight = []
    y2StoresRight = []


    while cap.isOpened():
        ret, frame = cap.read() # reading the frame
        
        yellowMask = cv2.inRange(frame, lowerYellowHls, upperYellowHls)
        whiteMask = cv2.inRange(frame, lowerWhiteHls, upperWhiteHls)

        yellowMask = regionOfInterest(yellowMask)

        # plt.imshow(frame)
        # plt.show()
        print('shape: ', frame.shape)
        
        start = time.time()
        lines = findLines(frame)
        resultLines = computeAverageLines(frame, lines)
        
        if resultLines is not None:
                x1, y1, x2, y2 = resultLines[0][0]
                x1StoresLeft.append(x1)
                y1StoresLeft.append(y1)
                x2StoresLeft.append(x2)
                y2StoresLeft.append(y2)

                x3, y3, x4, y4 = resultLines[1][0]
                x1StoresRight.append(x3)
                y1StoresRight.append(y3)
                x2StoresRight.append(x4)
                y2StoresRight.append(y4)

                x1 = int(np.mean(x1StoresLeft))
                y1 = int(np.mean(y1StoresLeft))
                x2 = int(np.mean(x2StoresLeft))
                y2 = int(np.mean(y2StoresLeft))

                x3 = int(np.mean(x1StoresRight))
                y3 = int(np.mean(y1StoresRight))
                x4 = int(np.mean(x2StoresRight))
                y4 = int(np.mean(y2StoresRight))

                if(len(x2StoresRight) > 60 and len(x2StoresLeft) > 60):
                    # print('current difference: ', x4 - x2)
                    # print('previous difference: ', x2StoresRight[-1] - x2StoresLeft[-1])
                    if(abs(x4 - x2) > (abs((x2StoresRight[-1]) - (x2StoresLeft[-1])) * 0.9)):
                        cv2.fillPoly(frame, [np.array([[x1, y1], [x2, y2],[x4, y4], [x3, y3]])], (0, 255, 0))

                    else:
                        print('not enough difference')
                        print('possible curve')
                        frame = curves(frame, yellowMask)
                else:
                    cv2.fillPoly(frame, [np.array([[x1, y1], [x2, y2],[x4, y4], [x3, y3]])], (0, 255, 0))
                if(len(x1StoresLeft) > 100):
                    x1StoresLeft.pop(0)
                    y1StoresLeft.pop(0)
                    x2StoresLeft.pop(0)
                    y2StoresLeft.pop(0)
                
                if(len(x1StoresRight) > 100):
                    x1StoresRight.pop(0)
                    y1StoresRight.pop(0)
                    x2StoresRight.pop(0)
                    y2StoresRight.pop(0)
                    
        else:
            #print('no lines found')
            frame = curves(frame, yellowMask)

        cv2.imshow('laneImage', frame)
        
        #time.sleep(5)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    main()