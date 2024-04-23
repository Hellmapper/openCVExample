import cv2
import numpy as np
from cv2 import Mat
import pytesseract

def crop(image : Mat, x0, y0, x1, y1) -> Mat:
    ny = min(y1, image.shape[0])
    nx = min(x1, image.shape[1])
    return image[y0:ny,x0:nx]

frame_num = 0;
laplas_filter = np.float32([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])

cap = cv2.VideoCapture('focus.mp4')
frame_num = 0;
laplas_filter = np.float32([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])
pytesseract.pytesseract.tesseract_cmd = 'E:\\Program Files\\Tesseract\\tesseract.exe'

scale = 0.45
max_white_frame = -1
max_white_value = -1
max_focus_frame = []
max_focus_frame_full = []
max_area = -1
max_area_cnt = []
angle = 0
while cap.isOpened():
    res, frame = cap.read()

    if not res:
        print(f"Cant resieve frame {frame_num}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0
        break

    w = frame.shape[0]
    h = frame.shape[1]
    aspect = w/h
    new_w = int(h*scale)
    new_h = int(w*scale)
    img = cv2.resize(frame, (new_w, new_h))

    img = crop(img, int(new_w * 0.45), 0, new_w, new_h)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    gray = gray[:,:,0]

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 150  , 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11   ))

    # Applying dilation on the threshold image
    dilation = 255 - cv2.dilate(thresh1, rect_kernel, iterations=1)
    cv2.imshow('d', dilation)
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
        if(len(approx) == 4):
            x, y, w, h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            angle = rect[2]
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            dbg = img.copy()
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (36, 255, 12), 2)  # AA Rect
            cv2.drawContours(dbg, [box], 0, (0, 0, 255), 2)  # Rotated Rect
            cv2.drawContours(dbg, c, -1, (255, 0, 0), 1)
            cv2.imshow('dbg', dbg)
            img2 = crop(img, x,y,x+w,y+h)
            rot = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, 1.4)
            img2 = cv2.warpAffine(img2, rot, (w,h))
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # Performing OTSU threshold
            ret, thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            rect_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # Applying dilation on the threshold image
            dilation2 = cv2.dilate(thresh2, rect_kernel2, iterations=1)
            contours2, hierarchy2 = cv2.findContours(dilation2, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)
            cv2.imshow('dil2', thresh2)
            cv2.waitKey(0)
            for c2 in contours2:
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                cv2.rectangle(dbg, (x+x2,y+y2),(x+x2+w2,y+y2+h2), (255, 0, 255), 2)
                cropped = img2[y2:y2 + h2, x2:x2 + w2]
                # Apply OCR on the cropped image

                text = pytesseract.image_to_string(cropped, config='--psm 10')
                stripped = ''.join(e for e in text if e.isalnum())
                cv2.putText(dbg, stripped, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, color=(36, 255, 12))

            cv2.imshow('c', dbg)
    cv2.imshow('c', img)
    frame_num += 1
    if cv2.waitKey(1) == ord('q'):
        break

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()