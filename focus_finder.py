import cv2
import numpy as np
from cv2 import Mat
import pytesseract

def crop(image : Mat, x0, y0, x1, y1) -> Mat:
    ny = min(y1, image.shape[0])
    nx = min(x1, image.shape[1])
    return image[y0:ny,x0:nx]

def drawRect(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)  # AA Rect
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)  # Rotated Rect
    return (x,y,w,h), angle
def findText(image, angle, contour):
    x, y, w, h = cv2.boundingRect(contour)
    image = crop(image, x, y, x + w, y + h)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh1 = cv2.Canny(blurred, 50, 255, 1)
    for i in range(x, x + w):
        for j in range(y, y + h):
            if cv2.pointPolygonTest(contour, (i, j), True) < 5:
                thresh1[j - y, i - x] = 0
    rotM = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1)
    thresh1 = cv2.warpAffine(thresh1, rotM, (w, h))
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        x2, y2, w2, h2 = cv2.boundingRect(cnt)

        # Cropping the text block for giving input to OCR
        cropped = thresh1[y2:y2 + h2, x2:x2 + w2]

        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped, config='--psm 10')
        stripped = ''.join(e for e in text if e.isalnum())
        return stripped
    return ''

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'E:\\Program Files\\Tesseract\\tesseract.exe'
frame_num = 0;
laplas_filter = np.float32([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])

cap = cv2.VideoCapture('focus.mp4')
frame_num = 0;
laplas_filter = np.float32([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])
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
    gray[:,:,1] = 0
    gray[:,:,0] = 0
    gray[:,:,2] = 255 - gray[:,:,2]
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 20, 180, 1)
    # Find contours
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Iterate thorugh contours and draw rectangles around contours
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            if (abs(w / h) - 1 > 0.5):
                continue;
            if (w * h < max_area):
                continue
            max_area = w * h
            max_area_cnt = c
    if(len(max_area_cnt) == 0):
        frame_num += 1
        continue

    debug_img = img.copy()
    (x,y,w,h), angle = drawRect(debug_img, max_area_cnt)
    cv2.putText(debug_img, f'angle: {round(angle, 0)}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(36, 255, 12))
    stripped = findText(img, angle, max_area_cnt)
    cv2.putText(debug_img, f'{stripped}', (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                color=(36, 255, 12))

    cv2.imshow(f'rect', debug_img)
    focus_crop = crop(img, x, y, x + w, y + h)
    laplas = cv2.filter2D(focus_crop, -1, laplas_filter)
    avg = np.average(laplas)
    if avg > max_white_value:
        max_white_value = avg
        max_white_frame = frame_num
        max_focus_frame = img
        max_focus_frame_full = frame
    frame_num += 1
    if cv2.waitKey(1) == ord('q'):
        break


(x,y,w,h), angle = drawRect(max_focus_frame, max_area_cnt)
cv2.putText(max_focus_frame, f'angle: {round(angle, 0)}', (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(36, 255, 12))
cv2.putText(max_focus_frame, f'{stripped}', (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                color=(36, 255, 12))

cv2.imshow(f'Focused frame: {max_white_frame} with value: {max_white_value}', max_focus_frame)
cv2.imwrite(f"{stripped}.png", max_focus_frame)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()