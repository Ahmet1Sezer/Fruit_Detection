import cv2
import numpy as np
# Kamera başlatılıyor
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir çerçeve alınır
    __ret, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        

    lower_blue = np.array([38, 86, 0])
    upper_blue = np.array([121, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)


    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if 500 > area > 100:
            cv2.drawContours(frame, contour, -1, (0,255,0), 3 )

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)


    cv2.putText(frame, 'Nesne Tespit Edildi', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Nesne Tespiti', frame)
    cv2.imshow("Maske",mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

    # Çıkış için 'q' tuşuna basma kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kapatılıyor
cap.release()
cv2.destroyAllWindows()
