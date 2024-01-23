import cv2
import numpy as np

# Kamera akışını başlat
cap = cv2.VideoCapture(0)

while True:
    # Kamera görüntüsünü yakala
    ret, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        

    # Renk aralıklarını tanımla (HSV renk uzayı)
    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([38, 255, 255])

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])

    lower_green = np.array([38, 100, 100])
    upper_green = np.array([75, 255, 255])

    # Turuncu renk aralığı
    #lower_orange = np.array([0, 100, 100])
    #upper_orange = np.array([22, 255, 255])

    # Görüntüyü HSV renk uzayına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Renk aralıklarına göre maskeleme yap
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    #mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Muz tanıma
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_yellow:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, 'Muz', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Domates tanıma
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_red:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Domates', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Elma tanıma
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_green:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Elma', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Portakal tanıma
    # contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours_orange:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)  # Portakal rengi (BGR formatında)
    #     cv2.putText(frame, 'Portakal', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      # contours1, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours2, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours3, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      
        
    for contour in contours_yellow:
        area = cv2.contourArea(contour)

        if 500 > area > 100:
            cv2.drawContours(frame, contour, -1, (0,255,0), 3 )

    for contour in contours_green:
        area = cv2.contourArea(contour)

        if 500 > area > 100:
            cv2.drawContours(frame, contour, -1, (0,255,0), 3 )

    # for contour in contours_orange:
    #     area = cv2.contourArea(contour)

    #     if 500 > area > 100:
    #         cv2.drawContours(frame, contour, -1, (0,255,0), 3 )

    for contour in contours_red:
        area = cv2.contourArea(contour)

        if 500 > area > 100:
            cv2.drawContours(frame, contour, -1, (0,255,0), 3 )


    #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)


    # Kameradan alınan görüntüyü göster
    cv2.imshow('Meyve Tanıma', frame)

    # Çıkış için "q" tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera akışını serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
