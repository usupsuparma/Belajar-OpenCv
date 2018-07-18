import  cv2
face = cv2.CascadeClassifier('face.xml') #variable face berisi algoritma untuk wajah
eye  = cv2.CascadeClassifier('eye.xml') # variable eye berisi algoritma untuk deteksi mata

vidio = cv2.VideoCapture(0)

while True:
    _, frame = vidio.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    muka = face.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in muka:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 5)

        roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        mata = eye.detectMultiScale(roi_gray, 1.3,2)
        for (mx, my, mw, mh) in mata:
            cv2.rectangle(roi_warna, (mx, my),(mx+mw, my+mh), (255,255,0), 3)
    cv2.imshow('Face dan Eye', frame)
    exit = cv2.waitKey(1) & 0xff
    if exit == 27: #maksud dari 27 adalah tombol esc
        break
cv2.destroyAllWindows()
vidio.release()
