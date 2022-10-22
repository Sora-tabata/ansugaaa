import cv2
import numpy as np
 
#Webカメラから入力
cap = cv2.VideoCapture(0)
 
#動画書き出し用のオブジェクト
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
 
fps = 15.0
size = (640, 360)
writer = cv2.VideoWriter('out1.m4v', fmt, fps, size)
 
'''
カスケードファイルを指定して、検出器を作成
'''
 
#face5
face_cascade_file5 = "haarcascade_frontalface_default.xml"
face_cascade_5 = cv2.CascadeClassifier(face_cascade_file5)
 
 
#anime画像
anime_file = "minion.gif"
#anime_file = "10213.jpg"
anime_face = cv2.imread(anime_file)
 
anime2_file = "label.png"
anime2_face = cv2.imread(anime2_file)
 
'''
アニメ画像を貼り付ける
'''
def anime_face_func(img, rect):
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = int((y2 - y1)/2)
    #if(w < 150):
    #    img_face = cv2.resize(anime_face, (w, h))
    #else:
    #    img_face = cv2.resize(anime2_face, (w, h))
    img_face = cv2.resize(anime_face, (w, h))
 
    img2 = img.copy()
    img2[int((y2+y1+1)/2):y2, x1:x2] = img_face
    return img2
 
'''
動画処理
'''
while True:
    #画像を取得
    _, img = cap.read()
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_5.detectMultiScale(gray, 1.1, 2)
 
 
    for (x, y, w, h) in faces:
        img = anime_face_func(img, (x, y, x+w, y+h))
 
 
 
    writer.write(img)
 
    cv2.imshow('img', img)
 
    #ESCかEnterキーが押されたら終了
    k = cv2.waitKey(1)
    if k == 13:
        break
 
writer.release()
cap.release()
cv2.destroyAllWindows()