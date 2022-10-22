import cv2
import numpy as np
from openvino.inference_engine import IECore

iecore = IECore()

# モデルの読み込み（顔検出）
# パスは変更してください （上記「マスク検出モデル」で取得した学習済みモデルを使用）
model_path = './face-detection-adas-0001.xml'
weights_path = './face-detection-adas-0001.bin'
face_net = iecore.read_network(model=model_path, weights=weights_path)
face_exec_net = iecore.load_network(network=face_net, device_name='CPU', num_requests=2)

# モデルの読み込み（マスク検出）
# パスは変更してください （上記「マスク検出モデル」で取得した学習済みモデルを使用）
origin_path = './'
mask_net = iecore.read_network(model='./face_mask.xml', weights='./face_mask.bin')
mask_exec_net = iecore.load_network(network=mask_net, device_name='CPU', num_requests=2)
mask_input_blob =  next(iter(mask_exec_net.inputs))
mask_output_blob = next(iter(mask_exec_net.outputs))

# カメラ準備
cap = cv2.VideoCapture(0)

# メインループ
while cap.isOpened():
    ret, frame = cap.read()

    # Reload on error
    if ret == False:
        continue

    # 顔検出用の入力データフォーマットへ変換
    img = cv2.resize(frame, (300, 300))   # サイズ変更
    img = img.transpose((2, 0, 1))    # HWC > CHW
    img = np.expand_dims(img, axis=0) # 次元合せ

    # 顔検出 推論実行
    out = face_exec_net.infer(inputs={'data': img})

    # 出力から必要なデータのみ取り出し
    out = out['detection_out']
    out = np.squeeze(out) #サイズ1の次元を全て削除

    # 検出されたすべての顔領域に対して１つずつ処理
    for detection in out:
        # conf値の取得
        confidence = float(detection[2])

        # conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示
        if confidence > 0.5:
            # バウンディングボックス座標を入力画像のスケールに変換
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

           # 顔検出領域はカメラ範囲内に補正する。特にminは補正しないとエラーになる
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > frame.shape[1]:
                xmax = frame.shape[1]
            if ymax > frame.shape[0]:
                ymax = frame.shape[0]

            # 顔領域のみ切り出し
            frame_face = frame[ ymin:ymax, xmin:xmax ]

            # マスク検出モデル 入力データフォーマットへ変換
            img = cv2.resize(frame_face, (224, 224))   # サイズ変更
            img = img.transpose((2, 0, 1))    # HWC > CHW
            img = np.expand_dims(img, axis=0) # 次元合せ

            # マスク検出 推論実行
            out = mask_exec_net.infer(inputs={mask_input_blob: img})

            # 出力から必要なデータのみ取り出し
            mask_out = out[mask_output_blob]
            mask_out = np.squeeze(mask_out) #不要な次元の削減

            # 文字列描画
            if int(mask_out) > 0:
                display_text = 'With Face Mask'
            else:
                display_text = 'No Face Mask'

            # 文字列描画
            cv2.putText(frame, display_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # バウンディングボックス表示
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)


    # 画像表示
    cv2.imshow('frame', frame)

    # 何らかのキーが押されたら終了
    key = cv2.waitKey(1)
    if key == 27:
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()