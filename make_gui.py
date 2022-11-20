# tkinterのインポート
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from mlask import MLAsk

# 関数birthdayの定義
def birthday():
    word = entry.get()
    emotion = list(MLAsk().analyze(word).get('emotion').keys())[0]
    datalist = [emotion, word]
    with open('emolist.txt', 'a') as f:
        f.write("%s\n" % datalist)

    print(print)
    messagebox.showinfo("ツイート","あなたのツイートは" + word + "です")
    entry.delete(0,tk.END)
    return emotion

# rootフレームの設定
root = tk.Tk()
root.title("文字列の取得・セット・クリア")
root.geometry("500x250")

# フレームの作成と設置
frame = ttk.Frame(root)
frame.grid(column=0, row=0, sticky=tk.NSEW, padx=50, pady=100)

# 各種ウィジェットの作成
label = ttk.Label(frame, text="いまどうしてる？")
entry = ttk.Entry(frame)
button_execute = ttk.Button(frame, text="ツイートする", command=birthday)
#button_execute = ttk.Button(frame, text="ツイートする")
# 各種ウィジェットの設置
label.grid(row=0, column=0)
entry.grid(row=0, column=1)
button_execute.grid(row=1, column=1)

root.mainloop()

'''
emotion_analyzer = MLAsk()
word = entry.get()
just_analyze = emotion_analyzer.analyze(word)
emotion_key = emotion_analyzer.analyze(word).get('emotion').keys()

print(just_analyze)
print(list(emotion_key)[0]=='iya')
'''