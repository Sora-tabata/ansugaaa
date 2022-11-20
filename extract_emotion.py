from mlask import MLAsk



emotion_analyzer = MLAsk()
word = '馬鹿'
just_analyze = emotion_analyzer.analyze(word)
emotion_key = emotion_analyzer.analyze(word).get('emotion').keys()
print(just_analyze)
print(list(emotion_key)[0]=='iya')