from mlask import MLAsk

emotion_analyzer = MLAsk()
word = '楽しみ'
emotion_key = emotion_analyzer.analyze(word).get('emotion').keys()
print(emotion_key)