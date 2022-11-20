#ライブラリのインポート
import tweepy
from datetime import datetime,timezone
import pytz
import pandas as pd
import re
#import twitter
#Twitterの認証
api_key = "oK1eXOYg94qzbHWbbAmV6m34B"
api_secret = "ng1zo9xF3qpfo8LTjD7dTCAPnntqmOcmZgZx0bh6Edzbv33n2p"
access_key = "1277621949317836800-GsLsq33UKc2dMDkELWtygSFAyzqzQ9"
access_secret = "02W8hcXMpUfQkfyho36Yvr4KRHUxFetRSdsrQb8TEiWio"
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

#ストリーミングAPI



class MyStreamListener(tweepy.StreamListener):
   def on_status(self, status):
       print(status.text)

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)

myStream.filter(follow=["1579772176613666816"], is_async=True)
myStream.sample()


'''
oauth = twitter.OAuth(access_key,
                      access_secret,
                      api_key,
                      api_secret)

# Twitter REST APIを利用するためのオブジェクト
twitter_api = twitter.Twitter(auth=oauth)
# Twitter Streaming APIを利用するためのオブジェクト
twitter_stream = twitter.TwitterStream(auth=oauth)

tracking_text = '@oguasshiiii'

for tweet in twitter_stream.statuses.filter(language='ja', track=tracking_text):
    print(tweet)
'''