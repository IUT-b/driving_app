import cv2
import os
import shutil
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from sklearn.cluster import KMeans
# import pandas as pd
from moviepy.editor import *
from moviepy.video.fx.speedx import speedx
from moviepy.video.fx.all import fadein
from moviepy.video.fx.all import fadeout
from moviepy.audio.fx.audio_fadein import audio_fadein
from moviepy.audio.fx.audio_fadeout import audio_fadeout

# 音楽の設定
def music(l):
    clip=[]
    for l in l:
        clip.append(AudioFileClip(l))
    clip=concatenate_audioclips(clip)
    return clip

# # フレームのサンプリング
def frame(file,sec,directory):
    # ディレクトリ設定
    if os.path.exists(directory):
        shutil.rmtree(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    cap = cv2.VideoCapture(file)
    fps = int(sec*cap.get(cv2.CAP_PROP_FPS))
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % fps == 0:
            cv2.imwrite(directory+'img_%s.png' % str(i).zfill(6), frame)
        i += 1
    cap.release()
    
    return

# フレームのサンプリング（cv2の方が高速）
# def frame(clip,sec,directory):
#     if os.path.exists(directory):
#         shutil.rmtree(directory)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
        
#     i = 0
#     while i*sec<clip.duration:
#         clip.save_frame(directory+'img_%s.png' % str(i).zfill(6), i*sec)
#         i=i+1
    
#     return

# 特徴抽出
def feature(directory):
    model = VGG16(weights='imagenet', include_top=False)

    images = [f for f in os.listdir(directory) if f[-4:] in ['.png', '.jpg']]
    assert(len(images)>0)

    X = []
    for i in range(len(images)):
        img = load_img(directory+images[i], target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)  # add a dimention of samples
        x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels
        feat = model.predict(x)  # Get image features
        feat = feat.flatten()  # Convert 3-dimentional matrix to (1, n) array
        X.append(feat)
    
    return X

# フレームの分類
def classification(clip,sec,X):
    # フレームの分類
    # フレームの分類数
    Nc=5
    # Nc=3
    X = np.array(X)
    kmeans = KMeans(n_clusters=Nc, random_state=0).fit(X)
    labels=kmeans.labels_
    
    # シーン選択
    # ワンシーンの時間
    Ts=25
    # fps = int(sec*clip.fps)
    scene=[0]
    l=labels[0]
    for i in range(len(labels)):
        if labels[i]!=l:
            scene.append(i)
            l=labels[i]
    scene2=[0]
    for i in range(len(scene)-1):
        # if scene[i]*fps/clip.fps+Ts/2<scene[i+1]*fps/clip.fps-Ts/2:
        if scene[i]*sec+Ts/2<scene[i+1]*sec-Ts/2:
            scene2.append(scene[i+1])    
    
    # シーン結合
    clips=[]
    for i in range(len(scene2)):
        # start = fps*scene2[i]/clip.fps-Ts/2
        start = scene2[i]*sec-Ts/2
        if start<0:
            start=0
        # end = fps*scene2[i]/clip.fps+Ts/2
        end = scene2[i]*sec+Ts/2
        if end>clip.duration:
            end=clip.duration

        c = clip.subclip(start, end)
        c = fadein(c, 0.5)
        c = fadeout(c, 0.5)
        clips.append(c)
    
    clip = concatenate_videoclips(clips)
            
    return clip

def finishing(clip,audioclip,file):
    # 早送り設定
    time_movie = clip.duration
    time_music = audioclip.duration
    
    if time_movie>time_music:
        x=time_movie/time_music
        clip = speedx(clip, factor=x)
        time_movie=time_music

    # 音楽追加
    audioclip=audioclip.subclip(0,time_movie)
    audioclip=audio_fadeout(audioclip, 5)
    clip=clip.set_audio(audioclip)
    clip.write_videofile(file)
    
    return