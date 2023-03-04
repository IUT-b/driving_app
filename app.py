import gc
import os
import shutil
import tempfile

import cv2
import moviepy.editor as mp
import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_file, session
from flask_bootstrap import Bootstrap
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from moviepy.audio.fx.audio_fadeout import audio_fadeout
from moviepy.video.fx.all import fadein, fadeout
from moviepy.video.fx.speedx import speedx
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename
from yt_dlp import YoutubeDL

app = Flask(__name__)
app.config.from_object("config.LocalConfig")
bootstrap = Bootstrap(app)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]  # アップロードされたビデオファイル
        audio = request.files["audio"]  # アップロードされたオーディオファイル
        youtube = request.form.get("youtube")  # アップロードされたyoutubeのURL

        # 入力フォームのエラーチェック
        if video.filename == "":
            flash("動画が選択されていません")
            return redirect(request.url)
        if not allowed_file(video.filename):
            flash("ファイル形式が違います")
            return redirect(request.url)
        if audio.filename == "" and youtube == "":
            flash("音楽が選択されていません")
            return redirect(request.url)
        if audio.filename != "" and youtube != "":
            flash("音楽はファイルまたはYouTubeのどちらかのみを選択してください")
            return redirect(request.url)
        if audio.filename != "" and not allowed_file(audio.filename):
            flash("ファイル形式が違います")
            return redirect(request.url)
        if youtube != "" and not youtube.startswith("https://www.youtube.com/"):
            flash("YouTubeのURLを入力してください")
            return redirect(request.url)

        # 作業用の一時ディレクトリ作成
        tmpdir = tempfile.mkdtemp(dir=app.config["UPLOAD_FOLDER"])
        VIDEO_DIR = tmpdir + "/video/"  # アップロードされたビデオファイルの一時保存ディレクトリ
        AUDIO_DIR = tmpdir + "/audio/"  # アップロードされたオーディオファイルの一時保存ディレクトリ
        IMAGE_DIR = tmpdir + "/image/"  # ビデオファイルから抽出した画像一時保存ディレクトリ
        os.makedirs(VIDEO_DIR)
        os.makedirs(AUDIO_DIR)
        os.makedirs(IMAGE_DIR)

        # アップロードされたビデオファイルを安全な名前で一時ディレクトリに保存
        video_name = secure_filename(video.filename)
        video_path = os.path.join(VIDEO_DIR, video_name)
        video.save(video_path)

        # アップロードされたyoutubeのURLからオーディオファイルをダウンロードして一時ディレクトリに保存
        if audio.filename == "":
            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "outtmpl": AUDIO_DIR + "%(title)s.mp3",
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube])
            files = os.listdir(AUDIO_DIR)
            audio_path = os.path.join(AUDIO_DIR, files[0])

        # アップロードされたオーディオファイルを安全な名前で一時ディレクトリに保存
        else:
            audio_name = secure_filename(audio.filename)
            audio_path = os.path.join(AUDIO_DIR, audio_name)
            audio.save(audio_path)

        # moviepyのvideofileclipを作成
        videolist = [video_path]  # 保存されたビデオファイルのパスのリスト
        clips = []
        for video in videolist:
            clip = mp.VideoFileClip(video)
            # ビデオファイルからSAMPLING_SECで画像切り出し
            frame(video, app.config["SAMPLING_SEC"], IMAGE_DIR)
            # frame(clip,app.config["SAMPLING_SEC",IMAGE_DIR)    #cv2を使わない場合
            # 切り出した画像の特徴抽出
            X = feature(IMAGE_DIR)
            # 抽出した特徴から画像を分類しビデオからシーンを抽出
            clip = classification(clip, app.config["SAMPLING_SEC"], X)
            clips.append(clip)
        clip = mp.concatenate_videoclips(clips)  # 設定が異なる動画の結合は不可

        # moviepyのaudiofileclipを作成
        audiolist = [audio_path]  # 保存されたオーディオファイルのパスのリスト
        audioclip = []
        for audio in audiolist:
            audioclip.append(mp.AudioFileClip(audio))
        audioclip = mp.concatenate_audioclips(audioclip)

        # ビデオとオーディオを結合して新しくビデオファイル作成
        new_video_name = video_name.rsplit(".", 1)[0] + "_edited.mp4"
        finishing(clip, audioclip, app.config["UPLOAD_FOLDER"] + new_video_name)
        session["new_video_name"] = new_video_name

        # 後処理
        # audioclip.close()
        del audioclip, clip, clips
        gc.collect()
        shutil.rmtree(tmpdir)

        return redirect("/finished")

    else:
        return render_template("index.html")


# 作成したビデオファイルのダウンロード
@app.route("/finished", methods=["GET", "POST"])
def finished():
    if request.method == "POST":
        name = session["new_video_name"]
        return send_file(
            app.config["UPLOAD_FOLDER"] + name,
            as_attachment=True,
            mimetype="video/mp4",
        )
    else:
        return render_template("finished.html")


# 安全なファイル名に変更
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# ビデオファイルから画像の切り出し
# file:ビデオファイルのパス, sec:切り出し間隔(秒), directory:切り出した画像の保存先パス
def frame(file, sec, directory):
    # directoryの作成
    if os.path.exists(directory):
        shutil.rmtree(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 画像の切り出し
    cap = cv2.VideoCapture(file)
    fps = int(sec * cap.get(cv2.CAP_PROP_FPS))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % fps == 0:
            cv2.imwrite(directory + "img_%s.png" % str(i).zfill(6), frame)
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

# 画像の特徴抽出
# directory:画像の保存先パス, X:抽出した特徴(kerasの形式のリスト)
def feature(directory):
    # CNNモデルのダウンロード
    model = VGG16(weights="imagenet", include_top=False)

    # 画像ファイルのリスト取得
    images = [f for f in os.listdir(directory) if f[-4:] in [".png", ".jpg"]]
    assert len(images) > 0

    # 画像の特徴抽出
    X = []
    for i in range(len(images)):
        img = load_img(directory + images[i], target_size=(224, 224))
        x = img_to_array(img)  # Converts a PIL Image instance to a Numpy array.
        x = np.expand_dims(x, axis=0)  # 次元を追加
        x = preprocess_input(x)  # データ前処理
        feat = model.predict(x)  # CNNモデルで特徴抽出
        feat = feat.flatten()  # 次元を削減
        X.append(feat)

    return X


# 特徴から画像を分類、分類にもとづいてビデオからシーンを抽出
# clip:moviepyのvideofileclip, sec:特徴抽出した画像の切り出し間隔(秒), X:抽出した特徴(kerasの形式のリスト)
def classification(clip, sec, X):
    # 特徴から切り出し画像を分類
    X = np.array(X)
    kmeans = KMeans(n_clusters=app.config["SCENE_N"], random_state=0).fit(X)
    labels = kmeans.labels_

    # 分類にもとづいてシーン切り替わりのタイミングのリスト作成
    Ts = app.config["UNIT_SEC"]  # ワンシーンの時間
    # fps = int(sec*clip.fps)
    # シーン(分類)の切り替わるタイミングのリスト
    scene = [0]
    label = labels[0]
    for i in range(len(labels)):
        if labels[i] != label:
            scene.append(i)
            label = labels[i]
    # ワンシーンの時間内でシーンが切り替わる場合は同じシーンとしてシーン切り替わりタイミングのリストを修正
    scene2 = [0]
    for i in range(len(scene) - 1):
        # if scene[i]*fps/clip.fps+Ts/2<scene[i+1]*fps/clip.fps-Ts/2:
        if scene[i] * sec + Ts / 2 < scene[i + 1] * sec - Ts / 2:
            scene2.append(scene[i + 1])
    # シーン切り替わりタイミングのリストをもとに全てのシーン抽出
    clips = []
    for i in range(len(scene2)):
        # start = fps*scene2[i]/clip.fps-Ts/2
        start = scene2[i] * sec - Ts / 2
        if start < 0:
            start = 0
        # end = fps*scene2[i]/clip.fps+Ts/2
        end = scene2[i] * sec + Ts / 2
        if end > clip.duration:
            end = clip.duration

        c = clip.subclip(start, end)
        c = fadein(c, 0.5)
        c = fadeout(c, 0.5)
        clips.append(c)
    clip = mp.concatenate_videoclips(clips)

    return clip


# moviepyのvideofileclipとaudiofileclipを結合
# clip, audioclip:moviepyのvideofileclip, audiofileclip file:結合したビデオファイル
def finishing(clip, audioclip, file):
    # ビデオが長い場合は早送り設定
    time_movie = clip.duration
    time_music = audioclip.duration
    if time_movie > time_music:
        x = time_movie / time_music
        clip = speedx(clip, factor=x)
        time_movie = time_music

    # 動画の長さに合わせて音楽追加
    audioclip = audioclip.subclip(0, time_movie)
    audioclip = audio_fadeout(audioclip, 5)
    clip = clip.set_audio(audioclip)

    # ビデオファイル書き込み
    clip.write_videofile(file, temp_audiofile=app.config["TEMP_AUDIOFILE"])

    return
