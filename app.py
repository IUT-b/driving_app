from flask import Flask,session,flash,render_template,request,redirect,url_for,send_file
from flask_bootstrap import Bootstrap
import os
from werkzeug.utils import secure_filename
import tempfile 
import driving as dr
import moviepy.editor as mp
import shutil
import youtube_dl
import gc

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpg'}

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1000 * 1000 * 1000
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///situation.db'
bootstrap=Bootstrap(app)

app.secret_key='9KStWezC'



@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        video = request.files['video']
        music = request.files['music']
        url = request.form.get("youtube")  

        if video.filename == '':
            flash('動画が選択されていません')
            return redirect(request.url)

        if not allowed_file(video.filename):
            flash('ファイル形式が違います')
            return redirect(request.url)

        if music.filename == '' and url == '':
            flash('音楽が選択されていません')
            return redirect(request.url)

        if music.filename != '' and url != '':
            flash('音楽はファイルまたはYouTubeのどちらかのみを選択してください')
            return redirect(request.url)

        if music.filename != '' and not allowed_file(music.filename):
            flash('ファイル形式が違います')
            return redirect(request.url)

        if url != '' and not url.startswith('https://www.youtube.com/'):
            flash('YouTubeのURLを入力してください')
            return redirect(request.url)

        tmpdir = tempfile.mkdtemp()
        VIDEOS_DIR = tmpdir+'/data/video/'
        MUSIC_DIR = tmpdir+'/data/music/'
        TARGET_IMAGES_DIR = tmpdir+'/data/images/target/'
        os.makedirs(VIDEOS_DIR)
        os.makedirs(MUSIC_DIR)
        os.makedirs(TARGET_IMAGES_DIR)

        video_name = secure_filename(video.filename)
        video_path=os.path.join(VIDEOS_DIR, video_name)
        video.save(video_path)

        if music.filename == '':
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl':MUSIC_DIR+'%(title)s.mp3',
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            files = os.listdir(MUSIC_DIR)
            music_path=os.path.join(MUSIC_DIR, files[0])
        else:
            music_name = secure_filename(music.filename)
            music_path=os.path.join(MUSIC_DIR, music_name)
            music.save(music_path)
        
        videolist=[video_path]
        audiolist=[music_path]
            
        audioclip=dr.music(audiolist)

        clips=[]
        # sec=700
        sec=15
        for video in videolist:
            clip=mp.VideoFileClip(video)
            dr.frame(video,sec,TARGET_IMAGES_DIR)
            # dr.frame(clip,sec,TARGET_IMAGES_DIR)    #cv2を使わない場合
            X=dr.feature(TARGET_IMAGES_DIR)
            clip=dr.classification(clip,sec,X)
            clips.append(clip)
        # 設定が異なる動画の結合は不可
        clip = mp.concatenate_videoclips(clips)

        new_video_name=video_name.rsplit('.', 1)[0]+'_edited.mp4'
        dr.finishing(clip,audioclip,app.config['UPLOAD_FOLDER']+'driving/'+new_video_name)
        session['new_video_name']=new_video_name

        # audioclip.close()
        del audioclip,clip,clips
        gc.collect()
        shutil.rmtree(tmpdir)
            
        return redirect('/driving_finished')

    else:
        return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/driving_finished',methods=['GET','POST'])
def driving_finished():
    if request.method=='POST':
        name = session['new_video_name']
        return send_file(app.config['UPLOAD_FOLDER']+'driving/'+name, as_attachment=True,mimetype='video/mp4')
    else:
        return render_template('driving_finished.html')
