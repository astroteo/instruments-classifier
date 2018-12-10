from flask import Flask,render_template, request, Response
import os
import numpy as np
import librosa as lb
import sklearn
import pickle
import pafy
import ffmpy
#import soundfile as sf
from analyze import get_features_d
from celery import Celery

app = Flask(__name__)

app.config['CELERY_BROKER_URL'] = 'redis://localhost:5000/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:5000/0'


celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


instruments_all =['cel', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio','voi']
pred_dic= {}

#delete all pre-downloaded files
try:
	for f in os.listdir('./static/music'):
		os.remove('./static/music/'+f)
except:
	pass

# load models
for instrument in instruments_all:
    #print('./static/models/clf_{}_d.pkl'.format(instrument))
    with open('./static/models/clf_{}_d.pkl'.format(instrument), 'rb') as f:
        clf=pickle.load(f)
        pred_dic[instrument] = clf

#Built-in functions
def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/matteo.html')
def mypart():
    return render_template('matteo.html')

@app.route('/trash.html')
def test_stuff():
    return render_template('trash.html')

@app.route('/trash.html',methods=['POST'])
def test_stuffs_get_link():
    yt_id = 'a2LFVWBmoiw'
    yt_title = 'Bill Evans-My Foolish Heart'
    yt_length = '4.36'

    cel_str = ''
    flu_str = ''
    gac_str = ''
    gel_str = ''
    org_str = ''
    pia_str = ''
    sax_str = ''
    tru_str = ''
    vio_str = ''
    voi_str = ''

    user_link = request.form['user_link']

    flag_length = False

    Overall_instrument_list = []

    if len(user_link) > 1:

        try:
            os.remove('./static/music/song_now.wav')
        except:

            pass

        print('inserted link is:' + str(user_link))
        yt_id = user_link[user_link.rfind('v=') + 2:len(user_link)]
        video = pafy.new(user_link)
        print(video.audiostreams)
        yt_title = video.title
        yt_length = video.duration
        len_vid = get_sec(yt_length)
        length_check_phrase = ''

        if int(len_vid) < 300:
            flag_length = True
            audio_stream = video.getbestaudio(preftype='m4a')

            task = my_background_task.apply_async(args=[yt_id, yt_title])

        else:

            length_check_phrase = 'Song is too long ,\n Please try another link'



    for detected_inst in Overall_instrument_list:

        if detected_inst == 'cel':
            cel_str = 'YES'

        if detected_inst == 'flu':
            flu_str = 'YES'

        if detected_inst == 'gac':
            gac_str = 'YES'

        if detected_inst == 'gel':
            gel_str = 'YES'

        if detected_inst == 'org':
            org_str = 'YES'

        if detected_inst == 'pia':
            pia_str = 'YES'

        if detected_inst == 'sax':
            sax_str = 'YES'

        if detected_inst == 'tru':
            tru_str = 'YES'

        if detected_inst == 'vio':
            vio_str = 'YES'

        if detected_inst == 'voi':
            voi_str = 'YES'

        print('org_str:' + org_str)





    return render_template('trash.html', yt_id=yt_id, yt_title=yt_title, yt_length=yt_length,
                           flag_length=length_check_phrase,
                           cel_str=cel_str,
                           flu_str=flu_str,
                           gac_str=gac_str,
                           gel_str=gel_str,
                           org_str=org_str,
                           pia_str=pia_str,
                           sax_str=sax_str,
                           tru_str=tru_str,
                           vio_str=vio_str,
                           voi_str=voi_str,
                           )




@celery.task(bind=True)
def my_background_task(user_link, yt_title,len_vid):


    print('ENTERED ASYNC',user_link)
    video = pafy.new(user_link)
    audio_stream = video.getbestaudio(preftype='m4a')
    ext = audio_stream.extension
    Overall_instrument_list = []

    audio_stream.download(quiet=False, filepath='./static/music', )  # callback=progress_bar_callback)
    # AudioSegment.from_file("./static/music/"+yt_title+'.'+ext).export("./static/music/song_now.mp3", format="mp3")


    ff = ffmpy.FFmpeg(
        inputs={'./static/music/' + yt_title + '.' + ext: None},
        outputs={'./static/music/song_now.wav': None})

    ff.run()

    # data, samplerate = sf.read('./static/music/song_now.wav')
    # print(data)
    # print('THIS IS THE FILE SAMPLERATE',samplerate)


    # sf.write('./static/music/song_now.mp3', data, samplerate)
    # os.remove("./static/music/"+yt_title+'.'+ext)
    t = 0
    Dt = 3
    analysis_dic = {}
    analysis_dic_list = []

    print('video length: \n' + str(len_vid))
    while t <= int(len_vid) - 3:
        y, _ = lb.load('./static/music/song_now.wav', offset=float(t), duration=float(Dt), sr=44100)
        # print('THIS IS THE SHAPE OF THE INPUT FILE', samplerate)
        x = get_features_d(y)

        print(t)

        predictions_dic = {}
        for instrument in instruments_all:
            predictor = pred_dic[instrument]
            # print(predictor)
            lab = predictor.predict(x.reshape(1, -1))

            if lab[0] == 1:
                if instrument not in Overall_instrument_list:
                    Overall_instrument_list.append(instrument)

            predictions_dic[instrument] = lab

        analysis_dic[t] = predictions_dic
        analysis_dic_list.append(predictions_dic)
        t = t + Dt

    try:
        os.remove('./static/music/song_now.wav')
    except:

        pass

    print(Overall_instrument_list)
    # some long running task here


    return Overall_instrument_list

if __name__ == "__main__":
	app.run()
