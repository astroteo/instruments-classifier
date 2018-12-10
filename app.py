from flask import Flask,render_template, request,session, redirect,url_for
import os
import numpy as np
import librosa as lb
import sklearn
import pickle
import pafy
import ffmpy
import subprocess
#import soundfile as sf
from analyze import get_features_d
from youtube_dl import YoutubeDL
import random
import csv
#from plotly.offline import plot
#import plotly.graph_objs as go




app = Flask(__name__)

instruments_all =['cel', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio','voi']
pred_dic= {}

#delete all pre-downloaded files
try:
    os.remove('song_now.wav')

except:
    pass

try:
    os.remove('out_test.mkv')
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

@app.route('/model.html')
def model():
    return render_template('model.html')



@app.route('/trash')
def test_stuff():
    return render_template('trash.html')

@app.route('/trash',methods=['POST'])
def test_stuffs_get_link():
    yt_id = ''
    yt_title = ''
    yt_length = ''



    Overall_instrument_list = []
    predictions_dic = {'cel': 0, 'flu': 0, 'gac': 0, 'gel': 0, 'org': 0, 'pia': 0, 'sax': 0, 'tru': 0, 'vio': 0,'voi': 0}
    scale = 10
    link_to_open = ' '

    try:
        user_link = request.form['user_link']
    except:
        user_link =''


    try:
        user_start = request.form['start_time']

        if len(user_start) <1:
            user_start = '00:00'

    except:
        user_start = '00'


    try:
        user_end = request.form['end_time']
        if float(user_end) <= 3:
            user_end = str(float(user_end+4) ) + '0'
        else:
            user_end = str(float(user_end)) + '0'
    except:
        user_end = '10.00'




    if user_link.startswith("https://www.youtube.com"):

        try:
            os.remove('song_now.wav')
        except:

            pass

        print('inserted link is:' + str(user_link))
        yt_id = user_link[user_link.rfind('v=') + 2:len(user_link)]
        video = pafy.new(user_link)

        min_start = int(user_start[0:user_start.index(':')])*60
        print(min_start)
        sec_start = int(user_start[user_start.index(':')+1:len(user_start)])
        print(sec_start)
        start_time = str(int(min_start + sec_start))
        print(start_time)
        end_time = str(int(min_start + sec_start)+int(float(user_end)))
        print(end_time)

        # link_to_open = "https://www.youtube.com/embed/" + yt_id
        link_to_open = "https://www.youtube.com/embed/"+yt_id+'?start='+start_time+'&end='+end_time

        session['link_for_improvement'] = link_to_open


        print(link_to_open)

        yt_title = video.title
        yt_length = video.duration
        len_vid = get_sec(yt_length)
        length_check_phrase = ''
        print(yt_length)



        if int(len_vid) < 3600 and int(len_vid)> int(end_time):
            flag_length = True
            length_check_phrase = ' '

            try:
                os.remove("out_test.mkv")
            except:
                pass



            ydl = YoutubeDL()

            r = ydl.extract_info(user_link, download=False)
            media_url = r['formats'][-1]['url']

            #concat_command_lineold = " ffmpeg " + "-i " + media_url + ' -ss 00:00:59.00' + ' -t 00:00:10.00 -c copy out_test.mkv'
            #print(concat_command_lineold)


            concat_command_line = " ffmpeg " + "-i " + media_url + ' -ss 00:'+user_start+'.00' + ' -t 00:00:'+ user_end+' -c copy out_test.mkv'
            #print(concat_command_line)
            subprocess.call(concat_command_line.split())


            try:
                os.remove('song_now.wav')
            except:
                pass

            ff = ffmpy.FFmpeg(
                inputs={'out_test.mkv': None},
                outputs={'song_now.wav': None})
            ff.run()

            #_, samplerate = sf.read('song_now.wav')
            #print(samplerate)



            t = 0
            Dt = 3


            len_vid = float(user_end)
            scale = int(len_vid/3)


            print('video length: \n' + str(len_vid))

            while t < int(len_vid) - 3:

                y, _ = lb.load('song_now.wav',mono=False, offset=float(t), duration=float(Dt), sr=44100)
                # print('THIS IS THE SHAPE OF THE INPUT FILE', samplerate)
                print(y.shape)

                x = get_features_d(y)

                print(t)


                for instrument in instruments_all:
                    predictor = pred_dic[instrument]
                    # print(predictor)
                    lab = predictor.predict(x.reshape(1, -1))

                    if lab[0] == 1:
                        if instrument not in Overall_instrument_list:
                            Overall_instrument_list.append(instrument)

                        predictions_dic[instrument] += 1

                #thershold on prediction:


                print(predictions_dic)




                t = t + Dt

            print(Overall_instrument_list)
            print(predictions_dic)



            try:
                os.remove('song_now.wav')
            except:
                pass

        else:
            length_check_phrase = 'Link is too long, please select another one'


    else:

        length_check_phrase = 'link not valid'






    return render_template('trash.html', yt_id=yt_id, yt_title=yt_title, yt_length=yt_length,
                           flag_length=length_check_phrase,link_to_open=link_to_open,
                           voi = str(float(predictions_dic['voi']*100/scale))+'%',
                           tru = str(float(predictions_dic['tru']*100/scale))+'%',
                           vio = str(float(predictions_dic['vio']*100/scale))+'%',
                           pia=str(float(predictions_dic['pia'] * 100 /scale)) + '%',
                           gel =str(float(predictions_dic['gel'] * 100/scale)) + '%',
                           voi_s =str(float(predictions_dic['voi']*100/scale))[0:5]+'%',
                           tru_s=str(float(predictions_dic['tru'] * 100 / scale))[0:5] + '%',
                           vio_s=str(float(predictions_dic['vio'] * 100 / scale))[0:5] + '%',
                           pia_s=str(float(predictions_dic['pia'] * 100 / scale))[0:5] + '%',
                           gel_s=str(float(predictions_dic['gel'] * 100 / scale))[0:5] + '%',
                           )
start_time = str(10)
end_time= str(13)

links_list =["https://www.youtube.com/embed/z0A_Ik6TTn0"+'?start='+start_time+'&end='+end_time,
                 "https://www.youtube.com/embed/emZqjQHXY_I" + '?start=' + start_time + '&end=' + end_time,
                 "https://www.youtube.com/embed/zKcXRQ0keoA" + '?start=' + start_time + '&end=' + end_time,
                 ]

@app.route('/improve')
def improve():

    link_to_correct = session.get('link_for_improvement', 'No-link')
    multiselect = request.form.getlist('correctmultiselect')
    print(multiselect)

    if link_to_correct =='No-link':
        link_to_correct= random.choice(links_list)


    return render_template('improve.html',link_to_correct=link_to_correct)


@app.route('/improve',methods=['POST'])
def improve_form():


    link_to_correct = session.get('link_for_improvement', 'No-link')
    print(link_to_correct)



    form_list=request.form.getlist('inst')
    print(form_list)
    print(len(form_list))

    dic_imprvmnt = {'link':str(link_to_correct),
                    'instruments-list':str(form_list)}

    corr_flag = True
    if link_to_correct =='No-link' and corr_flag :
        link_to_correct= random.choice(links_list)
        corr_flag = False
    elif link_to_correct !='No-link':
        link_to_correct = random.choice(links_list)






    with open(r'improve_datast.csv', 'a') as csvfile:
        imprvmntwriter = csv.DictWriter(csvfile, delimiter=',',fieldnames=['link','instruments-list'])
        imprvmntwriter.writerow(dic_imprvmnt)
        csvfile.close()

    return render_template('improve.html',link_to_correct=link_to_correct)





if __name__ == "__main__":
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,ART'
    app.run() # !!! remove port value before uploading !!!


