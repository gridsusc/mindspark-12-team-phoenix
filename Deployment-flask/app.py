import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import gmplot
import folium
import random

app = Flask(__name__)
model = pickle.load(open('RNmodel.pkl', 'rb'))
clean = pd.read_csv('clean.csv')
clean = clean.drop_duplicates()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    final_features = []

    for x in request.form.values():
        final_features.append(x)
    lon = "-83.092860"
    lat = "40.108910"
    city = final_features[0]
    St = final_features[1]
    date = final_features[2].split("T")[0]
    time = final_features[2].split("T")[1]+":00"
    LocPoints = getLocationPoints(city,St)
    results = []

    for index, row in LocPoints.iterrows():
        pressure, humidity, temperature, visiibility, windSpeed = getWeatherAttributes(row['Start_Lng'],row['Start_Lat'],date,time)
        names = ['Start_Lng', 'Start_Lat', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
        final_features = [lat,lon,temperature,humidity,pressure,visiibility,windSpeed]
        df = pd.DataFrame(columns=names)
        df.loc[0] = final_features
        prediction = model.predict(df)
        predict = random.randint(1,5)
        results.append(((row['Start_Lat'],row['Start_Lng']),predict))
    prediction = [1234]
    output = round(prediction[0], 2)
    print(results)
    folium_map =plotGraph(results)
    return folium_map._repr_html_()
    #return render_template('index.html', prediction_text='Severity for given location is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

def getWeatherAttributes(lon, lat,userdate,usertime):
    path = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid=64d58baaf93e9cbfdb796a96c212998a'
    r = requests.get(path).json()

    locList = r.get('list')
    precipitation= ''
    visiibility= ''
    windSpeed =''
    temperature=''
    humidity=''
    pressure = ''
    hr = int(usertime.split(":")[0])
    remainder = hr % 3
    if(remainder>=2):
        hr = hr+1
    else:
        hr = hr - remainder
    if hr<=9:
        hr= "0"+str(hr)
    else:
        hr = str(hr)
    usertime = hr+":00:00"

    for l in locList:
        d = l.get('dt_txt').split(" ")[0]
        t = l.get('dt_txt').split(" ")[1]

        if(d == userdate and t == usertime):
            pressure = float(l.get('main').get('pressure'))
            pressure = float(pressure*0.02953)
            humidity = float(l.get('main').get('humidity'))
            temperature = float(l.get('main').get('temp'))
            temperature = (((temperature-273.15)*9)/5) + 32
            visiibility = round(float(l.get('visibility'))*0.000621371,2)
            windSpeed = float(l.get('wind').get('speed'))*2.23694
            precipitation = float(l.get('pop'))
            break
    return pressure,humidity,temperature,visiibility,windSpeed

def getLocationPoints(city,st):
    val = clean.loc[(clean['City']==city) & (clean['Street'] == st),['Start_Lat','Start_Lng']]
    return val

def get_dataframe(mytuple):
    mydict = {}
    x = []
    y = []
    z = []
    for x1, y1, z1 in mytuple:
      x.append(x1)
      y.append(y1)
      z.append(z1)

    mydict['lat'] = x
    mydict['lon'] = y
    mydict['sev'] = z

    df = pd.DataFrame(mydict)

    return df, x, y

def plotGraph(results):
    # df, lat_array, lon_array = get_dataframe(results)
    # lat = np.array(lat_array)
    # lon = np.array(lon_array)
    # groups = df.groupby('sev')
    # for name, group in groups:
    #     plt.plot(group.lat, group.lon, marker='o', linestyle='', markersize=15, label=name)
    #
    # plt.plot(lat, lon)
    # plt.axis('off')
    # plt.legend()
    # # for i in range(len(lat)):
    # #     text = "(" + str(lat[i]) + "," + str(lon[i]) + ")"
    # #     plt.annotate(text, (lat[i], lon[i]))
    # plt.savefig('my_image.png')
    #
    # # import gmplot package
    #
    # # GoogleMapPlotter return Map object
    # # Pass the center latitude and
    # # center longitude
    print(results)
    # gmap1 = gmplot.GoogleMapPlotter(33.961177,-118.282005, 13)
    # gmap1.draw("/Users/ritiagrawal/PycharmProjects/Deployment-flask/map11.html")

    map = folium.Map(location=[results[0][0][0], results[0][0][1]], zoom_start=17)
    for point in range(0, len(results)):
        sev = results[point][1]
        sevStr = str(sev)
        if(sev == 1):
            folium.Marker(results[point][0], popup=sev, icon=folium.Icon(color='darkblue', icon='map-marker', angle=0, prefix='fa')).add_to(map)
        elif(sev == 2):
            folium.Marker(results[point][0], popup=sev, icon=folium.Icon(color='yellow', icon='map-marker', angle=0, prefix='fa')).add_to(map)
        elif (sev == 3):
            folium.Marker(results[point][0], popup=sev, icon=folium.Icon(color='lightred', icon='map-marker', angle=0, prefix='fa')).add_to(map)
        elif (sev == 4):
            folium.Marker(results[point][0], popup=sev, icon=folium.Icon(color='darkred', icon='map-marker', angle=0, prefix='fa')).add_to(map)

    return map

if __name__ == "__main__":
    app.run(debug=True)