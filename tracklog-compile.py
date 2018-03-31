 # -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:43:07 2017

@author: bvjs
"""

import numpy as np
import pandas as pd

###########################

trackfiles = [#'059BAFBB-92A1-49FD-8A36-6824FEE0204F', # paraglider
              'BBC5821E-AFE0-441F-B634-38DE390480F8', # Bonanza
              '72A02B75-9758-4698-ACF1-BCEEA156F60D', # Archer
              '0FAC0350-4515-49ED-B1B1-5A276FBABDE4', # N122FR
              'E8E0430D-E4A3-4665-A6AB-23914C73BB56', # N122FR
              '832E6A08-BC72-4956-90A0-D6115FC32DAE', # N122FR
              'D0E9BA4C-25BF-4F67-ADF3-BABFD31A7904', # N122FR
              '6D03501A-F4BE-439A-9DEB-5CAE18B627C2', # N122FR
              '29F8CB78-F34F-4989-B147-3A5B0EA22EFF', # N122FR
              #'8759A7D3-56CE-4751-B57E-B6734056348A', # ?
              'C971CE6F-FFE7-44AE-BDA1-5BE4FBA3DA61', # N122FR
              '0DDCF457-6E55-465A-8416-E6546FECAE97', # N122FR
              '8D47E15C-DC41-4B32-8DDD-078D46141D65', # N231BZ
              '25FD0D74-EF75-4345-BDF8-215D1430C664', # N231BZ
              'E3E136B1-9B89-495D-BCE5-6B424795AB49', # N231BZ
              '37AEDD08-376E-4B02-9C3E-73FA4339B75E', # N93977
              #'E8D9CBEE-5FC7-41DD-89D8-1991450AC1C0', # FSX
              '188A123E-B211-47ED-8767-EACC09E0E8EF', # IFR
              #'6787844C-9252-4235-AD55-2A14D76FEA4B', # FSX
              #'72BE10F3-0D95-47DC-9E45-56DE670767D6', # FSX
              'A6716C93-90F3-4465-9F85-FE2347B5CE6B', # IFR
              #'7925868D-555D-4F81-BCED-F9D3C3D87452', # FSX
              #'0BA03CDD-029E-4A80-A2CD-67099E47E67C', # FSX
              'D78BFB47-6211-4307-BAE2-346CCF2B8FCA', # IFR
              'EB0BBB4D-99E8-4FDB-8D01-6598E77CCDB4', # N5777V
              'D487626A-6D98-4BF4-8CB8-C8ECAD3667B7', # N5777V
              '376BBC96-9CE0-4A94-BBBC-09CFCCB3E807', # N5777V
              '517D57E7-DEF4-4572-B689-11B598D9556F', # N5777V
              'AA53872B-17D8-4DE6-A11D-957CF339A0A0', # N5777V
              '9B628530-E7E4-41D8-99D4-840A660A1435', # N5777V
              '3D8AF3C5-84A5-4EA2-8AAC-A989BA941263', # N5777V
              '5A346F34-79BC-4AEA-977C-818C7542CC3E', # N5777V
              '9A58B1B6-07FB-4319-9674-CD8C0C43B488', # N5777V
              #'CB40E875-F0A9-48EB-A1EE-9D0F4A371156', # SAT-DAL
              'CBC38C71-1358-47E0-B3F1-324AAAF87102', # N5777V
              'F76D6291-9F0B-4998-BAC7-B614B05188E8', # N5777V
              '4D079303-A61A-4940-9118-70C07AB003A8', # N5777V
              '99DA8E50-BB25-4CA4-99CB-8CDB0F87C1D2', # N5777V
              '3BD2A8BB-D6E5-4F41-A800-99FF3B4CF679'] # N5777V
figx = 18
figy = 15
resample_rate = 2
spd_bin = 5
alt_bin = 500
crs_bin = 15
speed_thresh = 50
deltarm_thresh = 4

###########################

trackdata = pd.DataFrame()
for trackfile in trackfiles:
    url = 'https://plan.foreflight.com/tracklogs/export/'+trackfile+'/csv'    
    data = pd.read_csv(url, skiprows=2)
    data.sort_values('Timestamp', axis=0, ascending=True, inplace=True)
    data['Datetime'] = pd.to_datetime(data['Timestamp'], unit='ms')
    data['Altitude_deltarm'] = data['Altitude'].diff().rolling(30).mean()
    data = data.set_index('Datetime')
    data = data.resample(str(resample_rate)+'S').mean()
    data['Timestamp_cum'] = (data['Timestamp'] - data['Timestamp'][0]) / 1000
    data['Timestamp_delta'] = data['Timestamp'].diff()
    data['Latitude_delta'] = data['Latitude'].diff()
    data['Longitude_delta'] = data['Longitude'].diff()
    data['Altitude_delta'] = data['Altitude'].diff()
    data['Course_delta'] = data['Course'].diff()
    data['Speed_delta'] = data['Speed'].diff()
    data['Bank_delta'] = data['Bank'].diff()
    data['Pitch_delta'] = data['Pitch'].diff()
    data['Altitude_rate'] = data['Altitude_delta'] / resample_rate * 60
    data['Speed_rate'] = data['Speed_delta'] / resample_rate
    data['Course_rate'] = data['Course_delta'] / resample_rate
    data['Bank_rate'] = data['Bank_delta'] / resample_rate
    data['Pitch_rate'] = data['Pitch_delta'] / resample_rate
    data = data[pd.notnull(data['Timestamp'])]
    data['Altitude_bin'] = np.round(data['Altitude']/alt_bin,0)*alt_bin
    data['Altitude_bin'] = data['Altitude_bin'].map(int)
    data['Speed_bin'] = np.round(data['Speed']/spd_bin,0)*spd_bin
    data['Speed_bin'] = data['Speed_bin'].map(int) 
    data['Course_bin'] = np.round(data['Course']/crs_bin,0)*crs_bin
    data['Course_bin'] = data['Course_bin'].map(int)
    data['Status_deltarm'] = np.where(data['Altitude_deltarm'] > deltarm_thresh, 1, np.where(data['Altitude_deltarm'] < -deltarm_thresh, -1, 0))
    data['Status'] = data['Status_deltarm']
    header = pd.read_csv(url)[:1]
    for col in header.columns:
        data[col] = header[col]
    data['File'] = trackfile
    trackdata = trackdata.append(data)
    trackdata['Datapoint'] = trackdata.reset_index().index     
trackdata.to_csv('tracklog_analysis.csv')
