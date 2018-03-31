 # -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:43:07 2017

@author: bvjs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

###########################

trackfiles = [#'059BAFBB-92A1-49FD-8A36-6824FEE0204F', # paraglider
              #'BBC5821E-AFE0-441F-B634-38DE390480F8', # Bonanza
              #'72A02B75-9758-4698-ACF1-BCEEA156F60D', # Archer
              #'0FAC0350-4515-49ED-B1B1-5A276FBABDE4', # N122FR
              #'E8E0430D-E4A3-4665-A6AB-23914C73BB56', # N122FR
              #'832E6A08-BC72-4956-90A0-D6115FC32DAE', # N122FR
              #'D0E9BA4C-25BF-4F67-ADF3-BABFD31A7904', # N122FR
              #'6D03501A-F4BE-439A-9DEB-5CAE18B627C2', # N122FR
              #'29F8CB78-F34F-4989-B147-3A5B0EA22EFF', # N122FR
              #'8759A7D3-56CE-4751-B57E-B6734056348A', # ?
              #'C971CE6F-FFE7-44AE-BDA1-5BE4FBA3DA61', # N122FR
              #'0DDCF457-6E55-465A-8416-E6546FECAE97', # N122FR
              #'8D47E15C-DC41-4B32-8DDD-078D46141D65', # N231BZ
              #'25FD0D74-EF75-4345-BDF8-215D1430C664', # N231BZ
              #'E3E136B1-9B89-495D-BCE5-6B424795AB49', # N231BZ
              #'37AEDD08-376E-4B02-9C3E-73FA4339B75E', # N93977
              #'E8D9CBEE-5FC7-41DD-89D8-1991450AC1C0', # FSX
              #'188A123E-B211-47ED-8767-EACC09E0E8EF', # IFR
              #'6787844C-9252-4235-AD55-2A14D76FEA4B', # FSX
              #'72BE10F3-0D95-47DC-9E45-56DE670767D6', # FSX
              #'A6716C93-90F3-4465-9F85-FE2347B5CE6B', # IFR
              #'7925868D-555D-4F81-BCED-F9D3C3D87452', # FSX
              #'0BA03CDD-029E-4A80-A2CD-67099E47E67C', # FSX
              #'D78BFB47-6211-4307-BAE2-346CCF2B8FCA', # IFR
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
              '3BD2A8BB-D6E5-4F41-A800-99FF3B4CF679', # N5777V
              'E323F2F4-28F0-4A9F-9DDA-56E492AD3E41', # N5777V
              '2A3CD320-808C-421C-8711-632CF872F3DE', # N5777V
              '0A82B28D-2FBF-44C9-B7CE-FAB80380CBE8', # N5777V
              '7EA40596-473A-4063-AB13-4092B3773646', # N5777V
              'DE2A41AC-BC65-42AF-97F6-A95339FE368A', # N5777V
              'BF9D933A-DD9E-4AF0-A619-784A01786F2B', # N5777V
              '45C155F8-E32E-4E17-B23E-A047AE1E27CE', # N5777V
              'B1816407-5AF3-447A-81B7-A107B9606F00', # N5777V
              'BB537EE5-3071-4A0C-AC25-BDA487BCE942', # N5777V
              '3FED11DF-C813-4573-8A2D-B0E93AD54133', # N5777V
              '0E664EB1-04B9-43E1-AE0E-E40BC6739A2B', # N5777V
              '5EFB638E-FA55-4483-8095-E6C16E689CD9', # N5777V
              '52EF4DEA-DCAF-4C5D-BA4F-3B631038D2FC', # N5777V
              'FEAA88F1-1B95-43E8-ABB8-1B89046A6961', # N5777V,
              'CCCA0A9C-0D98-4C62-A35F-4CE4F2F1DD91', # N5777V,
              'DDEB149D-262D-4820-AFBD-45AD896F4EC5', # N5777V,
              '3049BD15-F206-445D-B85E-66749E4669C9', # N5777V,
              'CD2AAC0A-B306-490B-9564-FC53CA87E068', # N5777V,
              '9CBF257F-C8E2-40D5-AC7D-00EA7F9F2059', # N5777V,
              '0737CF13-AA2B-4FDD-9507-B99CEF4B54E1', # N5777V,
              '09E785A8-7CA1-4B50-8A9B-8B130E8F284F', # N5777V,
              '9AFDE5F4-19EE-4907-AB1D-178C5C6286C9', # N5777V,
              '0E700A55-3A0D-44F6-BD5F-AB80BC1D91F1', # N5777V,
              '2817EA98-2EDC-4BE2-A924-08B2D9FF6482', # N5777V,
              'ADD3679B-3E8E-4E29-A526-44A82C61FA90', # N5777V,
              '7DBEF314-6217-485B-A80E-6B46A7771704', # N5777V,
              '11C758FE-1680-4AE6-8AFB-E20EDC732CBB', # N5777V,
              '42D8A53F-73F7-4508-8773-0CFB56F0423A'] # N5777V

cruisefile = 'a2324_cruise_mod.csv'
nnumber = 'n5777v'
figx = 18
figy = 15
resample_rate = 2
spd_bin = 5
alt_bin = 500
crs_bin = 15
speed_thresh = 50
deltarm_thresh = 4
cruisedata_tempC = 0
cruisedata_wt = 2550
cruisedata_hp = 200
trackdata_tempC = 0
trackdata_wt = 2550

###########################

cruisedata = pd.read_csv(cruisefile)
cruisedata['tempC'] = cruisedata_tempC
cruisedata['wt'] = cruisedata_wt
cruisedata['tempK'] = 288.15 + cruisedata['tempC'] - 1.98 * cruisedata['ALT'] / 1000
cruisedata['presRat'] = (1 - 0.0065 * cruisedata['ALT'] / 3.28084 / 288.15) ** 5.2561
cruisedata['densSlg'] = 0.002377 * cruisedata['presRat'] / (cruisedata['tempK'] / 288.15)
cruisedata['bhp'] = cruisedata['%BHP'] / 100 * cruisedata_hp
cruisedata['kcas'] = cruisedata['TAS'] * (cruisedata['densSlg'] / 0.002377) ** 0.5
cruisedata['cl'] = cruisedata['wt'] / (0.5 * 0.002377 * (cruisedata['kcas'] * 6076 / 3600) ** 2 * 174)
cruisedata['powerND'] = cruisedata['bhp'] * 550 * ((cruisedata['densSlg'] / 2 * 174) ** 0.5) / cruisedata['wt'] ** 1.5
cruisedata['cdeta'] = cruisedata['powerND'] * cruisedata['cl'] ** 1.5
cruisedata['advRat'] = cruisedata['TAS'] * (6076 / 3600) / (cruisedata['RPM'] * 2 * np.pi / 60 * 76 / 12 / 2)
cruisedata['propRad'] = (cruisedata['TAS'] * 1215.22) / (cruisedata['advRat'] * cruisedata['RPM'])
powerND_fit = np.poly1d(np.polyfit(cruisedata['cl'], cruisedata['powerND'], 2))
cdeta_fit = np.poly1d(np.polyfit(cruisedata['cl'], cruisedata['cdeta'], 2))
advRat_fit = np.poly1d(np.polyfit(cruisedata['cl'], cruisedata['advRat'], 1))
fuelflow_fit = np.poly1d(np.polyfit(cruisedata['bhp'], cruisedata['FF'], 1))
cruisedata['powerND_fit'] = powerND_fit(cruisedata['cl'])
cruisedata['advRat_fit'] = advRat_fit(cruisedata['cl'])
cruisedata['cdeta_fit'] = cdeta_fit(cruisedata['cl'])

###########################

def xyz(points, values, xi, yi):
    rbfi = Rbf(points[:,0], points[:,1], values, smooth=0.002)
    zi = rbfi(xi, yi)
    return zi

###########################

trackdata = pd.DataFrame()
for trackfile in trackfiles:
    
    data = pd.read_csv('https://plan.foreflight.com/tracklogs/export/'+trackfile+'/csv', skiprows=2)
    data['File'] = trackfile
    data.sort_values('Timestamp', axis=0, ascending=True, inplace=True)
    data['Datetime'] = pd.to_datetime(data['Timestamp'], unit='ms')
    data = data[(data['Speed'] >= speed_thresh)]
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
    data['tempC'] = trackdata_tempC
    data['wt'] = trackdata_wt   
    data['tempK'] = 288.15 + data['tempC'] - 1.98 * data['Altitude'] / 1000
    data['presRat'] = (1 - 0.0065 * data['Altitude'] / 3.28084 / 288.15) ** 5.2561
    data['densSlg'] = 0.002377 * data['presRat'] / (data['tempK'] / 288.15)
    data['cl'] = data['wt'] / (0.5 * 0.002377 * (data['Speed'] * 6076 / 3600) ** 2 * 174)
    data['BHP_cruise'] = ((cdeta_fit.coeffs[0] * data['cl'] ** 2) + (cdeta_fit.coeffs[1] * data['cl']) + (cdeta_fit.coeffs[2])) / ((data['cl'] ** 1.5) * 550 / (data['wt'] ** 1.5) * (data['densSlg'] * 174 / 2) ** 0.5)
    data['powerND'] = data['BHP_cruise'] * 550 * ((data['densSlg'] / 2 * 174) ** 0.5) / data['wt'] ** 1.5
    data['cdeta'] = data['powerND'] * data['cl'] ** 1.5
    data['advRat'] = advRat_fit(data['cl'])
    data['Fuelflow'] = fuelflow_fit(data['BHP_cruise'])
    data['FuelEfficiency'] = data['Speed'] / data['Fuelflow']
    data['RPM'] = (data['Speed'] * 1215.22) / (data['advRat'] * cruisedata['propRad'].mean())
    trackdata = trackdata.append(data)
    trackdata['Datapoint'] = trackdata.reset_index().index     
trackdata.to_csv(nnumber+'_'+'tracklog_analysis.csv')

###########################

trackdata_climb = trackdata[(trackdata['Status'] > 0) & (trackdata['Altitude_rate'] > 0)]
trackdata_cruise = trackdata[(trackdata['Status'] == 0) & (trackdata['RPM'] >= 2200) & (trackdata['Altitude'] >= 2000)]
trackdata_descent = trackdata[trackdata['Status'] < 0]

###########################

columns = 3
rows = 3
ax = {}
pt = {}
plt.close("all")
fig = plt.figure()
fig.set_size_inches(figx, figy)

row = 0
x = 'Longitude'
y = 'Latitude'
c = 'Altitude'
cmap = 'jet'
ax[row] = plt.subplot(rows, columns, row+1)
pt[row] = ax[row].scatter(trackdata[x], trackdata[y], c=trackdata[c], cmap=cmap)
ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
ax[row].set_aspect('auto')
fig.colorbar(pt[row], ax=ax[row])
ax[row].set_xlabel(x)
ax[row].set_ylabel(y)
ax[row].set_title(nnumber+': '+y+' vs. '+x+' ('+c+')')

row = 1
x = 'Datapoint'
y = 'Altitude'
c = 'Status'
cmap = 'RdYlGn'
ax[row] = plt.subplot(rows, columns, row+1)
pt[row] = ax[row].scatter(trackdata[x], trackdata[y], c=trackdata[c], cmap=cmap)
ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
ax[row].set_aspect('auto')
fig.colorbar(pt[row], ax=ax[row])
ax[row].set_xlabel(x)
ax[row].set_ylabel(y)
ax[row].set_title(nnumber+': '+y+' vs. '+x+' ('+c+')')

row = 2
x = 'Course'
y = 'Speed'
c = 'Altitude'
cmap = 'jet'
ax[row] = plt.subplot(rows, columns, row+1)
pt[row] = ax[row].scatter(trackdata[x], trackdata[y], c=trackdata[c], cmap=cmap)
ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
ax[row].set_aspect('auto')
fig.colorbar(pt[row], ax=ax[row])
ax[row].set_xlabel(x)
ax[row].set_ylabel(y)
ax[row].set_title(nnumber+': '+y+' vs. '+x+' ('+c+')')

row = 3
x = 'RPM'
y = 'ALT'
c = 'TAS'
cmap = 'jet'
grid_x, grid_y = np.mgrid[2200:2700:50j, 2000:14000:50j]
grdshp = grid_x.shape
cruiseinterp = pd.DataFrame()
cruiseinterp['x'] = xi = grid_x.flatten('C')
cruiseinterp['y'] = yi = grid_y.flatten('C')
cruiseinterp[c] = xyz(cruisedata[[x,y]].values, cruisedata[c].values, xi, yi)
cruiseinterp[cruiseinterp[c] < 70] = np.nan
cruiseinterp[cruiseinterp[c] > 130] = np.nan
ax[row] = plt.subplot(rows, columns, row+1)
pt[row] = ax[row].imshow(cruiseinterp[c].reshape(grdshp).T, extent=(2200,2700,2000,14000), origin='lower', cmap=cmap)
ax[row].contour(grid_x, grid_y, cruiseinterp['TAS'].reshape(grdshp), 15, linewidths=0.5, colors='k')
ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
ax[row].set_aspect('auto')
fig.colorbar(pt[row], ax=ax[row])
ax[row].set_xlabel(x)
ax[row].set_ylabel(y)
ax[row].set_xlim(2200,2700)
ax[row].set_ylim(2000,14000)
pt[row].set_clim([80, 140])
ax[row].set_title(nnumber+': '+y+' vs. '+x+' ('+c+')')

row = 4
x = 'RPM'
y = 'Altitude'
c = 'Speed'
cmap = 'jet'
cruiseinterp[c] = xyz(trackdata_cruise[[x,y]].values, trackdata_cruise[c].values, xi, yi)
cruiseinterp[cruiseinterp[c] < 70] = np.nan
cruiseinterp[cruiseinterp[c] > 130] = np.nan
trackdata_cruise[trackdata_cruise[c] < 70] = np.nan
trackdata_cruise[trackdata_cruise[c] > 130] = np.nan
ax[row] = plt.subplot(rows, columns, row+1)
pt[row] = ax[row].imshow(cruiseinterp[c].reshape(grdshp).T, extent=(2200,2700,2000,14000), origin='lower', cmap=cmap)
ax[row].contour(grid_x, grid_y, cruiseinterp[c].reshape(grdshp), 15, linewidths=0.5, colors='k')
ax[row].scatter(trackdata_cruise[x], trackdata_cruise[y], marker='o', s=40, facecolors='none', edgecolors='k')
ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
ax[row].set_aspect('auto')
fig.colorbar(pt[row], ax=ax[row])
ax[row].set_xlabel(x)
ax[row].set_ylabel(y)
ax[row].set_xlim(2200,2700)
ax[row].set_ylim(2000,14000)
pt[row].set_clim([80, 140])
ax[row].set_title(nnumber+': '+y+' vs. '+x+' ('+c+')')

row = 5
x = 'RPM'
y = 'FuelEfficiency'
c = 'Altitude'
cmap = 'jet'
ax[row] = plt.subplot(rows, columns, row+1)
pt[row] = ax[row].scatter(trackdata[x], trackdata[y], c=trackdata[c], cmap=cmap)
ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
ax[row].set_aspect('auto')
fig.colorbar(pt[row], ax=ax[row])
ax[row].set_xlabel(x)
ax[row].set_ylabel(y)
ax[row].set_ylim(0,30)
ax[row].set_title(nnumber+': '+y+' vs. '+x+' ('+c+')')

row = 6
x = 'cdeta'
y = 'cl'
c = 'Altitude'
cmap = 'jet'
ax[row] = plt.subplot(rows, columns, row+1)
pt[row] = ax[row].scatter(trackdata[x], trackdata[y], c=trackdata[c], cmap=cmap)
pt[row] = ax[row].scatter(cruisedata[x], cruisedata[y], c=cruisedata['ALT'], cmap=cmap, marker="x")
ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
ax[row].set_aspect('auto')
fig.colorbar(pt[row], ax=ax[row])
ax[row].set_xlabel(x)
ax[row].set_ylabel(y)
ax[row].set_xlim(0,0.1)
ax[row].set_ylim(0,1)
ax[row].set_title(nnumber+': '+y+' vs. '+x+' ('+c+')')

row = 7
x = 'Speed_bin'
y = 'Altitude_rate'
ax[row] = plt.subplot(rows, columns, row+1)
pt[row] = trackdata_climb.boxplot(column=y, by=x, ax=ax[row])
ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
ax[row].set_aspect('auto')
ax[row].set_ylim(0,4000)
ax[row].set_title(nnumber+': Climb_rate vs. Speed_bin (>+/- thresh)')

row = 8
x = 'Altitude_bin'
y = 'Speed'
ax[row] = plt.subplot(rows, columns, row+1)
pt[row] = trackdata_cruise.boxplot(column=y, by=x, ax=ax[row])
ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
ax[row].set_aspect('auto')
ax[row].set_title(nnumber+': Cruise_speed vs. Altitude_bin')

fig.suptitle(nnumber+' tracklog analysis: ('+str(resample_rate)+'sec avgs)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.subplots_adjust(top=0.95)
fig.savefig(nnumber+'_'+'tracklog_analysis.pdf', format='pdf', dpi=200)
