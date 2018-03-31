# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:43:07 2017

@author: bvjs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf

cruisedata = pd.read_csv('a2324_cruise.csv')
nnumber = 'N5777V'
cruisefuel = 50
cruisedist = 100
fuelcost = 4
xax = 'RPM'
xpts = 50j
yax = 'ALT'
ypts = 50j
figx = 8
figy = 18

pltprops = {}
pltprops['RPM'] = 'RPM', 2200, 2750
pltprops['ALT'] = 'Altitude (ft)', 2000, 14000
pltprops['RNG'] = 'Range (nm)', 0, 1000
pltprops['TT'] = 'Time to Travel '+str(cruisedist)+' nm (hrs)', 0.5, 2
pltprops['FC'] = 'Fuel Cost of '+str(cruisedist)+' nm @ '+str(fuelcost)+'/gal ($)', 0 , 50
pltprops['FF'] = 'Fuel Flow (gph)', 6.5 , 20
pltprops['BHP'] = 'Brake Horsepower (%)', 0, 100
pltprops['TAS'] = 'True Airspeed (kts)', 90, 130
pltprops['END'] = 'Endurance (hrs)', 0, 10

cruisedata['END'] = cruisefuel / cruisedata['FF']
cruisedata['RNG'] = cruisedata['TAS'] * cruisedata['END']
cruisedata['TT'] = cruisedist / cruisedata['TAS']
cruisedata['FC'] = cruisedata['TT'] * cruisedata['FF'] * fuelcost
    
def xyz(points, values, xi, yi):
    rbfi = Rbf(points[:,0], points[:,1], values, smooth=0.002)
    zi = rbfi(xi, yi)
    return zi

def plot(rows, row, grid_z, points, name, colmap):  
    ax[row] = plt.subplot(rows, 1, row)
    a[row] = ax[row].imshow(grid_z.T, extent=(pltprops[xax][1],pltprops[xax][2],pltprops[yax][1],pltprops[yax][2]), origin='lower', cmap=colmap)
    ax[row].contour(grid_x, grid_y, grid_z, 15, linewidths=0.5, colors='k')
    ax[row].plot(points[:,0], points[:,1], 'ko')
    ax[row].set_aspect('auto')
    ax[row].axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
    ax[row].axes.grid(b=True, which='major', color='slategrey', linestyle='--')
    ax[row].set_title(name)
    ax[row].set_xlabel(pltprops[xax][0])
    ax[row].set_ylabel(pltprops[yax][0])
    fig.colorbar(a[row], ax=ax[row]) 

grid_x, grid_y = np.mgrid[pltprops[xax][1]:pltprops[xax][2]:xpts, pltprops[yax][1]:pltprops[yax][2]:ypts]
grdshp = grid_x.shape
cruiseinterp = pd.DataFrame()
cruiseinterp[xax] = xi = grid_x.flatten('C')
cruiseinterp[yax] = yi = grid_y.flatten('C')
ax = {}
a = {}
plt.close("all")
fig = plt.figure()
fig.set_size_inches(figx, figy)
pltprops_use = list(pltprops.keys())
pltprops_use.remove(xax)
pltprops_use.remove(yax)
for idx, pltprop in enumerate(pltprops_use):
    cruiseinterp[pltprop] = xyz(cruisedata[[xax,yax]].values, cruisedata[pltprop].values, xi, yi)
for idx, pltprop in enumerate(pltprops_use):
    cruiseinterp[cruiseinterp[pltprop] < pltprops[pltprop][1]] = np.nan
    cruiseinterp[cruiseinterp[pltprop] > pltprops[pltprop][2]] = np.nan
for idx, pltprop in enumerate(pltprops_use):
    plot(len(pltprops), idx+1, cruiseinterp[pltprop].reshape(grdshp), cruisedata[[xax,yax]].values, pltprops[pltprop][0], 'jet')
fig.tight_layout()
fig.savefig(nnumber+' Cruise Performance.pdf', format='pdf', dpi=300)

#def xy(XX, YY, ZZ, slots, slot, colmap):
#    ax = plt.subplot(slots,1,slot)
#    items = cruisedata[ZZ].unique()
#    nitems = range(len(items))
#    cm = plt.get_cmap(colmap) 
#    cNorm  = colors.Normalize(vmin=0, vmax=nitems[-1])
#    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
#    for idx, item in enumerate(items):
#        colorVal = scalarMap.to_rgba(nitems[idx])
#        cruisedata_use = cruisedata[cruisedata[ZZ] == item]
#        ax.plot(cruisedata_use[XX], cruisedata_use[YY], label=item, color=colorVal)        
#    ax.axes.grid(b=True, which='minor', color='slategrey', linestyle='--')
#    ax.axes.grid(b=True, which='major', color='slategrey', linestyle='--')
#    ax.axes.legend(loc='upper left', shadow=False)
#    ax.axes.set_xlabel(XX)
#    ax.axes.set_ylabel(YY)