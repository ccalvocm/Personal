# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:15:47 2020

@author: Carlos
"""
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import flopy.utils.binaryfile as bf
from flopy.utils.postprocessing import get_water_table

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy

print(sys.version)
print('numpy version: {}'.format(np.__version__))
print('matplotlib version: {}'.format(mpl.__version__))
print('flopy version: {}'.format(flopy.__version__))

#exe de MODFLOW
exe_name = r'C:\WRDAPP\MF2005.1_12\bin\mf2005.exe'

#carpeta de trabajo
workspace = os.path.join('data')
if not os.path.exists(workspace):
    os.makedirs(workspace)
    
ncol = 61
nrow = 61
nlay = 2

nper = 3
perlen = [365.25 * 200., 365.25 * 12., 365.25 * 18.]
nstp = [1000, 120, 180]
save_head = [200, 60, 60]
steady = True

# dis 
delr, delc = 50.0, 50.0
botm = np.array([-10., -30., -50.])

bot = -10.*np.zeros((nrow, ncol))

# Topografía como salar
xini = 0
yini = 61
zini = 10

for k in range(21):
    for i in range(xini,yini):
        for j in range(xini,yini):
            bot[i,j] = zini-.1
    xini = xini + 1
    yini = yini - 1
    zini = zini - 1.

# ibound - todo activo
ibound = np.ones((nlay, nrow, ncol), dtype= np.int)
# nivel inicial
ihead = np.zeros((nlay, nrow, ncol), dtype=np.float)

# lpf 
laytyp = [1,0]
hk = 15.
vka = hk/50.

# condiciones de borde 
# ghb 
colcell, rowcell = np.meshgrid(np.arange(0, ncol), np.arange(0, nrow))
index = np.zeros((nrow, ncol), dtype=np.int)
index[:, :10] = 1
index[:, -10:] = 1
index[:10, :] = 1
index[-10:, :] = 1

# definir filas y columnas de ghb
ghb_1050 = np.arange(10,51,1)
ghb_1 = np.ones(len(ghb_1050))

#layer row column head conductividad
lrchc = np.zeros((len(ghb_1050)*4, 5))
lrchc[:, 0] = 0
#filas
lrchc[:, 1] = np.concatenate((ghb_1*10,ghb_1050,ghb_1*50,ghb_1050))
#columnas
lrchc[:, 2] = np.concatenate((ghb_1050,ghb_1*50,ghb_1050,ghb_1*10))
lrchc[:, 3] = -10.
lrchc[:, 4] = 50.0 * 50.0 / 40.0

# crear ghb 
ghb_data = {0:lrchc}

# recarga 
rch = np.zeros((nrow, ncol), dtype=np.float)
rch[:] = 0.00001
# rch[index == 0] = -0.001
factor_rch = 1.
rch[:,0] = 8.E-4*factor_rch
rch[:,-1] = 8.E-4*factor_rch
rch[0,:] = 8.E-4*factor_rch
rch[-1,:] = 8.E-4*factor_rch

# crear recarga 
rch_data = {0: rch}

#Evaporacion
evt = rch = np.zeros((nrow, ncol), dtype=np.float)
evt[index == 0] = 0.0005
evt_data = {0: evt, 1: evt, 3: evt}

#well 
nwells = 2
lrcq = np.zeros((nwells, 4))
lrcq[0, :] = np.array((0, 30, 30, 0))
lrcq[1, :] = np.array([1, 30, 30, 0])
lrcqw = lrcq.copy()
factor_bombeo = 1.
lrcqw[0, 3] = -250*factor_bombeo
lrcqsw = lrcq.copy()
lrcqsw[0, 3] = -250.*factor_bombeo
lrcqsw[1, 3] = -25.*factor_bombeo

# crear well 
base_well_data = {0:lrcq, 1:lrcqw}
swwells_well_data = {0:lrcq, 1:lrcqw, 2:lrcqsw}

# swi2 
nadptmx = 10
nadptmn = 1
nu = [0.0, 0.023]
numult = 5.0
toeslope = nu[1] / numult  #0.005
tipslope = nu[1] / numult  #0.005
z1 = -10.0 * np.ones((nrow, ncol))
z1[index == 0] = -9.0

z = np.array([[z1, z1]])
iso = np.zeros((nlay, nrow, ncol), dtype=np.int)
iso[0, :, :][index == 0] = -2
iso[0, :, :][index == 1] = 1
iso[1, 30, 35] = -2
ssz=0.2


spd = {(0,199): [ 'SAVE HEAD','SAVE BUDGET','PRINT BUDGET'],
       (0,200): [],
       (0,399): ['SAVE HEAD','SAVE BUDGET','PRINT BUDGET'],
       (0,400): [],
       (0,599): ['SAVE HEAD','SAVE BUDGET','PRINT BUDGET'],
       (0,600): [],
       (0,799): ['SAVE HEAD','SAVE BUDGET', 'PRINT BUDGET'],
       (0,800): [],
       (0,999): ['SAVE HEAD','SAVE BUDGET','PRINT BUDGET'],
       (1,0): [],
       (1,59): ['SAVE HEAD','SAVE BUDGET','PRINT BUDGET'],
       (1,60): [],
       (1,119): ['SAVE HEAD','SAVE BUDGET','PRINT BUDGET'],
       (1,120): [],
       (2,0): [],
       (2,59): ['SAVE HEAD','SAVE BUDGET','PRINT BUDGET'],
       (2,60): [],
       (2,119): ['SAVE HEAD','SAVE BUDGET','PRINT BUDGET'],
       (2,120): [],
       (2,179): ['SAVE HEAD','SAVE BUDGET','PRINT BUDGET']}

modelname = 'MickeyMouse'
ml2 = flopy.modflow.Modflow(modelname, version='mf2005', exe_name=exe_name, model_ws=workspace)

discret = flopy.modflow.ModflowDis(ml2, nlay=nlay, nrow=nrow, ncol=ncol, laycbd=0,
                                   delr=delr, delc=delc, top=bot, botm=botm[1:],
                                   nper=nper, perlen=perlen, nstp=nstp)
bas = flopy.modflow.ModflowBas(ml2, ibound=ibound, strt=ihead)
lpf = flopy.modflow.ModflowLpf(ml2, laytyp=laytyp, hk=hk, vka=vka)
wel = flopy.modflow.ModflowWel(ml2, stress_period_data=swwells_well_data)
ghb = flopy.modflow.ModflowGhb(ml2, stress_period_data=ghb_data)
rch = flopy.modflow.ModflowRch(ml2, rech=rch_data)
evt = flopy.modflow.ModflowEvt(ml2, nevtop=3, surf=-11.0, evtr=evt_data, exdp=1.0, ievt=1, extension='evt')
swi = flopy.modflow.ModflowSwi2(ml2, nsrf=1, istrat=1, 
                                toeslope=toeslope, tipslope=tipslope, nu=nu,
                                zeta=z, ssz=ssz, isource=iso, nsolver=1,
                                nadptmx=nadptmx, nadptmn=nadptmn,
                                iswizt=255)
oc = flopy.modflow.ModflowOc(ml2, stress_period_data=spd)
pcg = flopy.modflow.ModflowPcg(ml2, hclose=1.0e-6, rclose=3.0e-3, mxiter=100, iter1=50)

ml2.write_input()
ml2.run_model(silent=False)

#%% Postproceso


plt.close("all")

# x e y
x = np.linspace(-1500, 1500, 7)
xcell = np.linspace(-1500, 1500, 7) + delr / 2.
xedge = np.linspace(-1525, 1525, 7)

hds = bf.HeadFile(r'.\\data\\'+modelname+'.hds') 
head = hds.get_data(totim=hds.get_times()[-1])
watertable = get_water_table(heads=head, nodata=-999.99)
plt.imshow(watertable, extent=[-1500,1500,-1500,1500])
plt.colorbar(label='Nivel (m.s.n.m.)',shrink=0.75)
plt.gca().set_xticks(x)
plt.gca().set_yticks(x)
plt.xlabel('Distancia x (m)')
plt.ylabel('Distancia y (m)')

# leet zeta
zfile = flopy.utils.CellBudgetFile(os.path.join(ml2.model_ws, modelname+'.zta'))
kstpkper = zfile.get_kstpkper()
zeta = []
for kk in kstpkper:
    zeta.append(zfile.get_data(kstpkper=kk, text='ZETASRF  1')[0])
zeta = np.array(zeta)
zeta2 = zeta
# figuras
fwid, fhgt = 8.00, 5.50
flft, frgt, fbot, ftop = 0.125, 0.95, 0.125, 0.925


# x e y
x = np.linspace(-1500, 1500, 61)
xcell = np.linspace(-1500, 1500, 61) + delr / 2.
xedge = np.linspace(-1525, 1525, 62)
years = [40, 80, 120, 160, 200, 6, 12, 18, 24, 30]

#color de linea
icolor = 5
colormap = plt.cm.jet
cc = []
cr = np.linspace(0.9, 0.0, icolor)
for idx in cr:
    cc.append(colormap(idx))
   
plt.rcParams.update({'legend.fontsize': 6, 'legend.frameon' : False})
fig = plt.figure(figsize=(fwid, fhgt), facecolor='w')
fig.subplots_adjust(wspace=0.25, hspace=0.25, left=flft, right=frgt, bottom=fbot, top=ftop)
# primer plot
ax = fig.add_subplot(2, 2, 1)
# limites
ax.set_xlim(-1500, 1500)
ax.set_ylim(-50, -10)
for idx in range(5):
    # layer 1
    ax.plot(xcell, zeta[idx, 0, 30, :], drawstyle='steps-mid', 
            linewidth=0.5, color=cc[idx], label='{:2d} años'.format(years[idx]))
    # layer 2
    ax.plot(xcell, zeta[idx, 1, 30, :], drawstyle='steps-mid',
            linewidth=0.5, color=cc[idx], label='_None')
ax.plot([-1500, 1500], [-30, -30], color='k', linewidth=1.0)
# leyenda
plt.legend(loc='lower left')
# ejes, leyenda y texto
ax.set_xlabel('Distancia horizontal (m)')
ax.set_ylabel('Elevación de la interfaz \n  salina (msnm)')
ax.text(0.025, .55, 'Layer 1', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.025, .45, 'Layer 2', transform=ax.transAxes, va='center', ha='left', size='7')

# segundo plot
ax = fig.add_subplot(2, 2, 2)
# limites
ax.set_xlim(-1500, 1500)
ax.set_ylim(-50, -10)
for idx in range(5, len(years)):
    # layer 1
    ax.plot(xcell, zeta[idx, 0, 30, :], drawstyle='steps-mid', 
            linewidth=0.5, color=cc[idx-5], label='{:2d} años'.format(years[idx]))
    # layer 2
    ax.plot(xcell, zeta[idx, 1, 30, :], drawstyle='steps-mid',
            linewidth=0.5, color=cc[idx-5], label='_None')
ax.plot([-1500, 1500], [-30, -30], color='k', linewidth=1.0)
# leyenda
plt.legend(loc='lower left')
# ejes, leyenda y texto
ax.set_xlabel('Distancia horizontal (m)')
ax.set_ylabel('Elevación de la interfaz \n  salina (msnm)')
ax.text(0.025, .55, 'Layer 1', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.025, .45, 'Layer 2', transform=ax.transAxes, va='center', ha='left', size='7')

# tercer plot
ax = fig.add_subplot(2, 2, 3)
# limites
ax.set_xlim(-1500, 1500)
ax.set_ylim(-50, -10)
for idx in range(5, len(years)):
    # layer 1
    ax.plot(xcell, zeta2[idx, 0, 30, :], drawstyle='steps-mid', 
            linewidth=0.5, color=cc[idx-5], label='{:2d} años'.format(years[idx]))
    # layer 2
    ax.plot(xcell, zeta2[idx, 1, 30, :], drawstyle='steps-mid',
            linewidth=0.5, color=cc[idx-5], label='_None')
ax.plot([-1500, 1500], [-30, -30], color='k', linewidth=1.0)
# leyenda
plt.legend(loc='lower left')
# ejes, leyenda y texto
ax.set_xlabel('Distancia horizontal (m)')
ax.set_ylabel('Elevación de la interfaz salina (msnm)')
ax.text(0.025, .55, 'Layer 1', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.025, .45, 'Layer 2', transform=ax.transAxes, va='center', ha='left', size='7')


