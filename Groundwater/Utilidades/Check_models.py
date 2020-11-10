# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:25:22 2020

@author: ccc
"""

import flopy as fp
import matplotlib.pyplot as plt


mf_dir = r'C:\gwv7\models\transitorioCalibracion'
strt = mf_dir + '\modelCalR2.hds'


plt.close("all")

hdobj = fp.utils.HeadFile(strt)
times = hdobj.get_times()
hdobj.plot(totim=times[-1], colorbar=True, contour = "True" , vmin=0,vmax=3000)
plt.title("Layer 1")
plt.show()



ml = fp.modflow.Modflow.load(mf_dir+"\modelCalR2.nam", model_ws='', verbose=False,
                               check=False, exe_name="mf2k", version = "mf2k"   )


ml.check(verbose=True, level=12)
ml.plot(colorbar=True, contour = "True")



#%%

#sw_dir = r'C:\gwv7\models\Peine\Transitorio'
#sw_dir = r'C:\gwv7\models\Quelana\Transitorio'
sw_dir = r'C:\gwv7\models\puntaBrava\Transitorio'

sw = fp.seawat.swt.Seawat.load(sw_dir+"\PuntaBrava.nam", model_ws='', verbose=False, exe_name="swt_v4x64", version = "seawat"   )

sw.check(verbose=True, level=12)