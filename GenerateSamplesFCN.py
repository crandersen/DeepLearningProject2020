"""
Finished on Mon January 11 2021

@author: chrisan@dtu.dk
"""

import numpy as np
from random import randint
from ase.build import bulk
from ase.visualize import view
from ase.io import write
from scipy.cluster.hierarchy import fcluster, linkage
from abtem.potentials import Potential
from abtem.waves import PlaneWave

num_examples = 500
num_classes = 2
L = 256

classes = ('WZ','ZB','Both')

dir_name = 'dataFCN'
first_number = 0

models_list = []
sites_list = []
classes_list = []

for i in range(num_examples):
    buildclass = randint(0,1)
    
    L_wire = randint(100,200) # Wire length 20 nm
    W_wire = 50 #Wire width 100 Å
    a_wire = 5.6 #Lattice constant
    
    # # Create structure
    if buildclass == 'WZ':
        atoms = bulk('GaAs','wurtzite',a_wire,a_wire,a_wire*1.63,orthorhombic=True)
        atoms = atoms.repeat((round(W_wire/a_wire),round(W_wire/a_wire),round(L_wire/(a_wire*1.63))))
    else:    
        atoms = bulk('GaAs','zincblende',a_wire,a_wire,a_wire,orthorhombic=True)
        atoms = atoms.repeat((round(W_wire/a_wire),round(W_wire/a_wire),round(L_wire/a_wire)))
    
    atoms.center(vacuum=0.0)
    
    xy = atoms.positions[:,:2]
    center = (max(xy[:,0])/2,max(xy[:,1])/2)
    
    r2 = ((xy-center)**2).sum(axis=1)
    
    rmin = atoms.cell[0][0]/2
    height = atoms.cell[2][2]
    
    cutatoms = atoms[r2<rmin**2]
    
    # view(cutatoms)
    
    zrotation = randint(0,90)
    if buildclass == 'WZ':
        cutatoms.rotate(90,'x',center=(0,0,0), rotate_cell=False)
        cutatoms.rotate(-45,'y',center=(0,0,0),rotate_cell=False)
    else:
        cutatoms.rotate(90,'x',center=(0,0,0), rotate_cell=False)
        cutatoms.rotate(45,'y',center=(0,0,0),rotate_cell=False)
    
    cutatoms.rotate(zrotation,'z',center=(0,0,0),rotate_cell=False)
    cutatoms.center(vacuum=0)
    size=np.diag(cutatoms.get_cell())

    # view(cutatoms)
    
    cutatoms.cell.niggli_reduce()

    cutatoms.set_cell((L,)*3)
    cutatoms.center()

    # tx=(L-size[0]-5)*(np.random.rand()-.5)
    # ty=(L-size[1]-5)*(np.random.rand()-.5)

    # cutatoms.translate((tx,ty,0))
    
    positions=cutatoms.get_positions()[:,:2]
    # clusters = fcluster(linkage(positions), 1, criterion='distance')
    
    # unique,indices=np.unique(clusters, return_index=True)
    
    # c=np.array([np.sum(clusters==u) for u in unique])-1
    # sites_list.append(np.array([np.mean(positions[clusters==u],axis=0) for u in unique]))
    
    c = np.full((1,len(positions)),buildclass)
    sites_list.append(positions)
    classes_list.append(c)
    models_list.append(cutatoms)
    
    print('Models finished: ' + str(100*(i+1)//num_examples) + '%')

from matplotlib import gridspec
import matplotlib.pyplot as plt

n=3
m=4

fig=plt.figure(figsize=(12,4))
gs=gridspec.GridSpec(n,m+2,width_ratios=[1,1,1,1,.05,.05])
gs.update(wspace=.025,hspace=.025)

for i in range(n):
    for j in range(m):
        
        k=i*m+j
        
        ax=plt.subplot(gs[i,j])
        
        classes=classes_list[k]
        
        sites=sites_list[k]
        
        #sites=sites[np.argsort(sites[:,2])]
        
        # cmap=discrete_cmap(m,'Paired')
        sc=ax.scatter(sites[:,0],sites[:,1],
                               s=30,c=classes,vmin=0,vmax=1,lw=0)
        
        ax.set_aspect('equal',adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0,L])
        ax.set_ylim([0,L])


inner = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=gs[:,4],height_ratios=[1,20,1])
cax=plt.subplot(inner[1])

cbar=plt.colorbar(sc,cax=cax,ticks=np.arange(1,m+1,1),orientation='vertical')
#plt.colorbar(sc, cax=cax, orientation='vertical',ticks=[-5,-2.5,0,2.5,5],label='$\epsilon_p$ [\%]')        

#plt.tight_layout()
plt.show()

for i, model in enumerate(models_list):
    
    
    # # # Building the potential
    potential = Potential(model, 
                          gpts=L,
                          slice_thickness=1, 
                          parametrization='kirkland', 
                          projection='infinite')
    
    wave = PlaneWave(
        energy=300e3 # acceleration voltage in eV
    )
    
    exit_wave = wave.multislice(potential)
    
    np.savez('{0}/points/points_{1:04d}.npz'.format(dir_name,first_number+i), sites=sites_list[i], classes=classes_list[i])
    exit_wave.write('{0}/wave/wave_{1:04d}.hdf5'.format(dir_name,first_number+i))
    write('{0}/model/model_{1:04d}.cfg'.format(dir_name,first_number+i),model)
    
    print('TEM images finnished: ' + str(100*(i+1)//num_examples) + '%')
    
    
    