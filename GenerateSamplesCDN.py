"""
Finished on Mon January 11 2021

@author: chrisan@dtu.dk
"""


from ase.build import bulk
from ase.visualize import view
from random import randint
from abtem.potentials import Potential
from abtem.structures import orthogonalize_cell
from abtem.waves import PlaneWave
from abtem.transfer import CTF
from abtem.noise import poisson_noise

# # 500 samples of each structure type was used in the training.
# # This was generated by running the script two times. 

# # First run: 
# total = 500 
# buildstructure = 'ZB'
#
# # Second run:
# total = 500 
# buildstructure = 'WZ'


total = 1 
buildstructure = 'ZB' # 'WZ' or 'ZB'

for i in range(total):
    L_wire = randint(150,250) # Wire length 15 to 25 nm
    W_wire = 100 # Wire width 10 nm
    a_wire = 5.6 # Lattice constant
    
    # # Create structures
    if buildstructure == 'WZ':
        atoms = bulk('GaAs','wurtzite',a_wire,a_wire,a_wire*1.63,orthorhombic=True)
        atoms = atoms.repeat((round(W_wire/a_wire),round(W_wire/a_wire),round(L_wire/(a_wire*1.63))))
    else:    
        atoms = bulk('GaAs','zincblende',a_wire,a_wire,a_wire,orthorhombic=True)
        atoms = atoms.repeat((round(W_wire/a_wire),round(W_wire/a_wire),round(L_wire/a_wire)))
    
    atoms.center(vacuum=0.0)

    # # Cutting wire    
    xy = atoms.positions[:,:2]
    center = (max(xy[:,0])/2,max(xy[:,1])/2)
    
    r2 = ((xy-center)**2).sum(axis=1)
    
    rmin = atoms.cell[0][0]/2
    height = atoms.cell[2][2]
    
    cutatoms = atoms[r2<rmin**2]
    
    # # Show generated structure:
    # # Uncomment only for a single example. Used for illustrating samples.
    # view(cutatoms)
    
    # # Rotate samples to zone axis:
    zrotation = randint(0,90)
    if buildstructure == 'WZ':
        cutatoms.rotate(90,'x',center=(0,0,0), rotate_cell=False)
        cutatoms.rotate(-45,'y',center=(0,0,0),rotate_cell=False)
    else:
        cutatoms.rotate(90,'x',center=(0,0,0), rotate_cell=False)
        cutatoms.rotate(45,'y',center=(0,0,0),rotate_cell=False)
    
    # # Rotate randomly along zone axis:
    cutatoms.rotate(zrotation,'z',center=(0,0,0),rotate_cell=False)
    
    # # Adding vacuum around sample
    cutatoms.cell.niggli_reduce()
    cutatoms.center(vacuum=60.0)
    
    # # Show sample after rotation and with vaccum added. 
    # # Uncomment only for a single example. Used for illustrating samples.
    # view(cutatoms)
    
    # # # # # # # # # # # # # # # # # # # # #
    # # abtem - Simulating TEM Images:  # # #
    # # # # # # # # # # # # # # # # # # # # #
    
    # # Building the potential
    cutatoms = orthogonalize_cell(cutatoms)
    potential = Potential(cutatoms, 
                          gpts=512,
                          slice_thickness=1, 
                          parametrization='kirkland', 
                          projection='infinite')
    

    wave = PlaneWave(
        energy=300e3 # acceleration voltage in eV
    )
    
    exit_wave = wave.multislice(potential)
    
    # # Show exit_wave before adding noise and ctf
    # # Uncomment only for a single example. Used for illustrating samples.
    # exit_wave.intensity().mean(0).show();
    
    
    ctf = CTF(
        energy = wave.energy,
        semiangle_cutoff = 20, # mrad
        focal_spread = 40, # Å 40 originally
        defocus = -85.34, # Å [C1] Originally -160
        Cs = -31.77e-6 * 1e10, # Å [C3] Originally -7e-6 * 1e10
    )
    
    # # Show ctf
    # # Uncomment only for a single example. Used for illustrating samples.
    # ctf.show(max_semiangle=50);
    
    image_wave = exit_wave.apply_ctf(ctf)
    
    # # Show exit_wave before adding noise but after adding ctf
    # # Uncomment only for a single example. Used for illustrating samples.
    # image_wave.intensity().mean(0).show();

    
    measurement = image_wave.intensity()
    noisy_measurement = poisson_noise(measurement, 5000)
    
    # # Show final image
    # # Uncomment only for a single example. Used for illustrating samples.
    # noisy_measurement.mean(0).show()
    
    noisy_measurement.mean(0).save_as_image(path='dataCDN/'+ buildstructure+ str(i) + '.jpeg');
    file = open("dataCDN/"+ buildstructure + str(i) +".txt", "w") 
    file.write("Structure: "+ buildstructure) 
    file.write("\nNumber of layers: " + str(round(L_wire/(a_wire*1.63))*2))
    file.write("\nNumber of pairs:" + str(len(cutatoms.positions)/2)) 
    file.write("\nDiameter in nm: " + str(2*rmin))
    file.write("\nHeight in nm: " + str(height))
    file.write("\nrotation around z in Deg: " + str(zrotation))
    file.close() 
    print('Finished: ' + str(100*i+1/total) + '%')
    
