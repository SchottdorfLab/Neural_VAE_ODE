# Notes:
By Kathleen Higgins

### January 6th, 2026, 4:50pm
To run Dr. Schottdorf's Matlab code in the Visual Studio Code terminal: 
```
conda deactivate
/Applications/MATLAB_R2025b.app/bin/matlab -nodesktop -nosplash
```

```
matlab
```

```
cd('/Users/kathleenhiggins/Neural_VAE_ODE');

addpath(genpath('sandbox/MIND_experiments/mind_core'));
addpath('sandbox/MIND_experiments');

cd('src/mat_E65_data');   % so exp1’s load('./E65.mat') works
exp1_embeddingCrossval
```
