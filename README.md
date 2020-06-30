# sarscov2-ru

### R installation:
```
   source ~/anaconda3/bin/activate bio
   ~/anaconda3/envs/bio/bin/./conda install r-essentials r-base
   
   # install adegenet:
   ~/anaconda3/envs/bio/bin/./conda install -c conda-forge r-adegenet 

   # install udunits2:
   ~/anaconda3/envs/bio/bin/./conda install -c conda-forge r-udunits2
```

installing packages:<br>

for installing `aphid`:

```
   # open R interpreter:
   ~/anaconda3/envs/bio/bin/./R
   
   # run:
   install.packages('aphid', repos='http://cran.us.r-project.org')
```