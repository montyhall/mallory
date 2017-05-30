# mallory
Deep Learning for malware detection

Using a CNN to learn bimap images of malware from [MalIMG dataset](http://old.vision.ece.ucsb.edu/spam/malimg.shtml). 

## Data

 [Download MalIMG dataset](http://old.vision.ece.ucsb.edu/spam/malimg.shtml) and set `datasetRoot` in opt.lua
 


## Running

```bash
> th main.lua -gpu -threads 4 -batchSize 30 
```
