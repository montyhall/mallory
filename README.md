# mallory
Deep Learning for malware detection

Using a CNN to learn bimap images of malware from [MalIMG dataset](http://old.vision.ece.ucsb.edu/spam/malimg.shtml). 

## Data

- [Download MalIMG dataset](http://old.vision.ece.ucsb.edu/spam/malimg.shtml) 
 
- run `Data/split.lua` to split data into training, validation and test sets

- set `datasetRoot` in opt.lua


## Training

```bash
> th main.lua -gpu -threads 4 -batchSize 30 
```

## Testing

```bash
> th main.lua -testMode
```
