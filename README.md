# mallory

Code for training and testing a CNN to learn bimap images of malware. The intuition is that a malware's binary can be turned into an image whose features can be learnt by a CNN. We use the [MalIMG dataset](http://old.vision.ece.ucsb.edu/spam/malimg.shtml) to demonstrate the idea. 

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
