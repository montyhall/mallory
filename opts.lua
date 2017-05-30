--
-- User: peyman
-- Date: 11/14/16
-- Time: 9:50 PM
-- on cims: th -i main.lua -serializedData /scratch/peyman
--

require 'image'
require 'dp'

local M = { }

function M.parse(arg)
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Train a CNN for float images')
    cmd:text('Example:')
    cmd:text("th main.lua -gpu -threads 4 -batchSize 100")
    cmd:text("th main.lua -gpu -testMode true")
    cmd:text('Options:')

    -- Dataset options
    cmd:option('-cache', 'results', 'subdirectory in which to save/log experiments')
    cmd:option('-datasetRoot', '/home/faratin/git/mallorypoc/Data/malimg/', 'path to training data')
    cmd:option('-trainRoot', 'train', 'path to training data')
    cmd:option('-validationRoot', 'validation', 'path to validation data')
    cmd:option('-testRoot', 'test', 'path to testing data')
    cmd:option('-testMode', 'none', 'boolean flag. if true, output <filename> <predicted_label> on each test image into predicted.txt')
    cmd:option('-portionTrain', 0.8, 'portion of data to train on: Default=0.8')
    cmd:option('-standardize', true, 'apply Standardize preprocessing')
    cmd:option('-inputPixels', '{3,256,256}', 'number of pixels in c,h,w  of image: Default: 3,256,25')
    cmd:option('-labelPixels', '{64,64}', 'number of pixels in c,w of label image: Default: 1,4096')
    cmd:option('-batchSize', 33, 'number of examples per batch')
    cmd:option('-validateBatchSize',    33,   'mini-batch size for validating')
    cmd:option('-testBatchSize',    33,   'mini-batch size for testing')

    -- data transforms
    cmd:option('-augment', 'none', 'perform data augmentation?')
    cmd:option('-hflip', 0.7, 'Probability of Horizontal flip? Default: 0.7')
    cmd:option('-vflip', 0.7, 'Probability of Vertical flip? Default: 0.7')
    cmd:option('-rotate', 80, 'Degree to rotate the image (plus some random amount) Default: 90')
    cmd:option('-brightness', 0.4, 'Brightness Default: 0.4')
    cmd:option('-contrast', 0.4, 'Contrast. Default: 0.4')
    cmd:option('-saturation', 0.4, 'Saturation Default: 0.4')
    cmd:option('-minscale', 256, 'Minimium scaling. Default: 256')
    cmd:option('-maxscale', 4000, 'Minimium scaling.  Default: 4000')

    -- Model options
    cmd:option('-network', 'model.net', 'pretrained network')
    cmd:option('-padding', 1, 'add math.floor(kernelSize/2) padding to the input of each convolution')
    cmd:option('-channels', '{r,g,b}', 'Number of input image channels: Default: {r,g,b}')
    cmd:option('-channelSize', '{16,32,32,64}', 'Number of output channels for each convolution layer: Default: {16, 32, 32, 64}')
    cmd:option('-kernelSize', '{3,3,3,3}', 'kernel size of each convolution layer. Height = Width')
    cmd:option('-kernelStride', '{1,1,1,1}', 'kernel stride of each convolution layer. Height = Width')
    cmd:option('-poolSize', '{2,2,2,2}', 'size of the max pooling of each convolution layer. Height = Width')
    cmd:option('-poolStride', '{2,2,2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
    cmd:option('-dropout', false, 'use dropout')
    cmd:option('-dropoutProb', '{0.5,0.5}', 'dropout probabilities')
    cmd:option('-batchNorm', true, 'use batch normalization. dropout is mostly redundant with this')
    cmd:option('-activation', 'ReLU', 'transfer function (ReLU | Tanh | Sigmoid). Default: ReLU')
    cmd:option('-loss', 'NLL', 'Loss (NLL). Default: NLL')
    cmd:option('-labelThresh', 0.5, 'Classification threshold for leaf scores.')
    cmd:option('-hiddensize1', 2000, 'Number of hidden units in first FC layer. Default: 2000')
    cmd:option('-hiddensize2', 3000, 'Number of hidden units in second FC layer. Default: 3000')
    cmd:option('-convOnly', false, 'If true then end-to-end Convolutional network')

    -- Optimization options
    cmd:option('-optimization', 'SGD')
    cmd:option('-nEpochs', 100, 'maximum number of epochs to try to find a better local minima for early-stopping. Default: 1000')
    cmd:option('-epochSize', 100000 / 256, 'Number of batches per epoch')
    cmd:option('-learningRate', 0.2, 'learning rate at t=0')
    cmd:option('-momentum', 0, 'momentum')
    cmd:option('-weightDecay', 1e-5, 'L2 penalty on the weights. Default=')
    cmd:option('-lrDecay', 1e-7, 'learning rate decay (in # samples) : Default: 1e-7')
    cmd:option('-earlystop', 40, 'Early stopping patience. Default: 20')

    -- Output options
    cmd:option('-showWs', true, 'Dump image of learnt Weights. Default: false')
    cmd:option('-visualize', false, 'visualize sample of input images. Default: true')
    cmd:option('-plot', true, 'plot learning errors. Default: true')
    cmd:option('-save', 'results', 'save directory')
    cmd:option('-silent', true, 'don\'t print anything to stdout')
    cmd:option('-serializedModel', 'results/model.net', 'path to trained model')

    -- Backend options
    cmd:option('-threads', 3, 'number of threads. Default: 3')
    cmd:option('-gpu', false, 'use gpus. Default: False')
    cmd:option('-device', 1, 'sets the device (GPU) to use')

    cmd:text()

    opt = cmd:parse(arg or {})
    if not opt.silent then table.print(opt) end

    --opt.classes = opt.classes:gsub('[{}]',''):split(',')
    opt.channels = opt.channels:gsub('[{}]',''):split(',')
    opt.channelSize = table.fromString(opt.channelSize)
    opt.inputPixels = table.fromString(opt.inputPixels)
    opt.labelPixels = table.fromString(opt.labelPixels)

    opt.kernelSize = table.fromString(opt.kernelSize)
    opt.kernelStride = table.fromString(opt.kernelStride)
    opt.poolSize = table.fromString(opt.poolSize)
    opt.poolStride = table.fromString(opt.poolStride)
    opt.dropoutProb = table.fromString(opt.dropoutProb)

    return opt
end

return M
