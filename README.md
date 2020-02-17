# AudioEnhancer
Takes cruddy conference call audio and upsamples it to HD Audio

To read more about the research, visit https://crimata.com/AIPost

## Usage

Repository Structure

* `./data`: the data and the metadata (you should not need to touch this folder)
* `./main`: the model and its respective utilites
* `./samples`: audio files and spectrograms generated from the neural net

```
cd ./main
python 3 main.py -h
```

It will then ask you if you'd like to train or evaluate.  To get help for each respectivly, just type:

```
python 3 main.py train -h
python 3 main.py eval -h
```

Argument Reference:

'''
Arguments:
  -h, --help        help
  -i MODEL_ID       an index you can add to querey the model later
  -c FROM_CKPT      bool, begin training session from a checkpoint
  -k NEW_DATA       bool, make new set of data
  -d DIM_SIZE       size of each audio sample
  -x NUM_FILES      number of files to train
  -e EPOCHS
  -b BATCH_SIZE
  -o CYCLE_LENGTH   length of cycle in epochs (only for SGDR)
  -m MAX_LR         max learning rate (only for SGDR)
  -n MIN_LR         min learning rate (only for SGDR)
```

