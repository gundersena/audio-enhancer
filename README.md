# AudioEnhancer
Takes cruddy conference call audio and upsamples it to HD Audio

To read more about the research, visit https://crimata.com/AIPost

## Usage

Clone the repo.

Download dataset at https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html and place it in the data folder.

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

Argument Reference | train

```
-h, --help        
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

Argument Reference | eval

```
-h, --help
-i MODEL_ID       model ID to use
-n NUM_EXAMPLES   number of examples/files to run
-w WAVFILE_LIST   list to pull examples from
-r SCALE          level of upsampling
-s SAMPLE_RATE    target sample rate
-a MAKE_AUDIO     bool, make audio or just spectrograms
-c FROM_CKPT      bool, begin eval at last checkpoint
```

Example

Train
```
python3 main.py train \
  -i 1 -c false -k True -d 8192 -x 100 -e 10 -b 32
```

Eval
```
python3 main.py train \
  -i 1 -n 10 -w ../data/val-files.txt -r 4 -s 16000 -True -c False
```


