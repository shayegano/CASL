Config.py controlls all high-level flags.  
Please refer to Config.py for details of each flag.  
Important parameters are:
* GAME_CHOICE  : Parameter for choosing an environment to train/test
* USE_AUDIO    : Parameter for using audio or not during train/test
* USE_RNN      : Parameter for using LSTM or not during train/test
* USE_ATTENTION: Parameter for using attention or not during train/test

#### To Train:
Please first check Config.py to check whether correct parameters are used.  
Then train by `_ctt.sh`.
If train goes well, the table with time, episode, rolling score, etc will be displayed.  
Also, tensorboard will be running on the background.  
Trained models will be saved at `tmp-logs/*/checkpoints/network`.

#### To test:
Please enter the trained model directory in `LOGDIR = os.path.join('tmp-logs', 'DIRECTORY_IN_HERE')`, where an example of `DIRECTORY_IN_HERE` is '2017-10-28_23h-08m-24s'.  
Then please set `LOAD_CHECKPOINT` to True.  
Finally, test by `_play.sh`.
