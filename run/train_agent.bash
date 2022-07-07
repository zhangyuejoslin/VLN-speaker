name=VLNBERT-train-batch16

flag="--vlnbert prevalent
      --aug data/prevalent_aug.json
      --test_only 0

      --train auglistener
      --features places365
      --maxAction 15
      --batchSize 16
      --feedback sample
      --lr 1e-5
      --iters 600000
      --optim adamW
      
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5
      --selfTrain
      --accumulateGrad
      --speaker /VL/space/zhan1624/VLN-speaker/snap/VLNBERT-train-speaker/state_dict/best_val_unseen_bleu
     "
   
       
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=5 python r2r_src_speaker/train.py $flag --name $name