name=VLNBERT-train-speaker

flag="--vlnbert prevalent
      --aug data/prevalent_aug.json
      --train auglistener
      --features places365
      --maxAction 15
      --batchSize 8
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
      --speaker /home/joslin/VLN-BERT-Speaker/snap/speaker/state_dict/best_val_unseen_bleu
     "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python r2r_src/train.py $flag --name $name