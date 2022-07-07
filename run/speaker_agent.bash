name=VLNBERT-train-speaker

flag="--vlnbert prevalent
     --attn soft 
     --angleFeatSize 128
     --train speaker
     --subout max 
     --dropout 0.6 
     --optim adam 
     --lr 1e-4 
     --iters 80000 
     --maxAction 35
     "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=7 python r2r_src_speaker/train.py $flag --name $name