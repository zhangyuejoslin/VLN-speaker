name=VLNBERT-train-prevalent_update35

flag="--vlnbert prevalent
      --aug data/prevalent_aug.json
      --test_only 0

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
      --using_ob 0
      --obj_img_feat_path /home/hlr/shared/data/joslin/img_features/obj_feat_10.npy
     "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python LSTM_r2r_src/train.py $flag --name $name