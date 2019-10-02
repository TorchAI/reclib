
python main.py \
--dataset_name movielens1M \
--dataset_path data/ml-1m/ratings.dat \
--model_name xdfm \
--epoch 2 \
--learning_rate 0.001 \
--batch_size 2048 \
--weight_decay 1e-6 \
--device cuda:0 \
--save_dir chkpt