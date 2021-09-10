python train_sentence_bert.py \
 --task_type=classification \
 --train_file=./data/weikong/classification_train_dataset.xlsx \
 --val_file=./data/weikong/classification_val_dataset.xlsx \
 --lr=1e-5 \
 --batch_size= 32 \
 --epochs= 20
