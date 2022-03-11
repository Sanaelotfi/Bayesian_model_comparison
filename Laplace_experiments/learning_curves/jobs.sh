python3 main.py --model ResNet18 --width_multiplier 1 --epochs 300 --batch_size 128 --num_per_class 5000 --data_dir '/cmlscratch/goldblum/data' --num_runs 1 --save_model
python3 main.py --model ResNet18 --width_multiplier 2 --epochs 300 --batch_size 128 --num_per_class 5000 --data_dir '/cmlscratch/goldblum/data' --num_runs 1 --save_model
python3 main.py --model VGG19 --width_multiplier 1 --epochs 300 --batch_size 128 --num_per_class 5000 --data_dir '/cmlscratch/goldblum/data' --num_runs 1 --save_model
python3 main.py --model GoogLeNet --width_multiplier 1 --epochs 300 --batch_size 128 --num_per_class 5000 --data_dir '/cmlscratch/goldblum/data' --num_runs 1 --save_model
