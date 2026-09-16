@echo off
start /b "" "E:\audi deepfakes\venv\Scripts\python.exe" "E:\audi deepfakes\scripts\train_fer.py" --data "E:\audi deepfakes\data\fer2013" --output "E:\audi deepfakes\checkpoints\fer2013_vgg19.keras" --epochs 250 --batch_size 128 > "E:\audi deepfakes\data\fer_out2.log" 2> "E:\audi deepfakes\data\fer_err2.log"
echo FER launched
