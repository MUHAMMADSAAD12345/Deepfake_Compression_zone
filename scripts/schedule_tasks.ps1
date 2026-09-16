$ferCmd = 'cmd /c ""E:\audi deepfakes\venv\Scripts\python.exe" "E:\audi deepfakes\scripts\train_fer.py" --data "E:\audi deepfakes\data\fer2013" --output "E:\audi deepfakes\checkpoints\fer2013_vgg19.keras" --epochs 250 --batch_size 128 > "E:\audi deepfakes\data\fer_out3.log" 2> "E:\audi deepfakes\data\fer_err3.log""'
$serCmd = 'cmd /c ""E:\audi deepfakes\venv\Scripts\python.exe" "E:\audi deepfakes\scripts\train_ser.py" --data "E:\audi deepfakes\data\ravdess" --output "E:\audi deepfakes\checkpoints\ser_weights.keras" --epochs 500 --batch_size 256 > "E:\audi deepfakes\data\ser_out3.log" 2> "E:\audi deepfakes\data\ser_err3.log""'

schtasks /CREATE /SC ONCE /TN "FERTraining" /TR $ferCmd /ST 00:00 /F
schtasks /RUN /TN "FERTraining"
Write-Output "FER task scheduled and launched"

Start-Sleep -Seconds 5

schtasks /CREATE /SC ONCE /TN "SERTraining" /TR $serCmd /ST 00:00 /F
schtasks /RUN /TN "SERTraining"
Write-Output "SER task scheduled and launched"
