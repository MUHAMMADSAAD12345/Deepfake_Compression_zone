$ferLog = "E:\audi deepfakes\data\fer_out.log"
$ferErr = "E:\audi deepfakes\data\fer_err.log"
$serLog = "E:\audi deepfakes\data\ser_out.log"
$serErr = "E:\audi deepfakes\data\ser_err.log"

$ferArgs = @(
    "E:\audi deepfakes\scripts\train_fer.py",
    "--data", "E:\audi deepfakes\data\fer2013",
    "--output", "E:\audi deepfakes\checkpoints\fer2013_vgg19.keras",
    "--epochs", "250",
    "--batch_size", "128"
)
$serArgs = @(
    "E:\audi deepfakes\scripts\train_ser.py",
    "--data", "E:\audi deepfakes\data\ravdess",
    "--output", "E:\audi deepfakes\checkpoints\ser_weights.keras",
    "--epochs", "500",
    "--batch_size", "256"
)

$python = "E:\audi deepfakes\venv\Scripts\python.exe"

Start-Process -NoNewWindow -FilePath $python -ArgumentList $ferArgs -RedirectStandardOutput $ferLog -RedirectStandardError $ferErr
Write-Output "FER training started (PID: $((Get-Process -Name python | Select-Object -Last 1).Id))"

Start-Sleep -Seconds 2

Start-Process -NoNewWindow -FilePath $python -ArgumentList $serArgs -RedirectStandardOutput $serLog -RedirectStandardError $serErr
Write-Output "SER training started (PID: $((Get-Process -Name python | Select-Object -Last 1).Id))"
