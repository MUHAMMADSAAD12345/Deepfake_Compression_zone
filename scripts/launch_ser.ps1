$python = "E:\audi deepfakes\venv\Scripts\python.exe"
$script = "E:\audi deepfakes\scripts\train_ser.py"
$args = @(
    "--data", "E:\audi deepfakes\data\ravdess",
    "--output", "E:\audi deepfakes\checkpoints\ser_weights.keras",
    "--epochs", "500",
    "--batch_size", "256"
)
$outFile = "E:\audi deepfakes\data\ser_out.log"
$errFile = "E:\audi deepfakes\data\ser_err.log"
Start-Process -NoNewWindow -FilePath $python -ArgumentList @($script) + $args -RedirectStandardOutput $outFile -RedirectStandardError $errFile
Write-Output "SER PID: $((Get-Process -Name python | Select-Object -Last 1).Id)"
