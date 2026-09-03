if (-Not (Test-Path "cifar-10-batches-py")) {
    Invoke-WebRequest `
        -Uri "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" `
        -OutFile "cifar-10-python.tar.gz"

    tar -xzf "cifar-10-python.tar.gz"

    Remove-Item "cifar-10-python.tar.gz"
}