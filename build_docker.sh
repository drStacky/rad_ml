#!/usr/bin/env bash

echo 'Building image'
sudo nvidia-docker build -f Dockerfile -t rad_ml:latest .
sudo docker tag rad_ml:latest 769003761693.dkr.ecr.us-east-1.amazonaws.com/mstackpo/rad_ml:latest
echo 'Signing into AWS'
sudo $(aws --profile rdml ecr get-login --no-include-email --region us-east-1)
echo "Pushing image to ECR"
sudo docker push 769003761693.dkr.ecr.us-east-1.amazonaws.com/mstackpo/rad_ml:latest
