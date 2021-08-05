# rad_ml

MVP of training workflow to be deployed in AWS Batch.

1. Build the docker container using `Dockerfile`. Store in ECR.

1. Setup `Job Definition`, `Job Queue`, and `Compute Environment`
   in AWS Batch to accept jobs.
   
1. Put the file in `experiment` into S3.

    `data` - Directory containing `img` and `msk` chip pairs

    `data.json` - Contains paths to chip pairs in S3 as well as `train`, `valid`, or `test` designations

   `config.yml` - Contains training parameters, including a path to
   `data.json` in S3
   
1. Kick off training job(s) locally from the terminal.
    ```
    python run.py s3://<bucket>/config.yml
    ```
