import boto3
import click


@click.command()
@click.argument('yaml_pths',
                nargs=-1,
                type=click.Path(file_okay=True, dir_okay=False))
def main(yaml_pths):
    boto3.setup_default_session(profile_name='rdml',
                                region_name='us-east-1')
    batch = boto3.client('batch')
    for yaml_pth in yaml_pths:
        cmd = ['python3', 'src/train_segmentation.py', yaml_pth]

        resp = batch.submit_job(
            jobName='rad_ml',
            jobQueue='rad_ml_queue',
            jobDefinition='rad_ml_training',
            containerOverrides={
                'command': cmd,
                'vcpus': 1,
                'memory': 15000,
            },
        )
        print(str(resp['jobId']))


if __name__ == "__main__":
    main()
