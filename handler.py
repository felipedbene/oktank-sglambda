import boto3, os, datetime
import json

def main(event, context):

    #Definitions
    #training_job_name = os.environ['training_job_name']
    #job = sm.describe_training_job(TrainingJobName=training_job_name)
    training_job_prefix = os.environ['training_job_prefix']


    sm = boto3.client('sagemaker')

    training_job_name = training_job_prefix+str(datetime.datetime.today()).replace(' ', '-').replace(':', '-').rsplit('.')[0]
    instanceType = os.environ['instance_type']
    instanceCount = int(os.environ['instance_count'])
    diskSize = int(os.environ['diskSize'])
    s3Uri = "s3://sm-benfelip-input/public/extract1576726379301.csv"

    hyperParameters= {
    "sagemaker_container_log_level": "20",
    "sagemaker_enable_cloudwatch_metrics": "false",
    "sagemaker_estimator": "\"RLEstimator\"",
    "sagemaker_job_name": "\"{}\"".format(training_job_name),
    "sagemaker_program": "\"train_vehicle_routing_problem.py\"",
    "sagemaker_region": "\"us-east-1\"",
    "sagemaker_s3_output": "\"s3://vrp-sagemaker-us-east-1-803186506512/\"",
    "sagemaker_submit_directory": "\"s3://vrp-sagemaker-us-east-1-803186506512/vrp-oktank-logistics-2020-01-03-19-45-39-122/source/sourcedir.tar.gz\""
    }
    
    algorithmSpecification = {
    "TrainingImage": "520713654638.dkr.ecr.us-east-1.amazonaws.com/sagemaker-rl-tensorflow:ray0.6.5-gpu-py3",
    "TrainingInputMode": "File",
    "MetricDefinitions": [
      {
        "Name": "episode_reward_mean",
        "Regex": "episode_reward_mean: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)"
      },
      {
        "Name": "episode_reward_max",
        "Regex": "episode_reward_max: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)"
      },
      {
        "Name": "episode_reward_min",
        "Regex": "episode_reward_min: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)"
      }
    ]
    }
      
    inputDataConfig = [
        {
            'ChannelName': 'inputfile',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': s3Uri,
                    'S3DataDistributionType': 'FullyReplicated',

                }
            },
            'ContentType': 'text',
            'CompressionType': 'None',
            'RecordWrapperType': 'None',
            'InputMode': 'File',
            'ShuffleConfig': {
                'Seed': 123
            }
        },
    ]
    
    outputDataConfig = {
    "KmsKeyId": "",
    "S3OutputPath": "s3://vrp-sagemaker-us-east-1-803186506512/"
  }
  
    resourceConfig = {
    "InstanceType": "{}".format(instanceType),
    "InstanceCount": instanceCount,
    "VolumeSizeInGB": diskSize
  }
    
    stoppingCondition = { "MaxRuntimeInSeconds": 300 }
    
    roleArn = "arn:aws:iam::803186506512:role/service-role/AmazonSageMaker-ExecutionRole-20191125T103535"
    
    enableManagedSpotTraining = False
    
    print("Starting training job %s" % training_job_name)


    resp = sm.create_training_job(
    TrainingJobName=training_job_name,
    HyperParameters=hyperParameters,
    AlgorithmSpecification=algorithmSpecification,
    InputDataConfig = inputDataConfig,
    OutputDataConfig = outputDataConfig,
    ResourceConfig = resourceConfig,
    StoppingCondition = stoppingCondition,
    RoleArn = roleArn,
    EnableManagedSpotTraining = enableManagedSpotTraining
    )
    
    #print(resp)

    return {
        "statusCode": 200,
        "body": json.dumps(resp)
    }
