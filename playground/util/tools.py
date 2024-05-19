


def list_all():
    import boto3
    from dotenv import dotenv_values
    import os
    userdata = dotenv_values(".env")
    os.environ["AWS_REGION"] = "ca-central-1"
    os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")


    s3 = boto3.client('s3')

    tasks = [
        'visual_manipulation',
        'manipulate_old_neighbor',
        'novel_adj',
        'novel_noun',
        'pick_in_order_then_restore',
        'rearrange',
        'rearrange_then_restore',
        'same_profile',
        'rotate',
        'scene_understanding',
        'simple_manipulation',
        'sweep_without_exceeding',
        'follow_order',
        'twist'
    ]

    bucket_name = 'vima'

    for task in tasks:
        folder_name = f'{task}/'  
        continuation_token = None
        folders = []

        # Loop to paginate through all objects
        while True:
            if continuation_token:
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name, Delimiter='/', ContinuationToken=continuation_token)
            else:
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name, Delimiter='/')

            # Collect common prefixes (subfolders)
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    folders.append(prefix['Prefix'])

            # Check if more results are available
            if response.get('IsTruncated'):  # True if there are more results
                continuation_token = response['NextContinuationToken']
            else:
                break

    # Print the collected subfolders
    for folder in folders:
        print(folder)