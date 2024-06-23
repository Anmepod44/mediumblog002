import json
import boto3
import base64

comprehend = boto3.client('comprehend')
rekognition = boto3.client('rekognition')
dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    print(event)

    # Get the base64 encoded image from the request body
    try:
        body = event['body']
        image_base64 = body
    except (KeyError, json.JSONDecodeError) as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'Invalid request, missing or malformed image data'})
        }
    
    # Decode the base64 encoded image
    try:
        image_bytes = base64.b64decode(image_base64)
    except base64.binascii.Error as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'message': f'Invalid base64 image data {e}'})
        }
    
    # Call Rekognition to detect labels in the image
    try:
        rekognition_response = rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10
        )
    except rekognition.exceptions.InvalidImageFormatException as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'message': f'Invalid image format {e}'})
        }
    
    # Extract labels
    labels = [label['Name'] for label in rekognition_response['Labels']]
    description = ', '.join(labels)  # Combine labels into a natural language description
    
    # Use Amazon Comprehend to detect key phrases in the description
    comprehend_response = comprehend.detect_key_phrases(Text=description, LanguageCode='en')
    key_phrases = [phrase['Text'] for phrase in comprehend_response['KeyPhrases']]
    
    # Print the key phrases
    print(f"Extracted key phrases: {key_phrases}")
    
    # Construct a DynamoDB query using the key phrases
    table_name = 'products'
    table = dynamodb.Table(table_name)
    
    # Use expression attribute names to handle reserved keywords
    expression_attribute_names = {'#n': 'name'}
    filter_expression = ' OR '.join([f"contains(#n, :phrase{i})" for i in range(len(key_phrases))])
    expression_attribute_values = {f":phrase{i}": phrase for i, phrase in enumerate(key_phrases)}
    
    # Query DynamoDB
    response = table.scan(
        FilterExpression=filter_expression,
        ExpressionAttributeNames=expression_attribute_names,
        ExpressionAttributeValues=expression_attribute_values
    )
    
    # Extract and print the matching items
    items = response.get('Items', [])
    print(f"Matching items: {items}")
    
    #Return the matching items
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Successfully queried DynamoDB',
            'items': items
        })
    }
