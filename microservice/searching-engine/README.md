# Searching Engine Service

### Command to run the image
```$bash
docker run -v /Users/shengyipan/demo/:/demo/ -e INDEX_FILE=/demo/final_result.csv \
-e IMAGES_DIR=/demo/bag_test/ -p 8543:8543 --rm -d psyking841/searching-engine:0.1
```

### Python Example
```python
from search import SearchingEngine
INDEX_DICT = SearchingEngine(index_csv_file=INDEX_FILE).index
similar_list = INDEX_DICT.get(file_name)
```

## RESTful Interface
### Help message for training in image searching service
Get help message.

URL: https://{HOST}:{PORT}/images/help

HTTP METHOD: GET

CURL Example:
curl -X GET -H 'x-api-key: {REST_API_TOKEN}' http://{HOST}:{PORT}/images/help

Response: plain text for how to use.

### Searching similar image
Will return a list of similar images.

URL: https://{HOST}:{PORT}/images/search

HTTP METHOD: POST

Body: 
* image_file - file - Path to the model building Python script

CURL Example:
curl -X POST -H 'Content-Type: multipart/form-data' -H 'x-api-key: {REST_API_TOKEN}' -F 'image_file=@/Users/shengyipan/demo/test.jpg' http://{HOST}:{PORT}/image/search

Response:
```
```

Response Example:
```
```

### Get image
Return the image based on the image name

URL: http://{HOST}:{PORT}/images/get

HTTP METHOD: GET

Parameters: 
* image_name - string - The name of the image to get

WGET Example:
wget -O bag_216_23.jpg http://localhost:8543/images/get?image_name=bag_216_23.jpg 

## Web Interface
### Display search results for a specific image

URL: http://{HOST}:{PORT}/web/images/display_results

Parameters: 
* image_name - string - The name of the image to search on

Example:
http://{HOST}:{PORT}/web/images/display_results?image_name=bag_220_5.jpg
