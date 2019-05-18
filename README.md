# Image Searching Demo

## To run the training image
For testing, we use mounted volumes so code can be changed easily (in the volume).

### Prereq
* Put training python code in a folder, e.g. /Users/shengyipan/IdeaProjects/image-searching/python/
* Put training images in a folder, e.g. /Users/shengyipan/demo/bag_train/
* Create a model folder, e.g. /Users/shengyipan/demo/model/

### Command to run the image
```$bash
docker run -v /Users/shengyipan/demo/bag_train/:/training_data/ \
-v /Users/shengyipan/demo/model/:/model/ \
-v /Users/shengyipan/IdeaProjects/image-searching/python/:/python/ \
--rm -d psyking841/searching-training:0.1
```

### Minikube
* Mount the folders with training images and code to minikube VM using `--mount-string="/path/in/host:/path/in/vm"`. 
Or You can share folders to minikube VM using (Note you have to keep this alive, and it has some size constrain...)
`minikube mount /mnt/WDShare/image-searching/python/:/python`
Or You can mount using "nfs-share" to share folders but that only works with hyperkit.

Training data should be zipped in training_data.tar.gz. The pod will unfold it.
