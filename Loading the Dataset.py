dataset_path = '/content/drive/MyDrive/Fabric project/Fabric-Defect-Classification_CNN-main/Dataset'
sub_folders = os.listdir(dataset_path)
IMAGE_SHAPE = (224,224)
i=0
images=[]
labels=[]

path_lists=[]
temp=sub_folders

for sub_folder in sub_folders:
  sub_folder_index = temp.index(sub_folder)
  label = sub_folder_index

  if label==2:
    continue
  if label==3:
    label=2
  path = dataset_path+'/'+sub_folder
  sub_folder_images = os.listdir(path)

  #Reading images from the subfolder one at a time
  for image in sub_folder_images:
    image_path = path+'/'+image
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image,IMAGE_SHAPE)
    path_lists.append(image_path)
    images.append(resized_image)

    labels.append(label)
    i+=1
