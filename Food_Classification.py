from fastai.vision.all import * 

foodPath = r"C:\Users\DELL\Desktop\Astitva\food-101"    #C:/Users/DELL/Desktop/Astitva

labelA = 'samosa'
labelB = 'hot_and_sour_soup'

for img in get_image_files(foodPath):
  
  if labelA in str(img):
    img.rename(f"{img.parent}/{labelA}-{img.name}")
  elif labelB in str(img):
    img.rename(f"{img.parent}/{labelB}-{img.name}")
  else: os.remove(img) 

len(get_image_files(foodPath))

def GetLabel(fileName):
  return fileName.split('-')[0]


dls = ImageDataLoaders.from_name_func(
    foodPath, get_image_files(foodPath), valid_pct=0.2, seed=420,
    label_func=GetLabel, item_tfms=Resize(224))

dls.train.show_batch()

learn = cnn_learner(dls, resnet34, metrics=error_rate, pretrained=True)
learn.fine_tune(epochs=10)

from google.colab import files
uploader = files.upload()

for img in uploader.items():
  uploadedImg = img[0]

img = PILImage.create(uploadedImg)
img.show()

label,_,probs = learn.predict(img)

print(f"This is a {label}.")
print(f"{labelA} {probs[1].item():.6f}")
print(f"{labelB} {probs[0].item():.6f}")

#---------------------- Test using dataset images ----------------------

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
SA
# for i in range(0,10):
#   #Load random image
#   randomIndex = random.randint(0, len(get_image_files(foodPath))-1)
#   img = mpimg.imread(get_image_files(foodPath)[randomIndex])
#   #Put into Model
#   label,_,probs = learn.predict(img)

#   #Create Figure using Matplotlib
#   fig = plt.figure()
#   ax = fig.add_subplot() #Add Subplot (For multiple images)
#   imgplot = plt.imshow(img) #Add Image into Plot
#   ax.set_title(label) #Set Headline to predicted label

#   #Hide numbers on axes
#   plt.gca().axes.get_yaxis().set_visible(False)
#    plt.gca().axes.get_xaxis().set_visible(False)