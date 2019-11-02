import os
import glob
from PIL import Image
import os.path
file = r'images\*.jpg'
saveDir = r'output' 
if __name__=="__main__":
    
    for jpgfile in glob.glob(file):
        img=Image.open(jpgfile).convert('RGB')
        new_img=img.resize((750,650),Image.BILINEAR) 
        new_img.save(os.path.join(saveDir,os.path.basename(jpgfile)))
    print('a')
