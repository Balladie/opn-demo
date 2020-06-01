import os

#for i in range(10,50):
#    os.remove("%05d.jpg" % i)
#    os.remove("%05d.png" % i)
for i in range(0,380):
    os.system("convert -resize 1920X1080 %05d.jpg %05d.jpg" % (i,i))
    os.system("convert -resize 1920X1080 %05d.png %05d.png" % (i,i))
