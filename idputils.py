import cv,cv2

def imshow(img,title=''):
	cv2.imshow(title,img)
	disp()
	
def disp():
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def red_rect(img,r1,c1,r2,c2,width=2):
	""" Draws a red rectangle on given 3 channel image """
	w=width
	#vertical lines
	img[r1:r2,c1:c1+w] = (0,0,255)
	img[r1:r2,c2:c2+w] = (0,0,255)
	#horizontal lines
	img[r1:r1+w,c1:c2] = (0,0,255)
	img[r2:r2+w,c1:c2] = (0,0,255)
