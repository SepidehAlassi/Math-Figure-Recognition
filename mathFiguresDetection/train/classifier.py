from lxml import etree
import sys,os
cwd=os.getcwd()
def classifier(annotationDir, imagesDir, outputFile):

    for file in os.listdir(annotationDir):
        if file.endswith(".xml"):
            tree = etree.parse(os.path.join(annotationDir, file))
            root = tree.getroot()
            root.getchildren()
            filename=root.xpath("filename")[0].text
            filename=filename.replace('.tiff', '.jpg')
            filename = filename.replace('.tif', '.jpg')
            classifier_file = os.path.join(imagesDir, filename)

            objects = root.xpath("object")
            for obj in objects:
                line=classifier_file+','
                bndBox = obj.xpath('bndbox')[0]
                for child in bndBox.getchildren():
                    if child.tag=='xmin':
                        x1=child.text
                    elif child.tag=='xmax':
                        x2=child.text
                    elif child.tag=='ymin':
                        y1=child.text
                    else:
                        y2=child.text
                line+=x1+','+y1+','+x2+','+y2+','
                name = obj.xpath('name')[0].text
                if name=='.':
                    name='dot'
                elif name=='::':
                    name='double_colon'
                elif name=='(':
                    name='left_paranthesis'
                elif name==')':
                    name='right_paranthesis'
                elif name=="=":
                    name="equal"
                elif name==',':
                    name='comma'
                elif name==':':
                    name='colon'
                elif name == '-':
                    name='minus'
                elif name=='+':
                    name='plus'
                else:
                    pass

                line += name+'\n'
                outputFile.write(line)




if __name__ == '__main__':
     args = sys.argv
     annotationDir, imagesDir= args[1], args[2]
     outputFile=open(os.path.join(cwd,'train','annotations.csv'), 'w')
     classifier(annotationDir, imagesDir, outputFile)
     outputFile.close()
