import cv2
import glob
import os
import pandas as pd
from scan_doc import *
from viewport_renderer import *
from template_maker import *
from data_extractor import *
from feature_based_transformer import *


class Project:
    def __init__(self, project_name_):
        self.project_name = project_name_
        self.path_input = 'projects/' + self.project_name +'/inputs/'
        self.path_scanned = 'projects/' + self.project_name +'/scanned/'
        self.path_form_template = 'projects/' + self.project_name +'/form_template/'
        self.setup()

    def setup(self):

        isExist = os.path.exists(self.path_input)
        if not isExist:
            os.makedirs(self.path_input)
        
        isExist = os.path.exists(self.path_scanned)
        if not isExist:
            os.makedirs(self.path_scanned)
        
        isExist = os.path.exists(self.path_form_template)
        if not isExist:
            os.makedirs(self.path_form_template)

#load an image of the form to make a template out of it.
def load_template(project):
    img_path = None
    for filename in os.listdir(project.path_form_template):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(project.path_form_template, filename)

    doc_template = cv2.imread(img_path)

    #doc_template = scan(doc_template)
    #cv2.imshow("template", doc_template)
    return doc_template

if __name__ == "__main__":

    project = Project("project2")
    view_port = ViewPort(600,750)
    img_template = load_template(project)
    #cropped_template = scan(img_template)
    canvas = view_port.load(img_template)
    template = Template(canvas, 2, (0,255,0))
    df = pd.DataFrame()
    extractor = DataExtractor(None)

    cv2.imshow("canvas", canvas)
    
    scale_factor_x = img_template.shape[1]/canvas.shape[1]
    scale_factor_y = img_template.shape[0]/canvas.shape[0]

    template.mark_fields("canvas")
    key = cv2.waitKey(0)
    if key & 0xFF == ord("p"):
        img = template.scale_points(img_template, scale_factor_x, scale_factor_y)
        cv2.imwrite("projects/project2/masks/template.jpg", img)
        print ("in main")
        print(template.points_holder.points)
        #extractor.mask_fields(template.points_holder, cropped_template)

    for filename in os.listdir(project.path_input):
        cv2.imwrite("projects/project2/aligned/reference.jpg", img_template)
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(project.path_input, filename)
            input_form = cv2.imread(img_path)
            #form_scan = scan(input_form)
            #aligned, h = alignImages(form_scan, cropped_template)
            aligned, h = alignImages(input_form, img_template)
            cv2.imwrite("projects/project2/aligned/" + filename, aligned)
            extractor.mask_fields(template.points_holder, aligned)

    num_columns = 3
    table_data = [extractor.values[i:i+num_columns] for i in range(0, len(extractor.values), num_columns)]

    # Create DataFrame from the table data
    #df = pd.DataFrame(table_data, columns=['name', 'reg no', 'gender', 'phone', 'guardian', 'street', 'city', 'pincode'])
    df = pd.DataFrame(table_data, columns=['name', 'reg no', 'gender'])
    # Print the DataFrame
    #print(df)

    df.to_csv('form_data.csv', index=False)



    

    cv2.waitKey(0)
    cv2.destroyAllWindows()