import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import cv2

class DataExtractor:
    def __init__(self, df_):
        self.df = df_
        self.values = []
        self.count = 0

    def mask_fields(self, points_holder, img):
        
        print(points_holder.points)
        for point in points_holder.points:
            x1 = point[0][0]
            x2 = point[1][0]
            y1 = point[0][1]
            y2 = point[1][1]
            if x1>x2:
                x1 = point[1][0]
                x2 = point[0][0]
            if y1>y2:
                y1 = point[1][1]
                y2 = point[0][1]
            section = img[y1:y2, x1:x2]
            cv2.imwrite("projects/project2/masks/" + str(self.count) + ".jpg", section )
            data = self.ocr(section)
            self.values.append(data)
            self.count +=1


    def ocr(selef, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Perform OCR using Tesseract
        data = pytesseract.image_to_string(gray)
        return data

    def export_to_csv(self):
        pass

    def export_to_excel(self):
        pass