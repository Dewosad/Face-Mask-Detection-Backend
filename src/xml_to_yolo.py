import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_folder, txt_folder, class_list):
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith(".xml"):
            continue

        #parsing xml
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)

        yolo_annotations = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_id = class_list.index(class_name)

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # Save annotations to a .txt file
        txt_file = os.path.join(txt_folder, xml_file.replace(".xml", ".txt"))
        with open(txt_file, "w") as f:
            f.write("\n".join(yolo_annotations))

if __name__ == "__main__":
    class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']
    # Convert training annotations
    convert_voc_to_yolo("../annotations", "../labels/train", class_names)
    # Convert validation annotations
    convert_voc_to_yolo("./annotations", "./labels/val", class_names)
    print("XML to YOLO format conversion completed.")
