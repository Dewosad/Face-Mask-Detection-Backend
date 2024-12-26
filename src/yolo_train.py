from ultralytics import YOLO 

def train_yolo(data_path, weights_path, epochs=30, img_size=640):
    # Load the YOLOv8 model
    model = YOLO("././models/yolov8n.pt") 

    # Train the model
    model.train(
        data=data_path,      
        epochs=epochs,       
        imgsz=img_size,      
        project="./models",  
        name="yolov8_mask"   
    )

    # Save the trained weights
    model.save(weights_path)
    print(f"YOLOv8 training complete! Weights saved at {weights_path}")

if __name__ == "__main__":
    # Define paths
    data_yaml_path = "data.yaml"          
    output_weights_path = "./models/yolov8_mask4"  #

    # Call the training function
    train_yolo(data_yaml_path, output_weights_path)
