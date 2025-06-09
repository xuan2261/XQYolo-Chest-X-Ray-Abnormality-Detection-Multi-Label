 Use  ultralytics\cfg\models\cfg2024\YOLOv8-Loss\Focaler  When the content is under the file 

 Need to be ultralytics\utils\NewLoss\iouloss.py In the file 
 Code 
Focaler = False
 Change to 
Focaler = True
 To 


 After it is changed like this ， Call  python train_v8.py --cfg ultralytics\cfg\models\cfg2024\YOLOv8-Loss\YOLOv8-CIoU.yaml
 Just use Focaler-CIoU function ， other ultralytics\cfg\models\cfg2024\YOLOv8-Loss Other level functions in the directory （ like SIoU、DIoU、PIoU） Also empathy 