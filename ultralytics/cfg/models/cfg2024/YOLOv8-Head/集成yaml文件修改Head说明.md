##  Different use IoU Loss train 

 Now the integration is directly   Configure files in the network yaml Modify faster Head

 Just need the field  newhead Converted Head Terrate name ,  You can train 

 for example   Add to 
newhead: AsDDet
 Immediately indicate  AsDDet Detection head   train ， Very convenient 

 Can be newly added Head Detect head core file .py To ， No other operations do not need to be changed ， Very convenient ， The first project adopts this method 