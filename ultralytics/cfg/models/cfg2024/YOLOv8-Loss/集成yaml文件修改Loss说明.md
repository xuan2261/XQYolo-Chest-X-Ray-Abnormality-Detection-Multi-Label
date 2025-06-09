##  Different use IoU Loss train 

 Now the integration is directly   Configure files in the network yaml Modify loss

 Just need the field   Converted IoU Loss function name ,  You can train 

 for example   Add to 
loss：SIoU
 Immediately indicate SIoU Loss function training ， Very convenient ， The first project adopts this method 