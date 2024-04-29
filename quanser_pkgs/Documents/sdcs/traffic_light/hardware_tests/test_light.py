import os

#Get the current status of the light (Red = 1, Yellow = 2, Green = 3)
# os.system("CommandLight.py 192.168.2.11 status")

#Set the light timing to automatic (Green 30s, Yellow 3s, Red 30s)
os.system("CommandLight.py 192.168.2.11 auto")

#Command the light to be a particular colour (Red Yellow Green)
#os.system("CommandLight.py 192.168.2.11 immediate 0 1 0") #Turn on the Yellow light

#Set custom timing for the Red, Yellow, and Green lights in seconds
#os.system("CommandLight.py 192.168.2.11 timed 10 3 10") #Turn on the Red for 10s, Yellow for 3s, Green for 10s

#Shutdown the server
#os.system("CommandLight.py 192.168.2.11 shutdown")