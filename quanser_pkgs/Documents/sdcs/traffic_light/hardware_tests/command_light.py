#Traffic Light Client
#Dev: Peter Martin
#Last Update: April 1, 2022
#Summary: This script calls the traffic light API using the following syntax - CommandLight.py [auto/immediate/timed/status/shutdown] [R] [Y] [G]
#e.g. 
# > CommandLight.py auto 
# will set the timing to the default of green for 30s, yellow for 2s, red for 30s

# > CommandLight.py status
# returns the status of the light where 1 = red, 2 = yellow, 3 = green

# > CommandLight.py immediate 0 1 0
# stops any other task and sets the yellow light to ON

# > CommandLight.py timing 10 3 10
# sets the timing to 10 3 10 and kicks off a thread to run the timing 

import urllib.request, sys
from urllib.error import HTTPError, URLError
from socket import timeout

#Format the arguments into the proper request for the endpoint
def formatreq(params):

    request = ""
    command = params[0]

# if no parameters are expected and the received command is the same as the request
    if command == 'status' or command == 'shutdown':
        request = command
# if automatic mode is requested then switch to timed mode with r/y/g set to 0 to flag the default timing
    elif command == 'auto':
        request = "timed/0/0/0"
# if an immediate light change is requested, then parse out the parameter that is != 0 as the one to turn on
    elif command == 'immediate':
        if int(params[1]) > 0:
            request = "immediate/red"
        elif int(params[2]) > 0:
            request = "immediate/yellow"
        elif int(params[3]) > 0:
            request = "immediate/green"
        else:
            request = "immediate/off"
# if a custom timing is requested, then pull the timings into the URL
    elif command == 'timed':
        request = "timed/" + params[1] + "/" + params[2] + "/" + params[3]
    
    return request

#Send the formatted request
def sendreq(url):
    #Format the HTTP get request with a timeout of 1s to account for async tasks that will not return

    response = "Call complete!"
    try:
        response = urllib.request.urlopen(url, timeout=1).read().decode('utf-8')
    #If the URL is not correct
    except (HTTPError, URLError) as error:
        response = "Error endpoint not found at " + url
    #If the request was not expected to return the call it complete, otherwise flag a timeout
    except timeout:
        if url.find("timed") == -1:
            response = "Call timed out"
        else:
            response = "Async call complete"

    return response

if __name__ == "__main__":

#sample call to the methods

    url = "http://" + sys.argv[1] + ":5000/"
    request = formatreq(sys.argv[2:])
    print(sendreq(url + request))
