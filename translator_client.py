import xmlrpc.client
import sys

# Define the server URL
SERVER_URL = "http://127.0.1.1:8000/RPC2"
# Define the method and parameters
METHOD = "translate"

if __name__ == '__main__':
    # Create an XML-RPC client
    client = xmlrpc.client.ServerProxy(SERVER_URL)
    try:
        params = [sys.argv[1]]
    except IndexError:
        params = ["Der Zug ICE 234 aus Mainz fährt um 4:40 Uhr."]

    # Make the call
    try:
        result = client.__getattr__(METHOD)(*params)
        print("Translation result:", result)
    except xmlrpc.client.Fault as fault:
        print("XML-RPC Fault:", fault)
    except Exception as e:
        print("Error:", e)