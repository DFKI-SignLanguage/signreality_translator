import xmlrpc.client

# Define the server URL
server_url = "http://127.0.1.1:8000/RPC2"
print(f"Contacting translation server at URL '{server_url}'...")

# Create an XML-RPC client
client = xmlrpc.client.ServerProxy(server_url)

# Define the method and parameters
method = "translate"
params = ["Der Zug ICE 234 aus Mainz fährt um 4:40 Uhr."]

# Make the call
try:
    result = client.__getattr__(method)(*params)
    print(f"Translation result (type {type(result)}):")
    print(result)
except xmlrpc.client.Fault as fault:
    print("XML-RPC Fault:", fault)
except Exception as e:
    print("Error:", e)

print("All done.")
