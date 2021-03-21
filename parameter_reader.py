import json 

def read_parameter():
    # Opening JSON file 
    f = open('parameter.json',) 
    
    # returns JSON object as  
    # a dictionary 
    parameter = json.load(f) 
    # Closing file 
    f.close() 

    return parameter

if __name__ == "__main__":
    print(read_parameter())
    
    