import json
import housebuilder
with open('2r_simple.json', 'r') as file:
    data = json.load(file)
    height = data['floors'][0]['height']
    something = housebuilder.build_house("2r_simple.json")
    print(something)
    first_room_volume = something.rooms[0][0].get_volume()
    second_room_volume = something.rooms[0][1].get_volume()
    print(first_room_volume, second_room_volume)
    
