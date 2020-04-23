users =  [

    {"id": 0, "name": "Zero" },
    {"id": 1, "name": "One"},
    {"id": 2, "name": "Two"},
    {"id": 3, "name": "Three"},
    {"id": 4, "name": "Four"},
    {"id": 5, "name": "Five"},
]

friendships: [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (5, 4)
]

for user in users:
    user["friends"] = []

for i, j in friendships:
    users[i]["friends"].append(users[j])
    users[j]["friends"].append(users[i])

def number_of_friends(user):
    return len(user["friends"])
