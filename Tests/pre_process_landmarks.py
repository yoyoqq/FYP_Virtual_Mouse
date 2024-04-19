
import copy 
import itertools

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    print(temp_landmark_list)
    print()

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    print(temp_landmark_list)
    print()

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    # temp_landmark_list = temp_landmark_list[2:]
    # print(temp_landmark_list)
    return temp_landmark_list

a = pre_process_landmark([[662, 543], [575, 543], [511, 487], [716, 323], [717, 344], [717, 370]])
print(a)