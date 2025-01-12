class Pointer():
    def __init__(self, number_of_circles, pointer_delay = 1) -> None:
        self.center = [width//2, height//2]
        self.circle_radius = self.create_circles(number_of_circles)
        self.mouse_speed = self.__create_mouse_speed(number_of_circles)
        self.pointer_delay = [0, pointer_delay] # current delay, total delay
    
    def move_pointer_delay(self) -> bool:
        if self.pointer_delay[0] == self.pointer_delay[1]:
            self.pointer_delay[0] = 0
            return True
        self.pointer_delay[0] += 1
        return False
        

    def move_pointer(self, index_finger_tip_x, index_finger_tip_y, angle):
        center_x, center_y = self.center
        # Calculate distance from the center
        distance = math.sqrt((center_x - index_finger_tip_x) ** 2 + (center_y - index_finger_tip_y) ** 2)
        # Determine speed based on the zone
        speed = 0
        for i in range(len(self.circle_radius)):
            if distance <= self.circle_radius[i]:
                speed = self.mouse_speed[i]
                break
        if distance > self.circle_radius[len(self.circle_radius)-1]:
            speed = self.mouse_speed[len(self.mouse_speed)-1]
    
        # Convert angle to radians and calculate movement deltas
        angle_radians = math.radians(angle)
        delta_x = speed * math.cos(angle_radians)
        delta_y = -speed * math.sin(angle_radians)  # Screen coordinates: y increases downwards
        # Move the mouse pointer by the calculated deltas
        # ? changed for the delay 
        # pyautogui.move(delta_x, delta_y)
        pyautogui.move(delta_x, delta_y, duration=0.1)
    
    def draw_triangle_and_show_angle(self, image, fingertip):
        color=(255, 0, 0)
        # Calculate the angle
        angle_degrees_normalized = self.calculate_angle_with_respect_to_midpoint(fingertip[0], fingertip[1])
        # Choose a third point for the triangle (e.g., directly below the midpoint for simplicity)
        third_point = (self.center[0], self.center[1] + 100)
        # Draw lines to form the triangle
        cv.line(image, self.center, fingertip, color, 2)
        cv.line(image, fingertip, third_point, color, 2)
        cv.line(image, third_point, self.center, color, 2)
        # Display the angle
        text_position = (self.center[0] - 10, self.center[1] - 10)  # Slightly above and to the left of the midpoint
        cv.putText(image, "Angle: %s" %(angle_degrees_normalized), text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image
    
    
    def calculate_angle_with_respect_to_midpoint(self, x, y):
        dx = x - self.center[0]
        dy = y - self.center[1]
        # Calculate the angle in radians
        angle_radians = math.atan2(dy, dx)
        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)
        angle_degrees = 360 - angle_degrees
        # Normalize the angle to be between 0 and 360
        angle_degrees_normalized = angle_degrees % 360
        return int(angle_degrees_normalized)
           
    def create_circles(self, number_of_circles):
        circles_diameter = [self.calculate_circle_dimension(0.05)]
        prev = 0.05
        for i in range(number_of_circles):
            circles_diameter.append(self.calculate_circle_dimension(prev + circle_increment))
            prev += circle_increment
        return circles_diameter
        
    def calculate_circle_dimension(self, fraction):
        # radius = int(height * (1-fraction))
        radius = int(height * fraction)
        return radius

    def __create_mouse_speed(self, number_of_circles):
        speed = [0]
        for i in range(number_of_circles+1):
            speed.append(speed[-1] + mouse_increments)
        speed[-1] = 100
        return speed 
    