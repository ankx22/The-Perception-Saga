from djitellopy import Tello
import cv2
import time

pear_drone = Tello()
pear_drone.connect()
battery_status = pear_drone.get_battery()
print(f"Battery Status: {battery_status}")

# pear_drone.takeoff()
# time.sleep(2)
# pear_drone.move_forward(30)
# time.sleep(2)
# pear_drone.move_right(30)
# time.sleep(2)
# pear_drone.move_back(30)
# time.sleep(2)
# pear_drone.move_left(30)
# time.sleep(2)
# pear_drone.land()

pear_drone.streamon()
# pear_drone.takeoff()
i = 0
while True:
    frame = pear_drone.get_frame_read().frame
    # frame = cv2.resize(frame, (360, 240)) # Windows
    # frame = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_LINEAR) # Ubuntu
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('video', frame)
    key = cv2.waitKey(2) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        pear_drone.move_forward(30)
    elif key == ord('s'):
        pear_drone.move_back(30)
    elif key == ord('a'):
        pear_drone.move_left(30)
    elif key == ord('d'):
        pear_drone.move_right(30)
    elif key == ord('p'):
        print(f"Taking picture....\n")
        filename = f"frame{i}"
        cv2.imwrite(filename, frame)
    i += 1
# print(f"Landing....")
# pear_drone.land()
