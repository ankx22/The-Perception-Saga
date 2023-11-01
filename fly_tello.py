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
pear_drone.takeoff()
while True:
    frame = pear_drone.get_frame_read().frame
    frame = cv2.resize(frame, (360, 240))
    cv2.imshow('video', frame)
    key = cv2.waitKey(1) & 0xFF
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
        cv2.imwrite('picture.png', frame)
print(f"Landing ....")
pear_drone.land()
