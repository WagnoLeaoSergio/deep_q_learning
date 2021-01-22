import cv2
import mss
import numpy
import time


def get_and_process_screen(screen=mss.mss().monitors[1]):
    # Get raw pixels from the screen, save it to a Numpy array
    img = numpy.array(sct.grab(monitor))
    img = cv2.resize(img, (700, 500))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    return img


if __name__ == "__main__":
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = sct.monitors[1]

        while "Screen capturing":
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))
            img = cv2.resize(img, (700, 500))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            # Display the picture
            cv2.imshow("OpenCV/Numpy normal", img)

            # print("fps: {}".format(1 / (time.time() - last_time)))

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break