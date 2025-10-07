import RPi.GPIO as GPIO
import time

# GPIO pin numbers for 12 servos
servo_pins = [2,3,4,5,6,17,22,9,10,11,23,13]

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Create dictionary to store PWM objects
servo_pwms = {}

# Initialize each servo pin with PWM at 50Hz
for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, 50)
    pwm.start(0)
    servo_pwms[pin] = pwm

def set_angle(pin, angle):
    """Set servo on given GPIO pin to the specified angle (0–180)."""
    if 0 <= angle <= 180:
        duty = 2 + (angle / 18)
        servo_pwms[pin].ChangeDutyCycle(duty)
        time.sleep(0.5)
        servo_pwms[pin].ChangeDutyCycle(0)
    else:
        print("Angle must be between 0 and 180.")

try:
    while True:
        user_input = input("Enter servo number (0-11) and angle (0-180), e.g., '3 90', or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        try:
            servo_num, angle = map(int, user_input.strip().split())
            if 0 <= servo_num < 12:
                pin = servo_pins[servo_num]
                set_angle(pin, angle)
            else:
                print("Servo number must be between 0 and 11.")
        except ValueError:
            print("Invalid input format. Use: servo_number angle")

except KeyboardInterrupt:
    print("Program interrupted.")

finally:
    for pwm in servo_pwms.values():
        pwm.stop()
    GPIO.cleanup()
    print("GPIO cleaned up.")
