#!/usr/bin/env python3
"""
Continuous (360-degree) servo test via PCA9685 on Raspberry Pi 5.

This version drives raw pulse width directly so neutral can be calibrated
precisely (required for many continuous servos).

Controls:
  UP/RIGHT or w/d  : increase CW speed
  DOWN/LEFT or x/a : increase CCW speed
  SPACE or s       : stop (speed=0)
  r                : reverse current direction
  [ / ]            : neutral pulse -/+ (us)
  - / =            : speed range -/+ (us)
  p                : print current tuning
  q                : quit
"""

import argparse
import select
import sys
import termios
import tty

SERVO_FREQ = 50
DEFAULT_MIN_PULSE_US = 1000
DEFAULT_MAX_PULSE_US = 2000
DEFAULT_NEUTRAL_US = 1500
DEFAULT_SPEED_RANGE_US = 250


def clamp(value, low, high):
    return max(low, min(high, value))


def pulse_us_to_duty_cycle(pulse_us: float, freq_hz: int) -> int:
    period_us = 1_000_000.0 / freq_hz
    duty = int((pulse_us / period_us) * 65535)
    return clamp(duty, 0, 65535)


def read_key(fd: int) -> str:
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch

    seq = ch
    while select.select([fd], [], [], 0.005)[0]:
        seq += sys.stdin.read(1)
        if len(seq) >= 3:
            break
    return seq


def render(speed: float, neutral_us: int, speed_range_us: int, pulse_us: int, status: str = "") -> None:
    bar_len = 40
    filled = int((speed + 1.0) * bar_len / 2.0)
    bar = "-" * filled + "|" + "-" * (bar_len - filled)

    print("\r\033[2K", end="")
    print(
        f"Speed={speed:+.2f}  Pulse={pulse_us}us  Neutral={neutral_us}us  Range={speed_range_us}us  {status}"
    )
    print(f"CCW [-1.0] {bar} [+1.0] CW")
    print("\033[F\033[F", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous servo test with neutral calibration (PCA9685)")
    parser.add_argument("--channel", type=int, default=9, help="PCA9685 channel (default: 9)")
    parser.add_argument("--step", type=float, default=0.08, help="Speed step per key press (default: 0.08)")
    parser.add_argument("--min-pulse-us", type=int, default=DEFAULT_MIN_PULSE_US, help="Minimum pulse in us")
    parser.add_argument("--max-pulse-us", type=int, default=DEFAULT_MAX_PULSE_US, help="Maximum pulse in us")
    parser.add_argument("--neutral-us", type=int, default=DEFAULT_NEUTRAL_US, help="Neutral pulse in us")
    parser.add_argument("--speed-range-us", type=int, default=DEFAULT_SPEED_RANGE_US, help="Pulse deviation at full speed")
    parser.add_argument("--neutral-step-us", type=int, default=2, help="Neutral adjust step in us for [ and ]")
    args = parser.parse_args()

    if not (0 <= args.channel <= 15):
        print("ERROR: --channel must be 0..15")
        sys.exit(1)

    speed = 0.0
    speed_step = clamp(args.step, 0.01, 1.0)
    min_pulse_us = args.min_pulse_us
    max_pulse_us = args.max_pulse_us
    neutral_us = clamp(args.neutral_us, min_pulse_us, max_pulse_us)
    speed_range_us = clamp(args.speed_range_us, 50, 700)
    neutral_step_us = clamp(args.neutral_step_us, 1, 20)

    try:
        import board
        import busio
        from adafruit_pca9685 import PCA9685
    except Exception as exc:
        print(f"ERROR: Missing dependency: {exc}")
        print("Install: pip install adafruit-circuitpython-pca9685")
        sys.exit(1)

    pca = None
    channel = None

    try:
        i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
        pca = PCA9685(i2c)
        pca.frequency = SERVO_FREQ
        channel = pca.channels[args.channel]

        pulse_us = neutral_us
        channel.duty_cycle = pulse_us_to_duty_cycle(pulse_us, SERVO_FREQ)

        print("=" * 64)
        print("Continuous Servo Test (Raw Pulse PWM via PCA9685 / Raspberry Pi 5)")
        print(f"Channel={args.channel}  Step={speed_step:.2f}")
        print(f"min/max={min_pulse_us}/{max_pulse_us}us  neutral={neutral_us}us  range={speed_range_us}us")
        print("=" * 64)
        print("Controls:")
        print("  UP/RIGHT or w/d  : more CW")
        print("  DOWN/LEFT or x/a : more CCW")
        print("  SPACE or s       : stop (speed=0)")
        print("  [ / ]            : neutral -/+")
        print("  - / =            : speed range -/+")
        print("  r                : reverse")
        print("  p                : print tuning")
        print("  q                : quit")
        print("")
        print("If spinning at stop: press SPACE, then tune [ and ] until it fully stops.")
        print("")
        print("Press keys now...")
        print("")
        print("")

        if not sys.stdin.isatty():
            raise RuntimeError("This script needs an interactive terminal (TTY).")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            render(speed, neutral_us, speed_range_us, pulse_us, "Ready")

            while True:
                key = read_key(fd)

                if key in ("q", "Q", "\x03"):
                    break
                if key in ("\x1b[A", "\x1b[C", "w", "W", "d", "D"):
                    speed = clamp(speed + speed_step, -1.0, 1.0)
                elif key in ("\x1b[B", "\x1b[D", "x", "X", "a", "A"):
                    speed = clamp(speed - speed_step, -1.0, 1.0)
                elif key in ("s", "S", " "):
                    speed = 0.0
                elif key in ("r", "R"):
                    speed = -speed
                elif key == "[":
                    neutral_us = clamp(neutral_us - neutral_step_us, min_pulse_us, max_pulse_us)
                elif key == "]":
                    neutral_us = clamp(neutral_us + neutral_step_us, min_pulse_us, max_pulse_us)
                elif key == "-":
                    speed_range_us = clamp(speed_range_us - 5, 50, 700)
                elif key == "=":
                    speed_range_us = clamp(speed_range_us + 5, 50, 700)
                elif key in ("p", "P"):
                    pass
                else:
                    continue

                pulse_us = int(clamp(neutral_us + speed * speed_range_us, min_pulse_us, max_pulse_us))
                channel.duty_cycle = pulse_us_to_duty_cycle(pulse_us, SERVO_FREQ)
                render(speed, neutral_us, speed_range_us, pulse_us, "Applied")

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print("\n")

    except Exception as exc:
        print(f"ERROR: {exc}")
    finally:
        if channel is not None:
            try:
                # Hold neutral on exit to avoid runaway movement.
                channel.duty_cycle = pulse_us_to_duty_cycle(neutral_us, SERVO_FREQ)
            except Exception:
                pass
        if pca is not None:
            try:
                pca.deinit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
