"""
ephemeris.py — quick CLI helper to print Earth & Mars barycentric state vectors.

Usage:
    python -m src.utils.ephemeris            # uses default date
    python -m src.utils.ephemeris 2027-02-19 # pass a custom date
"""
import sys
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel


def query(date: str = '2027-02-19') -> None:
    t = Time(date)
    mars_pos, mars_vel = get_body_barycentric_posvel('mars', t)
    earth_pos, earth_vel = get_body_barycentric_posvel('earth', t)
    print(f"Date   : {date}")
    print(f"Mars   pos={mars_pos}  vel={mars_vel}")
    print(f"Earth  pos={earth_pos}  vel={earth_vel}")


if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else '2027-02-19'
    query(date)
