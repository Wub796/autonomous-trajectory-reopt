from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel

t = Time('2027-02-19')
mars_pos, mars_vel = get_body_barycentric_posvel('mars', t)
earth_pos, earth_vel = get_body_barycentric_posvel('earth', t)
print("Mars: ",mars_pos, mars_vel)
print("Earth: ", earth_pos, earth_vel)