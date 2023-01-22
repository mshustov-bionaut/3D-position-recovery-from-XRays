import XRaysBotPosition as XRP
import numpy as np

xr_pos = XRP.XRays3DRecovery()

real_coord1 = [(25,60,0),(55,18,0),(210,42,69),(210,17,108)]
screen_coord1 = [(236,784),(436,1074),(1584,917),(1669,1036)]
xr_pos.projection_calibration(1,real_coord1,screen_coord1)

real_coord2 = [(25,60,0),(55,18,0),(210,42,69),(210,17,108)]
screen_coord2 = [(469,682),(582,934),(1459,856),(1628,983)]
xr_pos.projection_calibration(2,real_coord2,screen_coord2)

res = xr_pos.bot_position((1127,788),(1187,765))
print(res)

xr_pos.load_stl()