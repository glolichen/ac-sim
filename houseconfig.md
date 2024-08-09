# House Configuration File

The house can be configured by a JSON file, its format and requirements are described below. The JSON should have 2 key-value pairs, one named "floors" and one named "constants".

# Constants
The following constants must be defined:
 * "cooler_btu" in $\mathrm{BTU}$ (number) (BTUs of the cooler)
 * "heater_btu" in $\mathrm{BTU}$ (number) (BTUs of the heater)
 * "ext_wall_thick" in meters (number) (thickness of external walls) ()
 * "int_wall_thick" in meters (number) (thickness of internal walls)
 * "ext_wall_therm_cond" in $\mathrm{W/mK}$ (number) (thermal conductivity of external walls)
 * "int_wall_therm_cond" in $\mathrm{W/mK}$ (number) (thermal conductivity of internal walls)
 * "ext_roof_thick" in meters (number) (thickness of external roofs)
 * "int_roof_thick" in meters (number) (thickness of internal ceilings)
 * "ext_roof_therm_cond" in $\mathrm{W/mK}$ (number) (thermal conductivity of external roofs)
 * "int_roof_therm_cond" in $\mathrm{W/mK}$ (number) (thermal conductivity of internal ceilings)
 * "outside_convection" in $\mathrm{W/m^2K}$ (number) (convection coefficient outside the house)
 * "inside_convection" in $\mathrm{W/m^2K}$ (number) (convection coefficient inside the house)
 * "settings" (array of numbers) (what power the heater can be set to on the interval $[-1,1]$. Negative numbers indicate cooler power, positive numbers indicate heater power, $0$ is off)

# Floors
The value of "floors" should be an array. Each entry in the array corresponds to a floor in the house. Each entry in the array is an object. That object should have 2 key-value pairs: "height" and "rooms". "height" is the height of the floow in meters. All rooms in the floor will have the same height. "rooms" is an array of objects, each consisting of a single key-value pair with name "walls". "walls" is also an array of objects and is described below. The program will automatically determine if segments of the wall are internal or external.

## Walls
Each wall object should have 6 key-value pairs. They are:
 * "x0" (number)
 * "y0" (number)
 * "x1" (number)
 * "y1" (number)
 * "windows" (array)
 * "doors" (array)

The wall is a straight line between $(x_0, y_0)$ and $(x_1, y_1)$. Wall objects in the array must surround completely enclose an area. In addition, the following must be true.

$\left(x_0\neq x_1\text{ or }y_0\neq y_1\right)$ and $x_0,y_0,x_1,y_1>0$

"windows" is an array of objects. Each window object contains 3 key-value pairs: "start", "end", "height" and "open". The starting position of the window is $\mathrm{start}$ meters from $(x_0, y_0)$ and its ending position is $\mathrm{end}$ meters from $(x_0, y_0)$. Consequently, the width of the window is $\mathrm{end}-\mathrm{start}$ meters. "open" is a boolean value; true indicates the window is open and false indicates closed. The following conditions must be true:

$0<\mathrm{start}<\mathrm{end}<\sqrt{\left(x_0-x_1\right)^2+\left(y_0-y_1\right)^2}$ and $0<\text{window height}\leq\text{floor height}$

"doors" is also an array of objects with the same format as "windows"

https://help.iesve.com/ve2021/table_6_thermal_conductivity__specific_heat_capacity_and_density.htm#
