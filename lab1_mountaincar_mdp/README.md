# Lab 1: Mountain Car V0 - Policy and Value iterations

### Policy iteration

Params:

`position_bins_count=18*4`   18 position bins mean bin size = 0.1

`velocity_bins_count=14*4`   14 velocity bins mean bin size = 0.01

`n_value_table_iter=50`

`gamma=0.8`

`max_value_updates_per_policy=50`

`eps=1e-6`

Reward used: `-1 + (next_position-current_position) ** n`, encourages movement.

Tried different powers (n), shown in the table.

|  Policy update iterations   |     training  reward power    |   Environment  testing reward  |  
|-----------------------------|-------------------------------|--------------------------------|
| 5                           | 0.5                           | -114.0                         |
| 5                           | 1                             | -112.0                         |
| 5                           | 2                             | -120.0                         |
| 5                           | 3                             | -121.0                         |
|-----------------------------|-------------------------------|--------------------------------|
| 10                          | 0.5                           | -158.0                         |
| 10                          | 1                             | -111.0                         |
| 10                          | 2                             | -116.0                         |
| 10                          | 3                             | -121.0                         |
|-----------------------------|-------------------------------|--------------------------------|
| 20                          | 0.5                           | -111.0                         |
| 20                          | 1                             | -111.0                         |
| 20                          | 2                             | -121.0                         |
| 20                          | 3                             | -121.0                         |



### Value iteration

Params:

`position_bins_count=18*4`   18 position bins mean bin size = 0.1

`velocity_bins_count=14*4`   14 velocity bins mean bin size = 0.01

`gamma=0.8`

`eps=1e-6`

Reward used: `-1 + (next_position-current_position) ** n`, encourages movement.


|         iterations          |     training  reward power    |   Environment  testing reward  |  
|-----------------------------|-------------------------------|--------------------------------|
| 10                          | 0.5                           | -113.0                         |
| 10                          | 1                             | -117.0                         |
| 10                          | 2                             | -111.0                         |
| 10                          | 3                             | -109.0                         |
|-----------------------------|-------------------------------|--------------------------------|
| 20                          | 0.5                           | -93.0                          |
| 20                          | 1                             | -117.0                         |
| 20                          | 2                             | -116.0                         |
| 20                          | 3                             | -85.0                          |
|-----------------------------|-------------------------------|--------------------------------|
| 50                          | 0.5                           | -86.0                          |
| 50                          | 1                             | -111.0                         |
| 50                          | 2                             | -110.0                         |
| 50                          | 3                             | -88.0                          |
