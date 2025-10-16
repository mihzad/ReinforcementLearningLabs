# Lab 2: Mountain Car - Monte Carlo Algorithm

Params:

`position_bins_count=18*4`   18 position bins mean bin size = 0.1

`velocity_bins_count=14*4`   14 velocity bins mean bin size = 0.01

`episodes=50000`

`gamma=0.99`

`max_states_per_episode=1000`

`eps=0.1`

Reward used: `-1 + (next_position-current_position)`, encourages movement.

Training result:

```console
Episode   500: avg reward -999.42 
Episode  1000: avg reward -999.27
Episode  1500: avg reward -997.84
Episode  2000: avg reward -1000.41
Episode  2500: avg reward -1000.55
Episode  3000: avg reward -1000.53
Episode  3500: avg reward -1000.61
Episode  4000: avg reward -997.12
Episode  4500: avg reward -1000.57
Episode  5000: avg reward -1000.42
Episode  5500: avg reward -999.96
Episode  6000: avg reward -1000.55
Episode  6500: avg reward -1000.61
Episode  7000: avg reward -1000.55
Episode  7500: avg reward -1000.54
Episode  8000: avg reward -1000.60
Episode  8500: avg reward -1000.45
Episode  9000: avg reward -1000.24
Episode  9500: avg reward -999.71
Episode 10000: avg reward -1000.53
Episode 10500: avg reward -997.81
Episode 11000: avg reward -1000.65
Episode 11500: avg reward -1000.65
Episode 12000: avg reward -999.82
Episode 12500: avg reward -999.89
Episode 13000: avg reward -1000.56
Episode 13500: avg reward -608.00
Episode 14000: avg reward -345.48
Episode 14500: avg reward -295.53
Episode 15000: avg reward -294.67
Episode 15500: avg reward -297.63
Episode 16000: avg reward -276.62
Episode 16500: avg reward -259.59
Episode 17000: avg reward -262.93
Episode 17500: avg reward -262.69
Episode 18000: avg reward -259.57
Episode 18500: avg reward -263.01
Episode 19000: avg reward -261.75
Episode 19500: avg reward -246.47
Episode 20000: avg reward -237.24
Episode 20500: avg reward -234.86
Episode 21000: avg reward -237.69
Episode 21500: avg reward -233.80
Episode 22000: avg reward -235.48
Episode 22500: avg reward -236.00
Episode 23000: avg reward -235.78
Episode 23500: avg reward -230.61
Episode 24000: avg reward -236.20
Episode 24500: avg reward -234.62
Episode 25000: avg reward -228.42
Episode 25500: avg reward -226.56
Episode 26000: avg reward -229.63
Episode 26500: avg reward -229.03
Episode 27000: avg reward -230.09
Episode 27500: avg reward -228.38
Episode 28000: avg reward -229.44
Episode 28500: avg reward -232.93
Episode 29000: avg reward -224.40
Episode 29500: avg reward -218.51
Episode 30000: avg reward -221.32
Episode 30500: avg reward -223.86
Episode 31000: avg reward -225.37
Episode 31500: avg reward -219.49
Episode 32000: avg reward -223.36
Episode 32500: avg reward -219.67
Episode 33000: avg reward -223.44
Episode 33500: avg reward -223.13
Episode 34000: avg reward -218.89
Episode 34500: avg reward -223.47
Episode 35000: avg reward -223.88
Episode 35500: avg reward -221.97
Episode 36000: avg reward -218.36
Episode 36500: avg reward -208.50
Episode 37000: avg reward -201.90
Episode 37500: avg reward -201.04
Episode 38000: avg reward -202.08
Episode 38500: avg reward -204.69
Episode 39000: avg reward -207.52
Episode 39500: avg reward -204.52
Episode 40000: avg reward -203.06
Episode 40500: avg reward -203.51
Episode 41000: avg reward -203.46
Episode 41500: avg reward -206.50
Episode 42000: avg reward -202.74
Episode 42500: avg reward -201.97
Episode 43000: avg reward -203.42
Episode 43500: avg reward -204.70
Episode 44000: avg reward -203.19
Episode 44500: avg reward -203.50
Episode 45000: avg reward -202.82
Episode 45500: avg reward -204.90
Episode 46000: avg reward -205.67
Episode 46500: avg reward -207.14
Episode 47000: avg reward -202.94
Episode 47500: avg reward -204.44
Episode 48000: avg reward -203.39
Episode 48500: avg reward -200.63
Episode 49000: avg reward -202.99
Episode 49500: avg reward -203.88
Episode 50000: avg reward -202.20

Training finished. Time spent: 4325.01970911026 seconds.
Actual Environment test reward: -130.0
```
We can see pretty good results, which are kind of similar compared to Policy and Value iterations. And we dont need any model details.
Yet it took very long time for them to be made - 4305s. For example, Policy Iteration took an order of magnitude less time, let alone the Value Iteration, which got done in less than a minute.
