# Interval DP Split Contract

## Convention: right-start

`combine(i, mid, j)` computes the cost of combining two solved
subproblems into the interval [i, j]:

    left subproblem:  [i, mid-1]
    right subproblem: [mid, j]
    combination cost: combine(i, mid, j)

`mid` is the **start index of the right half**.  Range: i+1 ≤ mid ≤ j.

`leaf(i)` is the base cost of a single-element interval [i, i].

## In the DP loop

```
for k in [i, j):          // k = last index of left half
    total = dp[i,k] + dp[k+1,j] + cost.combine(i, k+1, j)
                                              ^^^
                                              mid = k+1 = right-start
```

## In the candidate

`candidate.split(i, j)` returns k = **last index of the left half**.
To recover mid (right-start): `mid = split(i,j) + 1`.

## Matrix chain example

Matrices M_0 .. M_{n-1}, dimension array dims[0..n].
Matrix M_i has dimensions dims[i] × dims[i+1].

```
combine(i, mid, j) = dims[i] × dims[mid] × dims[j+1]
```

Because: left result is dims[i] × dims[mid], right result is
dims[mid] × dims[j+1], matrix multiply = dims[i] × dims[mid] × dims[j+1].
