"""
A linear program to jointly optimize investment decision for a group of people given constraints. Each group have different constraints!
Structure:
    There are two people: m and p (we can call them mom and pop). Both and m and p are seniors. There is a third person n (non-senior).
    There are two investment options: i and g (we can call it interest-bearing and growth) available to each person.
    The horizon is Y years. Currently, we are in year 0. Time (year) is indexed by y.
    The objective is to minimize lifetime combined tax liability. Future tax liabilities can be discounted by a factor, rho.
    Cash inflow or income sources are:
        1. rental income from a shop (SR) and a home (HR). HR for (m and p) comes from BS of n (BS stands for basic salary).
        2. interest income I^i comes from interest-bearing debt
    Cash outflow or expenses are:
        1. living expense made equal to SR and period-0 I^i. Any increases in expenses over time will be met by increases in SR
    Investment options:
        1. i: interest-bearing debt: this generates a cashflow of I^i
        2. g: growth investment: this does not generate cashflow but grows the corpus
        3. The rate of growth is different for (i,g) and (m,p,n) and also varies over time. Captured by r.
    Taxation:
        1. total income includes SR, HR, I^i, I^g
        2. deductions include: fixed deductions (D^F), discretionary deductions (D), and 30% of rental income
        3. A function, G computes tax liability given total income minus deductions
    Disposable income:
        1. Income net of expenses and taxes. Fixed deductions are also available for investment.
    Choice Variables:
        1. A static fraction, f, which decides in what proportion to share HR among m and p
        2. A fraction, d, which decides how disposable income is split between i and g
        3. Fixed deductions, D^F
        4. all deductions (fixed and discretionary), D
    Constraints:
        1. 0 <= f < 1
        2. -0.5 <= d < 1.5 (negative value allows reducing investment in one option to increase investment in another)
        3. 0 <= D^F <= 1e5
        4. 0 <= D - D^F <= 2e5 (for m) and 2.5e5 (for p)
            4a. 0 <= D <= 3e5 (for m) and 3.5e5 (for p)
            4b. D >= D^F (for both m and p)
            4c. D <= 2e5 + D^F (for m) and 2.5e5 + D^F (for p)
        5. HR[m] = f * 0.5 * BS
        6. HR[p] = (1-f) * 0.5 * BS
        7. SR[p] = 0 (all SR is kept by m)
        8. total income >= 0
        9. disposable income >= 0
    Recurrence Relations:
        1. I^i[y+1] = r^i[y] * A^i[y] (A is wealth)
        2. I^g[y+1] = r^g[y] * A^g[y]
        3. A^i[y+1] = A^i[y] + F^i[y] (F^i is new investment in i)
        4. A^g[y+1] = A^g[y] + F^g[y] + I^g[y] (F^i is new investment in i, I^g is interest from g)
        5. T[y] = SR[y] + HR[y] + I^i[y] + I^g[y] - D[y] - 0.3*(HR[y] + SR[y]) (T is total taxable income)
        6. t[y] = G(T[y]) (t is tax liability)
        7. K[y] = HR[y] + D^F[y] - D[y] + (I^i[y] - I^i[0]) - t[y] (K is disposable income)
        8. F^g[y] = d[y] * K[y] (F^g is new investment in g)
        9. F^i[y] = (1-d[y]) * K[y] (F^i is new investment in i)
        10. SR[y+1] = SR[y] + 0.1 * SR[y] (SR grows at 10%)
        11. BS[y+1] = BS[y] + 0.1 * BS[y] (BS grows at 10%)
    Constants and Initial Values:
        0. We can't have d as a choice variable in an LP framework. We can use a constant d instead!
        1. Y = 3 (horizon, y goes from 0 to 3)
        2. SR[0] = 1.74e5 (initial shop rental income)
        3. BS[0] = 21.12e5 (initial BS)
        4. A^i_m[0] = 20e5 (m's initial wealth in i)
        5. A^g_m[0] = 11.595e5 (m's initial wealth in g)
        6. A^i_p[0] = 17e5 (p's initial wealth in i)
        7. A^g_p[0] = 2.25e5 (p's initial wealth in g)
        8.  r^i_m = [0.08887] * 4 (interest rate for m in options i for period 0, 1, 2, 3)
        9.  r^g_m = [0.09844] * 4
        10. r^i_p = [0.08626] * 4
        11. r^g_p = [0.07978] * 4
        12. d = [1] * 4 (constant d)
"""

import itertools
import pulp
import string


# define constants/initial values
VERY_LARGE_NUM = 1e8
VERY_SMALL_NUM = 1e-3
# horizon (extra 1 since python starts from 0)
Y = 10+1
max_HR_fraction = 0.5 # maximum fraction of BS that can be given as HR income
SeniorPersons = ['m', 'p']
Persons = SeniorPersons + ['n']
Investments = ['i', 'g']
Person_Investments = list(itertools.product(Persons, Investments))
SAL_c = {y: round(30e5 * (1 + 0.1) ** y, -2) for y in range(Y)} # 10% growth in salary
BS_c = {y: 0.4*SAL_c[y] for y in range(Y)} # 40% of salary as BS
SR_c = {y: round(1.74e5 * (1 + 0.1) ** y, -2) for y in range(Y)} # assume 10% growth in shop rental income
DF_lim = {'m': 0.5e5, 'p': 0.5e5, 'n': 0.5e5}  # fixed deductions limit (0.5e5 TTA for m/p and 0.5e5 std_ded for n)
D_lim = {'m': 3e5, 'p': 3e5, 'n': 2.75e5}  # total deductions limit (fixed dec + 2e5 80C/CCD-1B, 0.25e5/0.5e5 80D)
DD_lim = {p: D_lim[p]-DF_lim[p] for p in Persons}  # discretionary deductions limit
A0 = {
    'm': {'i': 20e5, 'g': 11.595e5},
    'p': {'i': 17e5, 'g': 2.25e5},
    'n': {'i': 0, 'g': 0}
}
r = {
    'm': {'i': [0.08887] * Y, 'g': [0.09844] * Y}, # g: # 9.5% p.a. compounded quarterly
    'p': {'i': [0.08626] * Y, 'g': [0.07978] * Y}, # g: # 7.75% p.a. compounded quarterly
    'n': {'i': [0.09308] * Y, 'g': [0.09308] * Y}, # 9% p.a. compounded quarterly
}
d = {p: [1] * Y for p in Persons}
rho = [round((1-0)**y, 4) for y in range(Y)]  # discounting factor for future tax liabilities

# define the LP problem
problem = pulp.LpProblem("Investment_Optimization", pulp.LpMinimize)

# define choice variables
f = {p: pulp.LpVariable(f'f_{p}', 0, 1, cat='Continuous') for p in ['m', 'p']}
# combined for y and person
DF = {p: pulp.LpVariable.dict(f'DF_{p}', range(Y), DF_lim[p], cat='Continuous') for p in Persons}
D = {p: pulp.LpVariable.dict(f'D_{p}', range(Y), 0, D_lim[p], cat='Continuous') for p in Persons}

# variable definitions
BS = {
    **{p: {y: 0 for y in range(Y)} for p in SeniorPersons},
    'n': {y: BS_c[y] for y in range(Y)},
}
SAL = {
    **{p: {y: 0 for y in range(Y)} for p in SeniorPersons},
    'n': {y: SAL_c[y] for y in range(Y)},
}
SR = {
    'm': {y: SR_c[y] for y in range(Y)},
    'p': {y: 0 for y in range(Y)},
    'n': {y: 0 for y in range(Y)},
}
HR = {
    p: {
        y: pulp.LpVariable(f'HR_{p}_{y}', lowBound=0, cat='Continuous')
        for y in range(Y)
    }
    for p in Persons
}
I = {
    p: {
        i: {
            y: pulp.LpVariable(f'I_{p}_{i}_{y}', lowBound=0, cat='Continuous')
            for y in range(Y)
        }
        for i in Investments
    }
    for p in Persons
}
A = {
    p: {
        i: {
            y: pulp.LpVariable(f'A_{p}_{i}_{y}', lowBound=0, cat='Continuous')
            for y in range(Y)
        }
        for i in Investments
    }
    for p in Persons
}
t = {
    p: {
        y: pulp.LpVariable(f't_{p}_{y}', lowBound=0, cat='Continuous')
        for y in range(Y)
    }
    for p in Persons
}
F = {
    p: {
        i: {
            y: pulp.LpVariable(f'F_{p}_{i}_{y}', cat='Continuous')
            for y in range(Y)
        }
        for i in Investments
    }
    for p in Persons
}  # F (new investment) can be negative
T = {
    p: {
        y: pulp.LpVariable(f'T_{p}_{y}', lowBound=0, cat='Continuous')
        for y in range(Y)
    }
    for p in Persons
}
E = {
    p: {
        y: pulp.LpVariable(f'E_{p}_{y}', lowBound=0, cat='Continuous')
        for y in range(Y)
    }
    for p in Persons
}
K = {
    p: {
        y: pulp.LpVariable(f'K_{p}_{y}', lowBound=0, cat='Continuous')
        for y in range(Y)
    }
    for p in Persons
}

# Binary variables to represent the tax brackets
b = {
    p: {
        y: {
            string.ascii_uppercase[cnt] : pulp.LpVariable(f'b_{p}_{y}_{cnt}', cat='Binary')
            for cnt in range(3)
        }
        for y in range(Y)
    }
    for p in Persons
}
tax_comp = {
    p: {
        y: {
            string.ascii_uppercase[cnt]: pulp.LpVariable(f'tax_comp_{p}_{y}_{cnt}', lowBound=0, cat='Continuous')
            for cnt in range(3)
        }
        for y in range(Y)
    }
    for p in Persons
}


# objective function (include rho as well)
problem += pulp.lpSum(rho[y] * t[p][y] for p in Persons for y in range(Y))


# initial values
for (p,i) in Person_Investments:
    problem += A[p][i][0] == A0[p][i]

# constraints
for y in range(Y):
    # income constraints
    for (p,i) in Person_Investments:
        problem += I[p][i][y] == r[p][i][y] * A[p][i][y]

    # wealth constraints
    if y > 0:
        for p in Persons:
            problem += A[p]['i'][y] == A[p]['i'][y - 1] + F[p]['i'][y]
            problem += A[p]['g'][y] == A[p]['g'][y - 1] + F[p]['g'][y] + I[p]['g'][y]

    # fixed and total deductions constraints
    for p in Persons:
        problem += D[p][y] >= DF[p][y]
        problem += D[p][y] <= DF[p][y] + DD_lim[p]

    # HR and BS constraints for m and p
    problem += HR['m'][y] == f['m'] * BS['n'][y]
    problem += HR['p'][y] == f['p'] * BS['n'][y]
    problem += f['m'] + f['p'] <= max_HR_fraction

    # total taxable income constraints
    for p in Persons:
        # including
        problem += T[p][y] == (
            SAL[p][y] + # salary
            SR[p][y] + # shop rental
            HR[p][y] + # home rental
            I[p]['i'][y] + # interest from interest-bearing debt
            I[p]['g'][y] - # interest from growth investment
            D[p][y] - # deductions
            0.3*(HR[p][y] + SR[p][y]) - # 30% of rental income
            (0.12+0.1)*BS[p][y] - # 12% in comp PF and 10% in comp NPS
            (f['m']+f['p'] - 0.1)*BS[p][y] # fraction of BS paid to m/p minus 10% of BS (only applies to n)
        )


    # tax constraints
    for p in Persons:
        # tax constraints for b
        problem += b[p][y]['A'] + b[p][y]['B'] + b[p][y]['C'] == 1
        # total income
        problem += T[p][y] <= 5e5 + (1 - b[p][y]['A']) * VERY_LARGE_NUM
        problem += T[p][y] >= (5e5 + VERY_SMALL_NUM) * b[p][y]['B']
        problem += T[p][y] <= 10e5 + (1 - b[p][y]['B']) * VERY_LARGE_NUM
        problem += T[p][y] >= (10e5 + VERY_SMALL_NUM) * b[p][y]['C']
        # tax component-1
        problem += tax_comp[p][y]['A'] <= 0
        # tax component-2
        problem += tax_comp[p][y]['B'] <= 1.04 * (0.2 * T[p][y] - 87500 * b[p][y]['B'])
        problem += tax_comp[p][y]['B'] <= 1.04 * (VERY_LARGE_NUM * b[p][y]['B'])
        problem += tax_comp[p][y]['B'] >= 1.04 * (0.2 * T[p][y] - 87500 - VERY_LARGE_NUM * (1 - b[p][y]['B']))
        # tax component-3
        problem += tax_comp[p][y]['C'] <= 1.04 * (0.3 * T[p][y] - 187500 * b[p][y]['C'])
        problem += tax_comp[p][y]['C'] <= 1.04 * (VERY_LARGE_NUM * b[p][y]['C'])
        problem += tax_comp[p][y]['C'] >= 1.04 * (0.3 * T[p][y] - 187500 - VERY_LARGE_NUM * (1 - b[p][y]['C']))
        # final tax liability
        problem += t[p][y] == tax_comp[p][y]['A'] + tax_comp[p][y]['B'] + tax_comp[p][y]['C']

    # expense constraints
    for p in SeniorPersons:
        problem += E[p][y] == SR[p][y] + I[p]['i'][0]  # expenses equal to SR and period-0 I
    problem += E['n'][y] == BS['n'][y] + HR['m'][y] + HR['p'][y]  # assumption: n's expenses equal to BS plus HR given to m and p

    # disposable income constraints
    for p in Persons:
        problem += K[p][y] == SAL[p][y] + SR[p][y] + HR[p][y] + I[p]['i'][y] + DF[p][y] - D[p][y] - E[p][y] - t[p][y]

    # investment constraints
    for p in Persons:
        problem += F[p]['g'][y] == d[p][y] * K[p][y]
        problem += F[p]['i'][y] == (1 - d[p][y]) * K[p][y]


# solve the problem
problem.solve()

# print combined useful values for each year
rounding = -2
frequency = 1 # 1 annual, 12 monthly
print(f"fraction:                              --->    'm':{round(f['m'].varValue, 4):.4f}, 'p':{round(f['p'].varValue, 4):.4f}")
for y in range(Y):
    print('-'*40 + f' Year: {y} ' + '-'*40)
    print(f"Year {y}: Total (p/m) Investment:              --->    {round(sum(F[p][i][y].varValue for p in SeniorPersons for i in Investments) / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (n)   Investment:              --->    {round(sum(F['n'][i][y].varValue for i in Investments) / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (p/m) Wealth:                  --->    {round(sum(A[p][i][y].varValue for p in SeniorPersons for i in Investments) / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (n)   Wealth:                  --->    {round(sum(A['n'][i][y].varValue for i in Investments) / frequency, rounding):,.0f}")
    # print(f"Year {y}: Total (p/m) Fixed Deductions:        --->    {round(sum(DF[p][y].varValue for p in SeniorPersons) / frequency, rounding):,.0f}")
    # print(f"Year {y}: Total (n)   Fixed Deductions:        --->    {round(DF['n'][y].varValue / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (p/m) Deductions:              --->    {round(sum(D[p][y].varValue for p in SeniorPersons) / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (n)   Deductions:              --->    {round(D['n'][y].varValue / frequency, rounding):,.0f}")
    # print(f"Year {y}: Total (p/m) SR Income:               --->    {round(sum(SR[p][y] for p in SeniorPersons) / frequency, rounding):,.0f}")
    # print(f"Year {y}: Total (p/m) HR Income:               --->    {round(sum(HR[p][y].varValue for p in SeniorPersons) / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (p/m) Expenses:                --->    {round(sum(E[p][y].varValue for p in SeniorPersons) / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (n)   Expenses:                --->    {round(E['n'][y].varValue / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (p/m) interest-bearing Income: --->    {round(sum(I[p]['i'][y].varValue for p in SeniorPersons) / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (n)   interest-bearing Income: --->    {round(I['n']['i'][y].varValue / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (p/m) growth Income:           --->    {round(sum(I[p]['g'][y].varValue for p in SeniorPersons) / frequency, rounding):,.0f}")
    print(f"Year {y}: Total (n)   growth Income:           --->    {round(I['n']['g'][y].varValue / frequency, rounding):,.0f}")
    print(f"Year {y}: Total Taxable Income:                --->    {round(sum(T[p][y].varValue for p in Persons) / frequency, rounding):,.0f}")
    print(f"Year {y}: Total Tax Liability:                 --->    {round(sum(t[p][y].varValue for p in Persons) / frequency, rounding):,.0f}")
    for p in Persons:
        print(f"Year {y}: {p}'s Taxable Income:                  --->    {round(T[p][y].varValue / frequency, rounding):,.0f}")
    for p in Persons:
        print(f"Year {y}: {p}'s Tax Liability:                   --->    {round(t[p][y].varValue / frequency, rounding):,.0f}")


