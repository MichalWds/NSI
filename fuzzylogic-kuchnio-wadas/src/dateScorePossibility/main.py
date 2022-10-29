
##Authors: Karol Kuchnio s21912 and Michał Wadas s20495
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Antecednets (Inputs):
height = ctrl.Antecedent(np.arange(150, 201, 1), 'height')
appearance = ctrl.Antecedent(np.arange(0, 11, 1), 'appearance')
iq = ctrl.Antecedent(np.arange(100, 201, 1), 'iq')

# Consequents (Outputs):
successful_date_possibility = ctrl.Consequent(np.arange(0, 101, 1), 'successful_date_possibility')

# Auto-membership functions
height.automf(3, variable_type='quant')
appearance.automf(3, variable_type='quant')
iq.automf(3, variable_type='quant')

# Membership functions
successful_date_possibility['low'] = fuzz.trimf(successful_date_possibility.universe, [0, 0, 30])
successful_date_possibility['average'] = fuzz.trimf(successful_date_possibility.universe, [15, 45, 75])
successful_date_possibility['high'] = fuzz.trimf(successful_date_possibility.universe, [65, 100, 100])

# Membership views
height.view()
appearance.view()
iq.view()

# Defined rules
first_rule = ctrl.Rule(height['high'] | appearance['high'] | iq['high'], successful_date_possibility['high'])
second_rule = ctrl.Rule(height['average'] | appearance['average'] | iq['average'], successful_date_possibility['average'])
third_rule = ctrl.Rule(height['low'] | appearance['low'] | iq['low'], successful_date_possibility['low'])

# control system
resul_ctrl = ctrl.ControlSystem([
    first_rule, second_rule, third_rule
])

if __name__ == '__main__':

    is_done = True

    while is_done:
        val1 = int(input("Write your height (from 150 to 200): "))
        val2 = int(input("Write your appearance (from 1 to 10): "))
        val3 = int(input("Write you iq (from 100 to 200): "))

        if (200 >= val1 >= 150) and (10 >= val2 >= 0) and (200 >= val3 >= 100):
            is_done = False

            result = ctrl.ControlSystemSimulation(resul_ctrl)
            result.input['height'] = int(val1)
            result.input['appearance'] = int(val2)
            result.input['iq'] = int(val3)

            # Crunch the numbers
            result.compute()

            print("Your success possibility on the date is:", result.output['successful_date_possibility'])
            successful_date_possibility.view(result)

            plt.show()
