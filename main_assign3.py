from assignment3.toronto_rental import prepare_toronto_rental_dataset
from assignment3.energy_efficiency import prepare_energy_efficiency_dataset
from assignment3.employee_salaries import prepare_employee_salaries_dataset
from assignment3.cars import prepare_cars_dataset

if __name__ == "__main__":
    x_train_toronto, x_test_toronto, y_train_toronto, y_test_toronto = prepare_toronto_rental_dataset()
    x_train_salaries, x_test_salaries, y_train_salaries, y_test_salaries = prepare_employee_salaries_dataset()
    x_train_energy, x_test_energy, y_train_energy, y_test_energy = prepare_energy_efficiency_dataset()
    x_train_cars, x_test_cars, y_train_cars, y_test_cars = prepare_cars_dataset()

