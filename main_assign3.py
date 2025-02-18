from assignment3.toronto_rental import prepare_toronto_rental_dataset
from assignment3.energy_efficiency import prepare_energy_efficiency_dataset
from assignment3.employee_salaries import prepare_employee_salaries_dataset
from assignment3.cars import prepare_cars_dataset
from assignment3.model.run_training import run_training_on_preprocessed_dataset

if __name__ == "__main__":
    train_test_data_map: dict = {
        'toronto_rental': list(prepare_toronto_rental_dataset()),
        'employee_salaries': list(prepare_employee_salaries_dataset()),
        'energy_efficiency': list(prepare_energy_efficiency_dataset()),
        'cars': list(prepare_cars_dataset())
    }
    for dataset in train_test_data_map.values():
        run_training_on_preprocessed_dataset(x_train=dataset[0],
                                             x_test=dataset[1],
                                             y_train=dataset[2],
                                             y_test=dataset[3])

