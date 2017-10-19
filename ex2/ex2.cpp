#include <stdio.h>
#include <ctime>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;
using namespace std;

// количество строк в исходной квадратной матрице
const int MATRIX_SIZE = 1500;

void InitMatrix(double** matrix)
{
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		matrix[i] = new double[MATRIX_SIZE + 1];
	}

	for (int k = 0; k<MATRIX_SIZE; k++)
	{
		for (int j = 0; j <= MATRIX_SIZE; j++)
		{
			matrix[k][j] = rand() % 250 + 1;
		}
	}
}

double SerialGaussMethod(double **matrix, const int rows, double* result)
{
	int k;
	double koef;
	
	// прямой ход метода Гаусса
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (k = 0; k < rows; ++k)
	{
		//
		for (int i = k + 1; i < rows; ++i)
		{
			koef = -matrix[i][k] / matrix[k][k];
			for (int j = k; j <= rows; ++j)
				matrix[i][j] += koef * matrix[k][j];
		}
	}
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> duration = (t2 - t1);
	cout << "Время выполнения прямого хода для последовательного метода: " << duration.count() << " секунд" << endl;

	// обратный ход метода Гаусса
	result[rows - 1] = matrix[rows - 1][rows] / matrix[rows - 1][rows - 1];
	for (k = rows - 2; k >= 0; --k)
	{
		result[k] = matrix[k][rows];
		//
		for (int j = k + 1; j < rows; ++j)
		{
			result[k] -= matrix[k][j] * result[j];
		}
		result[k] /= matrix[k][k];
	}
	return duration.count();
}

double ParallelGaussMethod(double **matrix, const int rows, double* result)
{
	// прямой ход метода Гаусса
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int k = 0; k < rows; ++k)
	{
		cilk_for(int i = k + 1; i < rows; ++i)
		{
			double koef = -matrix[i][k] / matrix[k][k];
			for (int j = k; j <= rows; ++j)
				matrix[i][j] += koef * matrix[k][j];
		}
	}
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> duration = (t2 - t1);
	cout << "Время выполнения прямого хода для параллельного метода: " << duration.count() << " секунд" << endl;

	// обратный ход метода Гаусса
	cilk::reducer_opadd<double> summ(0.0);
	result[rows - 1] = matrix[rows - 1][rows] / matrix[rows - 1][rows - 1];
	for (int k = rows - 2; k >= 0; --k)
	{
		summ.set_value(matrix[k][rows]);
		cilk_for(int j = k + 1; j < rows; ++j)
		{
			summ += -matrix[k][j] * result[j];
		}
		result[k] = summ.get_value() / matrix[k][k];
	}
	return duration.count();
}

int main()
{
	setlocale(LC_ALL, "Russian");
	
	srand((unsigned)time(0));
	
	int i; double serial_time, parallel_time;

	/*
	const int test_matrix_lines = 4;
	double **test_matrix = new double*[test_matrix_lines];
	for (i = 0; i < (test_matrix_lines); ++i)
	{
	test_matrix[i] = new double[(test_matrix_lines + 1)];
	}
	// массив решений СЛАУ
	double *result = new double[test_matrix_lines];
	// инициализация тестовой матрицы
	test_matrix[0][0] = 2; test_matrix[0][1] = 5;  test_matrix[0][2] = 4;  test_matrix[0][3] = 1;  test_matrix[0][4] = 20;
	test_matrix[1][0] = 1; test_matrix[1][1] = 3;  test_matrix[1][2] = 2;  test_matrix[1][3] = 1;  test_matrix[1][4] = 11;
	test_matrix[2][0] = 2; test_matrix[2][1] = 10; test_matrix[2][2] = 9;  test_matrix[2][3] = 7;  test_matrix[2][4] = 40;
	test_matrix[3][0] = 3; test_matrix[3][1] = 8;  test_matrix[3][2] = 9;  test_matrix[3][3] = 2;  test_matrix[3][4] = 37;

	/*SerialGaussMethod(test_matrix, test_matrix_lines, result);

	printf("Solution:\n");
	for (i = 0; i < test_matrix_lines; ++i)
	printf("x(%d) = %lf\n", i, result[i]);
	*/
	
	// захват памяти для амтрицы и результатов
	double **matrix = new double*[MATRIX_SIZE];
	double *serial_res = new double[MATRIX_SIZE];
	double *parallel_res = new double[MATRIX_SIZE];
	
	InitMatrix(matrix); // инициализация матрицы 

	serial_time = SerialGaussMethod(matrix, MATRIX_SIZE, serial_res); // последовательная реализация
	parallel_time = ParallelGaussMethod(matrix, MATRIX_SIZE, parallel_res); // параллельная реализация
	
	// вывод результатов
	cout << endl << "Корень|Посл. вар.|Пар. вар." << endl;
	for (i = 0; i < MATRIX_SIZE; ++i)
		cout << "x[" << i << "] = " << serial_res[i] << " | " << parallel_res[i] << endl;
		
	cout << endl << "Полученное ускорение = " << serial_time / parallel_time << endl << endl; // вывод результата

	// освобождение памяти
	for (i = 0; i < MATRIX_SIZE; ++i)
		delete[]matrix[i];

	delete[] serial_res;
	delete[] parallel_res;

	return 0;
}