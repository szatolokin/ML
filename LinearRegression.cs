using System;
using System.IO;

using MathNet.Numerics.LinearAlgebra;

namespace ML
{
    namespace LinearRegression
    {
        using Data;

        public class Model
        {
            int degree;

            Matrix<double> parameters;

            public Model(string fileName)
            {
                LoadFromFile(fileName);
            }

            public Model(Data data, int deg = 3)
            {
                Training(data, deg);
            }

            public Model()
            {
                Data data = new Data();
                Training(data);
            }

            public void LoadFromFile(string fileName)
            {
                StreamReader file = new StreamReader(fileName);

                degree = Convert.ToInt32(file.ReadLine());

                parameters = Matrix<double>.Build.Dense(degree + 1, 1);

                for (int i = 0; i <= degree; ++i)
                {
                    parameters[i, 0] = Convert.ToDouble(file.ReadLine());
                }

                file.Close();
            }

            public void SaveToFile(string fileName)
            {
                StreamWriter file = new StreamWriter(fileName);

                file.WriteLine(degree);

                for (int i = 0; i <= degree; ++i)
                {
                    file.WriteLine(parameters[i, 0]);
                }

                file.Close();
            }

            public void Training(Data data, int deg = 3)
            {
                degree = deg;

                Matrix<double> A = Matrix<double>.Build.Dense(data.Size, degree + 1);

                for (int i = 0; i < data.Size; ++i)
                {
                    for (int j = 0; j <= degree; ++j)
                    {
                        A[i, j] = Math.Pow(data.X[i, 0], j);
                    }
                }

                parameters = A.TransposeThisAndMultiply(A).Inverse() * A.TransposeThisAndMultiply(data.Y);
            }

            public double Calc(double arg)
            {
                double res = 0;

                for (int i = 0; i <= degree; ++i)
                {
                    res += parameters[i, 0] * Math.Pow(arg, i);
                }

                return res;
            }

            double LossFunc(double val1, double val2)
            {
                return Math.Pow(val1 - val2, 2);
            }

            public double QualityFunc(Data data)
            {
                double res = 0;
                
                for (int i = 0; i < data.Size; ++i)
                {
                    res += LossFunc(Calc(data.X[i, 0]), data.Y[i, 0]);
                }

                return res / data.Size;
            }

            public double CrossValid(Data data, int breaks = 1000, int size = 2500)
            {
                double res = 0;

                for (int i = 0; i < breaks; ++i)
                {
                    Data[] split = data.Split(size);

                    Model model = new Model(split[0], degree);

                    res += model.QualityFunc(split[1]);
                }

                return res / breaks;
            }
        }
    }
}

/*
 * Model:
 * - создание модели из файла
 * - создание модели с обучением по данным
 * - создание модели с обучением по-умолчанию
 * - перезапись модели из файла
 * - повторное обучение модели по данным
 * - повторное обучение модели по умолчанию
 * - сохранение модели в файл
 * - вычисление приближенного значения на обученной модели
 * - вычисление функции потерь
 * - вычисление функционала качества на выборке
 * - вычисление скользящего контроля на выборке
*/
